import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from .llm_agent import EntityStruct
except ImportError:
    from llm_agent import EntityStruct


@dataclass
class FrameInfo:

    frame_id: str
    frame_path: str
    prompt_id: int
    associated_entities: List[str]
    score: float
    entity_score: float = 0.0
    visual_score: Optional[float] = None
    pixel_frame: Optional[torch.Tensor] = (
        None
    )
    kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None

    def to_dict(self) -> Dict:
        return {
            "frame_path": self.frame_path,
            "prompt_id": self.prompt_id,
            "associated_entities": self.associated_entities,
            "score": self.score,
            "entity_score": self.entity_score,
            "visual_score": self.visual_score,
        }


class MemoryBank:

    def __init__(
        self,
        text_encoder=None,
        max_memory_frames: int = 3,
        max_id_memory_frames: int = 4,
        min_id_memory_frames_multi_entity: int = 2,
        memory_allocation_mode: str = "dynamic",
        fixed_id_memory_frames: Optional[int] = None,
        selection_mode: str = "entity",
        ablation_mode: str = "full",
        frame_selection_score_mode: str = "fused",
        frame_seq_length: int = 1560,
        num_transformer_blocks: int = 30,
        save_dir: str = "data",
        save_frames_to_disk: bool = False,
        **kwargs,
    ):
        self.text_encoder = text_encoder
        self.max_memory_frames = max_memory_frames
        if memory_allocation_mode not in {"dynamic", "fixed"}:
            raise ValueError(
                f"Unsupported memory_allocation_mode: {memory_allocation_mode}"
            )
        self.memory_allocation_mode = memory_allocation_mode
        if fixed_id_memory_frames is None:
            fixed_id_memory_frames = max_id_memory_frames
        self.fixed_id_memory_frames = max(1, int(fixed_id_memory_frames))
        self.max_id_memory_frames = (
            self.fixed_id_memory_frames
            if self.memory_allocation_mode == "fixed"
            else max_id_memory_frames
        )
        self.min_id_memory_frames_multi_entity = max(
            1,
            min(min_id_memory_frames_multi_entity, self.max_id_memory_frames),
        )
        if selection_mode not in {"entity", "prompt"}:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")
        self.selection_mode = selection_mode
        self.ablation_mode = ablation_mode
        valid_score_modes = {"fused", "random", "semantic_only", "visual_only"}
        if frame_selection_score_mode not in valid_score_modes:
            raise ValueError(
                f"Unsupported frame_selection_score_mode: {frame_selection_score_mode}"
            )
        self.frame_selection_score_mode = frame_selection_score_mode
        self.frame_seq_length = frame_seq_length
        self.num_transformer_blocks = num_transformer_blocks
        self.save_dir = save_dir
        self.save_frames_to_disk = save_frames_to_disk

        self.global_registry: Dict[str, Dict] = {}
        self.frame_archive: Dict[str, FrameInfo] = {}

        self.active_memory: List[str] = []

        self.id_memory: List[str] = []

        self._frame_kv_store: Dict[str, List[Dict[str, torch.Tensor]]] = {}

        self._memory_kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None
        self._memory_kv_cache_key: Optional[Tuple[str, ...]] = None
        self._memory_kv_cache_device: Optional[torch.device] = None

        # Per-prompt caches for entity weights and q_agg
        self._entity_weights_cache: Dict[Any, torch.Tensor] = {}
        self._q_agg_cache: Dict[int, torch.Tensor] = {}  # layer_idx -> q_agg

        os.makedirs(save_dir, exist_ok=True)

    @property
    def frame_active_memory(self) -> List[str]:
        if self.active_memory:
            return list(dict.fromkeys(self.active_memory))
        return list(dict.fromkeys(self.id_memory))

    @frame_active_memory.setter
    def frame_active_memory(self, value: List[str]):
        normalized = list(dict.fromkeys(value))
        self.id_memory = list(normalized)
        self._set_active_memory(normalized)

    def _set_active_memory(self, frame_ids: List[str]) -> None:
        self.active_memory = list(dict.fromkeys(frame_ids))
        self._memory_kv_cache = None
        self._memory_kv_cache_key = None
        self._memory_kv_cache_device = None


    def register_entities(
        self,
        entities: List[EntityStruct],
        prompt_id: int,
        registry_update: Optional[Dict] = None,
    ) -> None:
        if registry_update:
            for gid, info in registry_update.items():
                if info.get("action") == "create":
                    self.global_registry[gid] = {
                        "name": info["name"],
                        "all_entities": info["all_entities"],
                        "all_attrs": info["all_attrs"],
                        "instances": info["instances"],
                    }
                elif info.get("action") == "update":
                    if gid in self.global_registry:
                        reg = self.global_registry[gid]
                        new_entity = info["new_entity"]
                        new_attrs = info["new_attrs"]

                        if new_entity not in reg["all_entities"]:
                            reg["all_entities"].append(new_entity)
                        for attr in new_attrs:
                            if attr not in reg["all_attrs"]:
                                reg["all_attrs"].append(attr)
                        reg["instances"].append(
                            {
                                "prompt_id": info["prompt_id"],
                                "entity": new_entity,
                                "attrs": new_attrs,
                            }
                        )
        else:
            for entity in entities:
                gid = str(entity.global_id)
                if gid not in self.global_registry:
                    entity_type = self._infer_entity_type(entity.entity)
                    type_count = sum(
                        1
                        for v in self.global_registry.values()
                        if v.get("name", "").startswith(entity_type)
                    )
                    self.global_registry[gid] = {
                        "name": f"{entity_type}_{type_count + 1}",
                        "all_entities": [entity.entity],
                        "all_attrs": entity.attrs.copy(),
                        "instances": [
                            {
                                "prompt_id": prompt_id,
                                "entity": entity.entity,
                                "attrs": entity.attrs.copy(),
                            }
                        ],
                    }
                else:
                    reg = self.global_registry[gid]
                    if entity.entity not in reg["all_entities"]:
                        reg["all_entities"].append(entity.entity)
                    for attr in entity.attrs:
                        if attr not in reg["all_attrs"]:
                            reg["all_attrs"].append(attr)
                    reg["instances"].append(
                        {
                            "prompt_id": prompt_id,
                            "entity": entity.entity,
                            "attrs": entity.attrs.copy(),
                        }
                    )

    def _infer_entity_type(self, entity_name: str) -> str:
        entity_lower = entity_name.lower()
        if any(w in entity_lower for w in ["woman", "girl", "lady", "female", "she"]):
            return "woman"
        elif any(
            w in entity_lower
            for w in ["man", "boy", "guy", "male", "he", "protagonist"]
        ):
            return "man"
        else:
            return "person"


    def _compute_dynamic_id_budget(self, required_entity_ids: List[str]) -> int:
        if not required_entity_ids or not self.frame_archive:
            return 0

        uncovered = set(required_entity_ids)
        budget = 0
        used_frames = set()

        while uncovered and budget < self.max_id_memory_frames:
            best_fid = None
            best_cover = 0
            best_score = -float("inf")

            for fid, fi in self.frame_archive.items():
                if fid in used_frames:
                    continue
                cover = len(set(fi.associated_entities) & uncovered)
                if cover > best_cover or (
                    cover == best_cover and fi.entity_score > best_score
                ):
                    best_fid = fid
                    best_cover = cover
                    best_score = fi.entity_score

            if best_fid is None or best_cover == 0:
                break

            used_frames.add(best_fid)
            uncovered -= set(self.frame_archive[best_fid].associated_entities)
            budget += 1

        return budget

    def _greedy_select_id_frames(
        self, required_entity_ids: List[str], budget: int
    ) -> List[str]:
        if not required_entity_ids or not self.frame_archive or budget <= 0:
            return []

        uncovered = set(required_entity_ids)
        selected = []

        while uncovered and len(selected) < budget:
            best_fid = None
            best_cover = 0
            best_score = -float("inf")

            for fid, fi in self.frame_archive.items():
                if fid in selected:
                    continue
                cover = len(set(fi.associated_entities) & uncovered)
                if cover > best_cover or (
                    cover == best_cover and fi.entity_score > best_score
                ):
                    best_fid = fid
                    best_cover = cover
                    best_score = fi.entity_score

            if best_fid is None or best_cover == 0:
                break

            selected.append(best_fid)
            uncovered -= set(self.frame_archive[best_fid].associated_entities)

        if len(selected) < budget:
            remaining = [
                (fid, fi.entity_score)
                for fid, fi in self.frame_archive.items()
                if fid not in selected
            ]
            remaining.sort(key=lambda x: x[1], reverse=True)
            for fid, _ in remaining:
                if len(selected) >= budget:
                    break
                selected.append(fid)

        return selected

    def _resolve_id_memory_budget(self, required_entity_ids: List[str]) -> int:
        if self.memory_allocation_mode == "fixed":
            return self.fixed_id_memory_frames

        budget = self._compute_dynamic_id_budget(required_entity_ids)

        if len(set(required_entity_ids)) >= 2 and len(self.frame_archive) >= 2:
            budget = max(budget, self.min_id_memory_frames_multi_entity)

        return min(budget, self.max_id_memory_frames)

    def retrieve_initial_frames(self, entity_ids: List[int]) -> List[str]:
        if not self.frame_archive:
            return []

        entity_id_strs = [str(eid) for eid in entity_ids]
        if self.selection_mode == "entity":
            if entity_id_strs:
                budget = self._resolve_id_memory_budget(entity_id_strs)
                self.id_memory = self._greedy_select_id_frames(entity_id_strs, budget)
                self._set_active_memory(self.id_memory)
            else:
                self.id_memory = []
                self._set_active_memory([])
        else:
            self.id_memory = []
            ranked = sorted(
                self.frame_archive.items(),
                key=lambda item: (-item[1].score, self._frame_sort_key(item[0])),
            )
            selected = [fid for fid, _ in ranked[: self.max_memory_frames]]
            self._set_active_memory(selected)

        return self.frame_active_memory


    def select_frame_from_chunk(
        self,
        evicted_chunk_kv: List[Dict[str, torch.Tensor]],
        crossattn_cache: List[Dict[str, torch.Tensor]],
        prompt_id: int,
        chunk_id: int,
        current_entity_ids: List[int],
        current_entities: Optional[List["EntityStruct"]] = None,
        prompt_text: Optional[str] = None,
        visual_scores: Optional[Dict[int, float]] = None,
        pixel_frames: Optional[torch.Tensor] = None,
        visual_weight: float = 0.3,
    ) -> Tuple[str, float]:
        if not evicted_chunk_kv or not crossattn_cache:
            raise ValueError("evicted_chunk_kv and crossattn_cache must not be empty")

        num_candidate_frames = max(
            1, evicted_chunk_kv[0]["k"].shape[1] // self.frame_seq_length
        )
        available_layers = min(len(evicted_chunk_kv), len(crossattn_cache))
        has_initialized_layer = any(
            crossattn_cache[layer_idx].get("is_init", False)
            for layer_idx in range(available_layers)
        )

        device = evicted_chunk_kv[0]["k"].device
        dtype = evicted_chunk_kv[0]["k"].dtype

        score_mode = getattr(self, "frame_selection_score_mode", "fused")
        if score_mode == "random":
            entity_scores = torch.zeros(
                num_candidate_frames, device=device, dtype=dtype
            )
            fused_scores = torch.zeros(num_candidate_frames, device=device, dtype=dtype)
            best_frame_idx = torch.randint(
                num_candidate_frames, (1,), device=device
            ).item()
        else:
            if score_mode == "visual_only":
                entity_scores = torch.zeros(
                    num_candidate_frames, device=device, dtype=dtype
                )
            elif not has_initialized_layer:
                import warnings

                warnings.warn(
                    "[MemoryBank] crossattn_cache not initialized, selecting first frame by default"
                )
                entity_scores = torch.ones(
                    num_candidate_frames, device=device, dtype=dtype
                )
            else:
                num_text_tokens = crossattn_cache[0]["k"].shape[1]

                if self.selection_mode == "prompt":
                    cache_key = ("prompt", prompt_text or "", num_text_tokens)
                    if cache_key in self._entity_weights_cache:
                        entity_weights = self._entity_weights_cache[cache_key]
                    else:
                        entity_weights = self._build_prompt_token_weights(
                            num_text_tokens, prompt_text
                        )
                        self._entity_weights_cache[cache_key] = entity_weights
                        self._q_agg_cache.clear()
                else:
                    cache_key = (
                        "entity",
                        prompt_text or "",
                        tuple(sorted(str(eid) for eid in current_entity_ids)),
                        num_text_tokens,
                    )
                    if cache_key in self._entity_weights_cache:
                        entity_weights = self._entity_weights_cache[cache_key]
                    else:
                        entity_weights = self._build_entity_token_weights(
                            current_entities, num_text_tokens, prompt_text
                        )
                        self._entity_weights_cache[cache_key] = entity_weights
                        # Invalidate q_agg cache when weights change
                        self._q_agg_cache.clear()
                entity_scores = self._consensus_score(
                    evicted_chunk_kv,
                    crossattn_cache,
                    entity_weights,
                    available_layers,
                    num_candidate_frames,
                )

            if score_mode == "semantic_only":
                fused_scores = entity_scores
            elif score_mode == "visual_only":
                fused_scores = self._visual_scores_tensor(
                    visual_scores, num_candidate_frames, device, dtype
                )
            else:
                fused_scores = self._fuse_scores(
                    entity_scores, visual_scores, num_candidate_frames, visual_weight
                )

            best_frame_idx = fused_scores.argmax().item()

        best_score = fused_scores[best_frame_idx].item()
        best_entity_score = entity_scores[best_frame_idx].item()
        best_visual_score = visual_scores.get(best_frame_idx) if visual_scores else None

        frame_id = f"p{prompt_id}_c{chunk_id}_f{best_frame_idx}"

        frame_kv = self._extract_frame_kv_all_blocks(evicted_chunk_kv, best_frame_idx)

        selected_pixel = None
        if pixel_frames is not None:
            # pixel_frames: [T, C, H, W] where T=12 pixel frames
            # best_frame_idx is latent frame index (0-2), each latent frame = 4 pixel frames
            pixel_start = best_frame_idx * 4
            pixel_end = pixel_start + 4
            if pixel_end <= pixel_frames.shape[0]:
                selected_pixel = pixel_frames[pixel_start]

        frame_info = FrameInfo(
            frame_id=frame_id,
            frame_path=os.path.join(self.save_dir, f"{frame_id}.pt"),
            prompt_id=prompt_id,
            associated_entities=(
                list(dict.fromkeys(str(eid) for eid in current_entity_ids))
                if self.selection_mode == "entity"
                else []
            ),
            score=best_score,
            entity_score=best_entity_score,
            visual_score=best_visual_score,
            pixel_frame=selected_pixel,
            kv_cache=frame_kv,
        )

        self.frame_archive[frame_id] = frame_info
        self._frame_kv_store[frame_id] = frame_kv

        self._save_frame_kv(frame_id, frame_kv)

        return frame_id, best_score

    def _visual_scores_tensor(
        self,
        visual_scores: Optional[Dict[int, float]],
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        visual_tensor = torch.full((num_frames,), 0.5, device=device, dtype=dtype)
        if visual_scores is None:
            return visual_tensor
        for idx, score in visual_scores.items():
            if 0 <= idx < num_frames:
                visual_tensor[idx] = score
        return visual_tensor

    def _fuse_scores(
        self,
        text_scores: torch.Tensor,
        visual_scores: Optional[Dict[int, float]],
        num_frames: int,
        visual_weight: float = 0.3,
    ) -> torch.Tensor:
        if visual_scores is None:
            return text_scores

        ts_min = text_scores.min()
        ts_max = text_scores.max()
        if ts_max - ts_min > 1e-8:
            text_norm = (text_scores - ts_min) / (ts_max - ts_min)
        else:
            text_norm = torch.ones_like(text_scores) * 0.5

        visual_tensor = self._visual_scores_tensor(
            visual_scores, num_frames, text_scores.device, text_scores.dtype
        )

        fused = (1.0 - visual_weight) * text_norm + visual_weight * visual_tensor
        return fused

    def apply_attribute_corrections(self, prompt_id: int, corrections: Dict) -> None:
        del prompt_id

        if not isinstance(corrections, dict):
            return

        for gid, corrected in corrections.items():
            if not isinstance(corrected, dict):
                continue

            gid_str = str(gid)
            if gid_str in self.global_registry:
                corrected_attrs = corrected.get("corrected_attrs", [])
                if not isinstance(corrected_attrs, list):
                    continue

                normalized_attrs = [
                    str(attr) for attr in corrected_attrs if str(attr).strip()
                ]
                if normalized_attrs:
                    self.global_registry[gid_str]["all_attrs"] = normalized_attrs
                    print(
                        f"[VLM] Corrected attrs for entity {gid_str}: {normalized_attrs}"
                    )

    def _consensus_score(
        self,
        evicted_chunk_kv: List[Dict[str, torch.Tensor]],
        crossattn_cache: List[Dict[str, torch.Tensor]],
        token_weights: torch.Tensor,
        available_layers: int,
        num_candidate_frames: int,
    ) -> torch.Tensor:
        device = evicted_chunk_kv[0]["k"].device
        dtype = evicted_chunk_kv[0]["k"].dtype

        valid_layers = []
        valid_weights = []
        for layer_idx, layer_weight in zip(
            self.CONSENSUS_LAYERS, self.CONSENSUS_WEIGHTS
        ):
            if layer_idx < available_layers and crossattn_cache[layer_idx].get(
                "is_init", False
            ):
                valid_layers.append(layer_idx)
                valid_weights.append(layer_weight)

        if not valid_layers:
            fallback_layer = 0
            for layer_idx in range(available_layers):
                if crossattn_cache[layer_idx].get("is_init", False):
                    fallback_layer = layer_idx
                    break
            valid_layers = [fallback_layer]
            valid_weights = [1.0]

        weight_sum = sum(valid_weights)
        if weight_sum <= 0:
            valid_weights = [1.0 / len(valid_weights)] * len(valid_weights)
        else:
            valid_weights = [w / weight_sum for w in valid_weights]

        frame_scores = None
        for layer_idx, layer_weight in zip(valid_layers, valid_weights):
            layer_scores = self._compute_frame_scores_fast(
                evicted_chunk_kv[layer_idx],
                crossattn_cache[layer_idx],
                token_weights,
                layer_idx=layer_idx,
            )
            # Skip normalization when single layer (no cross-layer calibration needed)
            if len(valid_layers) > 1:
                std = layer_scores.std()
                if std > 1e-8:
                    layer_scores = (layer_scores - layer_scores.mean()) / std
            if frame_scores is None:
                frame_scores = torch.zeros_like(layer_scores)
            frame_scores = frame_scores + layer_scores * layer_weight

        if frame_scores is None:
            frame_scores = torch.ones(num_candidate_frames, device=device, dtype=dtype)

        return frame_scores

    def _compute_frame_scores_with_crossattn(
        self,
        chunk_kv: Dict[str, torch.Tensor],
        crossattn_cache_block: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        chunk_k = chunk_kv["k"]  # [B, L, H, D]
        text_q = crossattn_cache_block["k"]  # [B, 512, H, D]

        B, L, H, D = chunk_k.shape

        num_frames = L // self.frame_seq_length
        if num_frames == 0:
            return torch.tensor([1.0], device=chunk_k.device, dtype=chunk_k.dtype)

        # q_reshaped: [B*H, 512, D]
        # k_reshaped: [B*H, L, D]
        q_reshaped = text_q.permute(0, 2, 1, 3).reshape(B * H, -1, D)
        k_reshaped = chunk_k.permute(0, 2, 1, 3).reshape(B * H, L, D)

        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * (D**-0.5)

        scores_per_token = attn_scores.mean(dim=1)  # [B*H, L]
        scores_per_token = scores_per_token.view(B, H, L).mean(dim=1)  # [B, L]

        frame_scores = []
        for i in range(num_frames):
            start = i * self.frame_seq_length
            end = (i + 1) * self.frame_seq_length
            frame_score = scores_per_token[:, start:end].mean()  # scalar
            frame_scores.append(frame_score)

        return torch.tensor(frame_scores, device=chunk_k.device, dtype=chunk_k.dtype)

    def _compute_frame_scores_with_entity_focus(
        self,
        chunk_kv: Dict[str, torch.Tensor],
        crossattn_cache_block: Dict[str, torch.Tensor],
        entities: Optional[List["EntityStruct"]] = None,
        prompt_text: Optional[str] = None,
    ) -> torch.Tensor:
        chunk_k = chunk_kv["k"]  # [B, L, H, D]
        text_q = crossattn_cache_block["k"]  # [B, 512, H, D]

        B, L, H, D = chunk_k.shape
        num_text_tokens = text_q.shape[1]  # 512

        num_frames = L // self.frame_seq_length
        if num_frames == 0:
            return torch.tensor([1.0], device=chunk_k.device, dtype=chunk_k.dtype)

        # q_reshaped: [B*H, 512, D]
        # k_reshaped: [B*H, L, D]
        q_reshaped = text_q.permute(0, 2, 1, 3).reshape(B * H, -1, D)
        k_reshaped = chunk_k.permute(0, 2, 1, 3).reshape(B * H, L, D)

        # attn_scores: [B*H, 512, L]
        attn_scores = torch.bmm(q_reshaped, k_reshaped.transpose(1, 2)) * (D**-0.5)

        entity_weights = self._build_entity_token_weights(
            entities, num_text_tokens, prompt_text
        )
        entity_weights = entity_weights.to(
            device=chunk_k.device, dtype=chunk_k.dtype
        )  # [512]

        # attn_scores: [B*H, 512, L]
        # entity_weights: [512] -> [1, 512, 1]
        weights = entity_weights.view(1, -1, 1)  # [1, 512, 1]

        weighted_scores = attn_scores * weights  # [B*H, 512, L]

        # scores_per_position: [B*H, L]
        scores_per_position = weighted_scores.sum(dim=1) / (weights.sum() + 1e-8)

        scores_per_position = scores_per_position.view(B, H, L).mean(dim=1)

        frame_scores = []
        for i in range(num_frames):
            start = i * self.frame_seq_length
            end = (i + 1) * self.frame_seq_length
            frame_score = scores_per_position[:, start:end].mean()  # scalar
            frame_scores.append(frame_score)

        return torch.tensor(frame_scores, device=chunk_k.device, dtype=chunk_k.dtype)


    CONSENSUS_LAYERS = [0]
    CONSENSUS_WEIGHTS = [1.0]

    def _compute_frame_scores_fast(
        self,
        chunk_kv: Dict[str, torch.Tensor],
        crossattn_cache_block: Dict[str, torch.Tensor],
        entity_weights: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        chunk_k = chunk_kv["k"]  # [B, L, H, D]
        B, L, H, D = chunk_k.shape

        num_frames = L // self.frame_seq_length
        if num_frames == 0:
            return torch.ones(1, device=chunk_k.device, dtype=chunk_k.dtype)

        valid_length = num_frames * self.frame_seq_length
        if valid_length != L:
            chunk_k = chunk_k[:, :valid_length]

        if layer_idx not in self._q_agg_cache:
            text_q = crossattn_cache_block["k"]  # [B, S, H, D]
            w = entity_weights.to(device=text_q.device, dtype=text_q.dtype)  # [S]
            w = w / (w.sum() + 1e-8)
            self._q_agg_cache[layer_idx] = torch.einsum(
                "bshd,s->bhd", text_q, w
            )  # [B, H, D]
        q_agg = self._q_agg_cache[layer_idx]

        k_frames = chunk_k.reshape(B, num_frames, self.frame_seq_length, H, D)
        k_agg = k_frames.mean(dim=2)  # [B, F, H, D]

        scores = torch.einsum("bhd,bfhd->bhf", q_agg, k_agg) * (D**-0.5)

        # Step 4: mean over batch and heads → [F]
        scores = scores.mean(dim=(0, 1))  # [F]
        return scores

    def _build_entity_token_weights(
        self,
        entities: Optional[List["EntityStruct"]],
        num_tokens: int,
        prompt_text: Optional[str] = None,
    ) -> torch.Tensor:
        weights = torch.ones(num_tokens)

        if entities is None or len(entities) == 0:
            return weights

        if prompt_text is None or len(prompt_text) == 0:
            entity_start = int(num_tokens * 0.10)
            entity_end = int(num_tokens * 0.85)
            weights[entity_start:entity_end] = 1.5
            return weights

        prompt_lower = prompt_text.lower()
        prompt_len = len(prompt_text)

        keyword_positions = []  # [(start_ratio, end_ratio), ...]

        for entity in entities:
            entity_lower = entity.entity.lower()
            pos = prompt_lower.find(entity_lower)
            if pos != -1:
                start_ratio = pos / prompt_len
                end_ratio = (pos + len(entity_lower)) / prompt_len
                keyword_positions.append((start_ratio, end_ratio))

            for attr in entity.attrs:
                attr_lower = attr.lower()
                pos = prompt_lower.find(attr_lower)
                if pos != -1:
                    start_ratio = pos / prompt_len
                    end_ratio = (pos + len(attr_lower)) / prompt_len
                    keyword_positions.append((start_ratio, end_ratio))

        if not keyword_positions:
            entity_start = int(num_tokens * 0.10)
            entity_end = int(num_tokens * 0.85)
            weights[entity_start:entity_end] = 1.5
            return weights

        base_weight = 1.0
        entity_weight = 2.5

        for start_ratio, end_ratio in keyword_positions:
            start_token = max(0, int((start_ratio - 0.02) * num_tokens))
            end_token = min(num_tokens, int((end_ratio + 0.02) * num_tokens))
            weights[start_token:end_token] = torch.clamp(
                weights[start_token:end_token], min=entity_weight
            )

        scene_end = int(num_tokens * 0.08)
        camera_start = int(num_tokens * 0.92)

        head_mask = weights[:scene_end] == base_weight
        weights[:scene_end] = torch.where(
            head_mask, torch.tensor(0.7), weights[:scene_end]
        )

        tail_mask = weights[camera_start:] == base_weight
        weights[camera_start:] = torch.where(
            tail_mask, torch.tensor(0.5), weights[camera_start:]
        )

        return weights

    def _build_prompt_token_weights(
        self, num_tokens: int, prompt_text: Optional[str] = None
    ) -> torch.Tensor:
        del prompt_text

        weights = torch.ones(num_tokens)
        if num_tokens <= 4:
            return weights

        body_start = int(num_tokens * 0.08)
        body_end = int(num_tokens * 0.92)
        if body_start < body_end:
            weights[body_start:body_end] = 1.2
        if body_start > 0:
            weights[:body_start] = 0.7
        if body_end < num_tokens:
            weights[body_end:] = 0.5
        return weights

    def _extract_frame_kv_all_blocks(
        self, all_blocks_kv: List[Dict[str, torch.Tensor]], frame_idx: int
    ) -> List[Dict[str, torch.Tensor]]:
        start = frame_idx * self.frame_seq_length
        end = (frame_idx + 1) * self.frame_seq_length

        k_views = [bkv["k"][:, start:end] for bkv in all_blocks_kv]
        v_views = [bkv["v"][:, start:end] for bkv in all_blocks_kv]

        k_cpu = torch.stack(k_views).cpu()  # [num_blocks, B, L, H, D]
        v_cpu = torch.stack(v_views).cpu()

        return [{"k": k_cpu[i], "v": v_cpu[i]} for i in range(k_cpu.shape[0])]

    def _save_frame_kv(
        self, frame_id: str, frame_kv: List[Dict[str, torch.Tensor]]
    ) -> None:
        if not self.save_frames_to_disk:
            return

        path = os.path.join(self.save_dir, f"{frame_id}.pt")
        torch.save(frame_kv, path)

    def _load_frame_kv(self, frame_id: str) -> Optional[List[Dict[str, torch.Tensor]]]:
        if frame_id in self._frame_kv_store:
            return self._frame_kv_store[frame_id]

        path = os.path.join(self.save_dir, f"{frame_id}.pt")
        if os.path.exists(path):
            kv = torch.load(path, weights_only=False, map_location="cpu")
            self._frame_kv_store[frame_id] = kv
            return kv
        return None


    def update_active_memory(self, frame_id: str, score: float) -> None:
        if frame_id not in self.frame_archive:
            return

        current_memory = self.frame_active_memory.copy()

        if len(current_memory) < self.max_memory_frames:
            if frame_id not in current_memory:
                current_memory.append(frame_id)
                self._set_active_memory(current_memory)
        else:
            min_score = float("inf")
            min_idx = -1
            min_fid = None

            for idx, fid in enumerate(current_memory):
                if fid in self.frame_archive:
                    finfo = self.frame_archive[fid]
                    if finfo.score < min_score:
                        min_score = finfo.score
                        min_idx = idx
                        min_fid = fid

            if score > min_score and min_idx >= 0:
                current_memory[min_idx] = frame_id
                self._set_active_memory(current_memory)

    def update_id_memory(self, frame_id: str, entity_score: float) -> None:
        if frame_id not in self.frame_archive:
            return

        if len(self.id_memory) < self.max_id_memory_frames:
            if frame_id not in self.id_memory:
                self.id_memory.append(frame_id)
                self._set_active_memory(self.id_memory)
        else:
            min_score = float("inf")
            min_idx = -1
            min_fid = None

            for idx, fid in enumerate(self.id_memory):
                if fid in self.frame_archive:
                    finfo = self.frame_archive[fid]
                    if finfo.entity_score < min_score:
                        min_score = finfo.entity_score
                        min_idx = idx
                        min_fid = fid

            if entity_score > min_score and min_idx >= 0:
                self.id_memory[min_idx] = frame_id
                self._set_active_memory(self.id_memory)

    def get_memory_kv(
        self, device: torch.device = None
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        if not self.frame_active_memory:
            return None

        frame_ids = sorted(self.frame_active_memory, key=self._frame_sort_key)
        cache_key = tuple(frame_ids)

        if (
            self._memory_kv_cache is not None
            and self._memory_kv_cache_key == cache_key
            and self._memory_kv_cache_device == device
        ):
            return self._memory_kv_cache

        all_frames_kv = []
        for frame_id in frame_ids:
            kv = self._load_frame_kv(frame_id)
            if kv is not None:
                all_frames_kv.append(kv)

        if not all_frames_kv:
            return None

        # all_frames_kv: List[List[Dict]] - [num_frames, num_blocks, {"k", "v"}]
        num_blocks = len(all_frames_kv[0])
        result = []

        for block_idx in range(num_blocks):
            k_list = []
            v_list = []
            for frame_kv in all_frames_kv:
                k = frame_kv[block_idx]["k"]
                v = frame_kv[block_idx]["v"]
                if device is not None:
                    k = k.to(device)
                    v = v.to(device)
                k_list.append(k)
                v_list.append(v)

            result.append(
                {"k": torch.cat(k_list, dim=1), "v": torch.cat(v_list, dim=1)}
            )

        self._memory_kv_cache = result
        self._memory_kv_cache_key = cache_key
        self._memory_kv_cache_device = device

        return result

    @staticmethod
    def _frame_sort_key(frame_id: str) -> Tuple[int, int, int, str]:
        match = re.match(r"p(\d+)_c(\d+)_f(\d+)", frame_id)
        if not match:
            return (1 << 30, 1 << 30, 1 << 30, frame_id)
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)), frame_id)

    def get_active_frame_count(self) -> int:
        return len(self.frame_active_memory)


    def save_to_json(self, path: str) -> None:
        data = {
            "ablation_mode": self.ablation_mode,
            "selection_mode": self.selection_mode,
            "frame_selection_score_mode": self.frame_selection_score_mode,
            "global_registry": self.global_registry,
            "frame_archive": {
                fid: finfo.to_dict() for fid, finfo in self.frame_archive.items()
            },
            "frame_active_memory": self.frame_active_memory,
            "active_memory": self.active_memory,
            "id_memory": self.id_memory,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_from_json(self, path: str) -> None:
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.ablation_mode = data.get("ablation_mode", self.ablation_mode)
        self.selection_mode = data.get("selection_mode", self.selection_mode)
        self.frame_selection_score_mode = data.get(
            "frame_selection_score_mode", self.frame_selection_score_mode
        )
        self.global_registry = data.get("global_registry", {})
        active_memory = data.get("active_memory", data.get("frame_active_memory", []))
        self._set_active_memory(active_memory)
        if "id_memory" in data:
            self.id_memory = data["id_memory"]
        elif self.selection_mode == "entity":
            self.id_memory = list(active_memory)
        else:
            self.id_memory = []

        self.frame_archive = {}
        for fid, finfo_dict in data.get("frame_archive", {}).items():
            self.frame_archive[fid] = FrameInfo(
                frame_id=fid,
                frame_path=finfo_dict.get("frame_path", ""),
                prompt_id=finfo_dict.get("prompt_id", 0),
                associated_entities=finfo_dict.get("associated_entities", []),
                score=finfo_dict.get("score", 0.0),
                entity_score=finfo_dict.get("entity_score", 0.0),
                visual_score=finfo_dict.get("visual_score"),
            )

    def clear(self) -> None:
        self.global_registry = {}
        self.frame_archive = {}
        self.active_memory = []
        self.id_memory = []
        self._frame_kv_store = {}
        self._memory_kv_cache = None
        self._memory_kv_cache_key = None
        self._memory_kv_cache_device = None
        self._entity_weights_cache = {}
        self._q_agg_cache = {}

    def clear_frame_store(self) -> None:
        self._frame_kv_store = {}
        self._memory_kv_cache = None
        self._memory_kv_cache_key = None
        self._memory_kv_cache_device = None


    def build_entity_attrs_query(self, entities: List[EntityStruct]) -> str:
        parts = []
        for entity in entities:
            entity_str = entity.entity
            attrs_str = " ".join(entity.attrs)
            parts.append(f"{entity_str} {attrs_str}")

        return " ".join(parts)

    def get_entity_ids(self, entities: List[EntityStruct]) -> List[int]:
        return [e.global_id for e in entities if e.global_id is not None]

    def get_registry_summary(self) -> str:
        lines = []
        for gid, info in self.global_registry.items():
            entities = info.get("all_entities", [])
            lines.append(
                f"ID {gid} ({info.get('name', 'unknown')}): entities={entities}"
            )

        return "\n".join(lines)

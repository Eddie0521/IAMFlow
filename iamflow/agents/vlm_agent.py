import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_SAMPLE_INDICES = [0, 4, 8]
_VLM_RESIZE = (256, 256)  # Resize frames before sending to VLM for speed


class VLMAgent:

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.15,
        enabled: bool = True,
        backend: Optional[str] = None,
        device_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enabled = enabled
        self.backend = str(backend or "vllm")
        self._device_id = device_id

        self._llm = None  # vLLM LLM instance
        self._processor = None

        # Thread pool for async scoring
        self._executor: Optional[ThreadPoolExecutor] = None

        # Caches (thread-safe via lock)
        self._lock = threading.Lock()
        self._score_cache: Dict[
            str, Dict[int, float]
        ] = {}  # "p{pid}_c{cid}" -> {latent_frame_idx: score}
        self._correction_cache: Dict[
            int, Optional[Dict]
        ] = {}  # prompt_id -> corrections
        self._futures: Dict[str, Future] = {}  # chunk_key -> future
        self._generation = (
            0  # Invalidate late async writes across clear()/new inference
        )

        # Profiling: per-chunk VLM inference time (recorded inside background thread)
        self._inference_times: List[float] = []  # milliseconds

    def preload(self) -> None:
        if not self.enabled:
            return

        if self._device_id is not None:
            logger.info(f"[VLMAgent] Pinning vLLM worker to GPU {self._device_id}")

        # Pin vLLM before importing it. vLLM inspects visible devices at import
        # and engine construction time, so changing CUDA_VISIBLE_DEVICES after
        # `import vllm` is too late on ROCm.
        # device_id is a *logical* CUDA index (e.g. cuda:1 inside the current
        # CUDA_VISIBLE_DEVICES mapping).  Translate to physical GPU id.
        saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self._device_id is not None:
            if saved_cvd is not None:
                physical_ids = [x.strip() for x in saved_cvd.split(",")]
                if self._device_id < len(physical_ids):
                    target_gpu = physical_ids[self._device_id]
                else:
                    target_gpu = str(self._device_id)
            else:
                target_gpu = str(self._device_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu

        try:
            from vllm import LLM, SamplingParams

            self._llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=2048,
                max_num_seqs=1,
                enforce_eager=True,
                limit_mm_per_prompt={"image": 3},
            )
        finally:
            if self._device_id is not None:
                if saved_cvd is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self._executor = ThreadPoolExecutor(max_workers=1)

        # Warmup with a dummy request
        logger.info("[VLMAgent] Model loaded, running warmup...")
        try:
            from vllm import SamplingParams

            dummy_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            params = SamplingParams(max_tokens=32, temperature=0.0)
            self._llm.generate(
                [
                    {
                        "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image briefly.<|im_end|>\n<|im_start|>assistant\n",
                        "multi_modal_data": {"image": [dummy_img]},
                    }
                ],
                sampling_params=params,
            )
        except Exception as e:
            logger.warning(f"[VLMAgent] Warmup failed (non-fatal): {e}")

        logger.info("[VLMAgent] Preload complete")

    def score_chunk_async(
        self,
        pixel_frames: torch.Tensor,
        prompt_text: str,
        prompt_id: int,
        chunk_id: int,
        entities: list,
        global_registry: Dict,
        is_first_chunk: bool = False,
    ) -> None:
        ready = self._llm is not None
        if not self.enabled or not ready or self._executor is None:
            return

        chunk_key = f"p{prompt_id}_c{chunk_id}"

        # Sample 3 frames from 12 pixel frames
        sampled_images = []
        for idx in _SAMPLE_INDICES:
            if idx < pixel_frames.shape[0]:
                frame = pixel_frames[idx]  # [C, H, W]
                # Convert to PIL
                img_np = (
                    (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
                )
                img = Image.fromarray(img_np)
                if _VLM_RESIZE is not None:
                    img = img.resize(_VLM_RESIZE, Image.BILINEAR)
                sampled_images.append(img)

        if not sampled_images:
            return

        entity_names = [e.entity for e in entities]
        entity_attrs = {}
        if is_first_chunk:
            for gid, reg in global_registry.items():
                entity_attrs[gid] = {
                    "name": reg.get("name", ""),
                    "all_attrs": reg.get("all_attrs", []),
                }

        with self._lock:
            generation = self._generation

        future = self._executor.submit(
            self._score_frames,
            sampled_images,
            prompt_text,
            entity_names,
            entity_attrs if is_first_chunk else None,
            prompt_id,
            chunk_key,
            generation,
        )
        with self._lock:
            self._futures[chunk_key] = future

    def score_chunk_sync(
        self,
        pixel_frames: torch.Tensor,
        prompt_text: str,
        prompt_id: int,
        chunk_id: int,
        entities: list,
        global_registry: Dict,
        is_first_chunk: bool = False,
    ) -> None:
        ready = self._llm is not None
        if not self.enabled or not ready:
            return

        chunk_key = f"p{prompt_id}_c{chunk_id}"
        sampled_images = self._sample_images(pixel_frames)
        if not sampled_images:
            return

        entity_names = [e.entity for e in entities]
        entity_attrs = (
            self._build_entity_attrs(global_registry) if is_first_chunk else None
        )
        with self._lock:
            generation = self._generation
        self._score_frames(
            sampled_images,
            prompt_text,
            entity_names,
            entity_attrs,
            prompt_id,
            chunk_key,
            generation,
        )

    def get_visual_scores(
        self, prompt_id: int, chunk_id: int, timeout: float = 30.0
    ) -> Optional[Dict[int, float]]:
        if not self.enabled:
            return None

        chunk_key = f"p{prompt_id}_c{chunk_id}"

        # Check cache first
        with self._lock:
            if chunk_key in self._score_cache:
                return self._score_cache[chunk_key]
            future = self._futures.get(chunk_key)

        if future is None:
            return None

        if not future.done():
            logger.warning(
                f"[VLMAgent] Blocking on VLM scores for {chunk_key} (not yet complete)"
            )

        try:
            future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"[VLMAgent] Failed to get scores for {chunk_key}: {e}")
            return None

        with self._lock:
            return self._score_cache.get(chunk_key)

    def get_attribute_corrections(self, prompt_id: int) -> Optional[Dict]:
        if not self.enabled:
            return None
        with self._lock:
            return self._correction_cache.get(prompt_id)

    def get_inference_times(self) -> List[float]:
        with self._lock:
            return list(self._inference_times)

    def clear(self) -> None:
        # Wait for pending futures
        with self._lock:
            self._generation += 1
            futures = list(self._futures.values())
        for f in futures:
            try:
                f.cancel()
                f.result(timeout=5.0)
            except Exception:
                pass
        with self._lock:
            self._score_cache.clear()
            self._correction_cache.clear()
            self._futures.clear()
            self._inference_times.clear()

    def shutdown(self) -> None:
        self.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ============ Internal ============

    def _sample_images(self, pixel_frames: torch.Tensor) -> List[Image.Image]:
        sampled_images = []
        for idx in _SAMPLE_INDICES:
            if idx < pixel_frames.shape[0]:
                frame = pixel_frames[idx]
                img_np = (
                    (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
                )
                img = Image.fromarray(img_np)
                if _VLM_RESIZE is not None:
                    img = img.resize(_VLM_RESIZE, Image.BILINEAR)
                sampled_images.append(img)
        return sampled_images

    @staticmethod
    def _build_entity_attrs(global_registry: Dict) -> Dict:
        entity_attrs = {}
        for gid, reg in global_registry.items():
            entity_attrs[gid] = {
                "name": reg.get("name", ""),
                "all_attrs": reg.get("all_attrs", []),
            }
        return entity_attrs

    def _score_frames(
        self,
        images: List[Image.Image],
        prompt_text: str,
        entity_names: List[str],
        entity_attrs: Optional[Dict],
        prompt_id: int,
        chunk_key: str,
        generation: int,
    ) -> None:
        entities_str = ", ".join(entity_names) if entity_names else "the main subject"

        # Build prompt
        score_instruction = (
            f"You are evaluating video frames for visual quality.\n"
            f'The video prompt is: "{prompt_text}"\n'
            f"Key entities: {entities_str}\n\n"
            f"Rate each of the {len(images)} frames from 0.0 to 1.0 on how well they "
            f"visually represent the entities and match the prompt description. "
            f"Consider clarity, consistency, and visual quality.\n"
        )

        if entity_attrs is not None:
            attrs_desc = []
            for gid, info in entity_attrs.items():
                attrs_str = ", ".join(info.get("all_attrs", []))
                attrs_desc.append(f"  ID {gid} ({info.get('name', '')}): {attrs_str}")
            attrs_block = "\n".join(attrs_desc)
            score_instruction += (
                f"\nAlso verify these entity attributes against the frames:\n"
                f"{attrs_block}\n"
                f"If any attribute is incorrect based on what you see, provide corrections.\n"
            )

        score_instruction += (
            f"\nRespond ONLY with valid JSON:\n"
            f'{{"scores": [<float>, ...], "corrections": null}}\n'
            f"If there are attribute corrections, use:\n"
            f'{{"scores": [<float>, ...], "corrections": {{"<global_id>": {{"corrected_attrs": ["attr1", "attr2"]}}}}}}\n'
        )

        t0 = time.monotonic()
        try:
            raw_text = self._score_frames_vllm(images, score_instruction)
            result = self._parse_vlm_output(raw_text, len(images))
        except Exception as e:
            logger.error(f"[VLMAgent] VLM inference failed for {chunk_key}: {e}")
            result = {"scores": [0.5] * len(images), "corrections": None}
        inference_ms = (time.monotonic() - t0) * 1000.0

        # Map sampled frame scores to latent frame indices
        # _SAMPLE_INDICES [0, 4, 8] correspond to latent frames [0, 1, 2]
        scores_dict: Dict[int, float] = {}
        raw_scores = result.get("scores", [0.5] * len(images))
        for i, score_val in enumerate(raw_scores):
            if i < len(_SAMPLE_INDICES):
                latent_idx = i  # latent frame 0, 1, 2
                scores_dict[latent_idx] = float(score_val)

        with self._lock:
            if generation != self._generation:
                self._futures.pop(chunk_key, None)
                return
            self._score_cache[chunk_key] = scores_dict
            self._inference_times.append(inference_ms)
            corrections = result.get("corrections")
            if corrections is not None:
                self._correction_cache[prompt_id] = corrections
            self._futures.pop(chunk_key, None)

    def _score_frames_vllm(
        self, images: List[Image.Image], score_instruction: str
    ) -> str:
        from vllm import SamplingParams

        image_tags = "".join(
            "<|vision_start|><|image_pad|><|vision_end|>" for _ in images
        )
        full_prompt = (
            f"<|im_start|>user\n{image_tags}\n{score_instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        params = SamplingParams(max_tokens=64, temperature=0.0)
        outputs = self._llm.generate(
            [
                {
                    "prompt": full_prompt,
                    "multi_modal_data": {"image": images},
                }
            ],
            sampling_params=params,
        )
        return outputs[0].outputs[0].text.strip()

    @staticmethod
    def _parse_vlm_output(raw_text: str, num_frames: int) -> Dict:
        """Parse VLM JSON output, with fallback for malformed responses."""
        # Try to extract JSON from the response
        try:
            # Find JSON block
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw_text[start:end])
                scores = parsed.get("scores", [])
                # Validate scores
                if isinstance(scores, list) and len(scores) >= 1:
                    scores = [max(0.0, min(1.0, float(s))) for s in scores[:num_frames]]
                    while len(scores) < num_frames:
                        scores.append(0.5)
                    parsed["scores"] = scores
                    parsed["corrections"] = VLMAgent._normalize_corrections(
                        parsed.get("corrections")
                    )
                    return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        logger.warning(
            f"[VLMAgent] Failed to parse VLM output, using defaults: {raw_text[:200]}"
        )
        return {"scores": [0.5] * num_frames, "corrections": None}

    @staticmethod
    def _normalize_corrections(
        corrections: Optional[Dict],
    ) -> Optional[Dict[str, Dict[str, List[str]]]]:
        """Normalize VLM corrections payload; discard malformed structures."""
        if corrections is None or not isinstance(corrections, dict):
            return None

        normalized: Dict[str, Dict[str, List[str]]] = {}
        for gid, corrected in corrections.items():
            if not isinstance(corrected, dict):
                continue

            corrected_attrs = corrected.get("corrected_attrs")
            if not isinstance(corrected_attrs, list):
                continue

            attrs = [str(attr) for attr in corrected_attrs if str(attr).strip()]
            if attrs:
                normalized[str(gid)] = {"corrected_attrs": attrs}

        return normalized or None

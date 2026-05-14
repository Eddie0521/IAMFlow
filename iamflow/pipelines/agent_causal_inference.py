import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from iamflow.agents.async_vae_vlm import AsyncVAEVLMPipeline

from iamflow.agents.llm_agent import EntityStruct, LLMAgent
from iamflow.agents.memory_bank import MemoryBank
from iamflow.agents.vlm_agent import VLMAgent
from iamflow.utils.debug_option import DEBUG
from iamflow.utils.memory import (
    get_cuda_free_memory_gb,
    gpu,
    move_model_to_device_with_memory_preservation,
)
from iamflow.utils.profiling import (
    build_comparable_metrics,
    compute_pure_diffusion_time,
    summarize_memory_timings,
)
from iamflow.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from iamflow.pipelines.interactive_causal_inference import InteractiveCausalInferencePipeline


class AgentCausalInferencePipeline(InteractiveCausalInferencePipeline):

    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
        llm_model_path: str = "../Qwen3-4B-Instruct-2507",
        max_memory_frames: int = 3,
        save_dir: str = "data/agent_frames",
        save_frames_to_disk: bool = False,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.2,
        use_lightvae: bool = False,
        vlm_model_path: str = "../Qwen3-VL-2B-Instruct",
        vlm_enabled: bool = False,
        vlm_gpu_memory_utilization: float = 0.15,
        vlm_score_weight: float = 0.3,
        async_vae_enabled: bool = True,
        llm_device_id: Optional[int] = None,
        vlm_device_id: Optional[int] = None,
    ):
        super().__init__(
            args,
            device,
            generator=generator,
            text_encoder=text_encoder,
            vae=vae,
            use_lightvae=use_lightvae,
        )

        self.llm_agent = LLMAgent(
            model_path=llm_model_path,
            use_vllm=use_vllm,
            gpu_memory_utilization=gpu_memory_utilization,
            device_id=llm_device_id,
        )

        self.min_id_memory_frames_multi_entity = int(
            getattr(args, "min_id_memory_frames_multi_entity", 2)
        )

        self.agent_memory_bank = MemoryBank(
            text_encoder=self.text_encoder,
            max_memory_frames=max_memory_frames,
            min_id_memory_frames_multi_entity=self.min_id_memory_frames_multi_entity,
            frame_seq_length=self.frame_seq_length,
            num_transformer_blocks=self.num_transformer_blocks,
            save_dir=save_dir,
            save_frames_to_disk=save_frames_to_disk,
        )

        if self.bank_size < self.agent_memory_bank.max_id_memory_frames:
            self.bank_size = self.agent_memory_bank.max_id_memory_frames

        self.current_prompt_id = 0
        self.current_chunk_id = 0
        self.current_entities: List[EntityStruct] = []
        self.current_prompt_text: str = ""

        self._iam_bank_length = 0

        self._last_injected_memory_key: Optional[Tuple[str, ...]] = None

        self.llm_model_path = llm_model_path
        self.max_memory_frames = max_memory_frames
        self.save_dir = save_dir
        self.save_frames_to_disk = save_frames_to_disk
        self._precomputed_prompt_entities: Dict[int, List[EntityStruct]] = {}

        os.makedirs(save_dir, exist_ok=True)


        self.vlm_score_weight = vlm_score_weight
        if vlm_enabled:
            self.vlm_agent = VLMAgent(
                model_path=vlm_model_path,
                gpu_memory_utilization=vlm_gpu_memory_utilization,
                enabled=True,
                device_id=vlm_device_id,
            )
            self.vlm_agent.preload()
        else:
            self.vlm_agent = None

        if async_vae_enabled:
            self._async_vae_vlm: Optional[AsyncVAEVLMPipeline] = AsyncVAEVLMPipeline(
                vae=self.vae,
                vlm_agent=self.vlm_agent,  # None when vlm_enabled=false
            )
        else:
            self._async_vae_vlm = None
        self._sync_pixel_store: Dict[str, torch.Tensor] = {}
        self._sync_pixel_order: List[str] = []
        self._sync_vae_decode_times: List[float] = []

        # VLM wait profiling (accumulated per inference run)
        self._vlm_wait_log: List[
            Tuple[str, float, bool]
        ] = []  # (chunk_key, wait_ms, did_block)

    @staticmethod
    def _compute_prompt_distance(cond_old: dict, cond_new: dict) -> float:
        import torch.nn.functional as F

        emb_old = cond_old["prompt_embeds"].mean(dim=(0, 1))
        emb_new = cond_new["prompt_embeds"].mean(dim=(0, 1))
        cos_sim = F.cosine_similarity(emb_old.unsqueeze(0), emb_new.unsqueeze(0))
        return 1.0 - cos_sim.item()

    def _decode_chunk_pixels_for_vlm(self, denoised_pred: torch.Tensor) -> torch.Tensor:
        vae_param = next(self.vae.model.parameters(), None)
        if vae_param is not None:
            denoised_pred = denoised_pred.to(device=vae_param.device, non_blocking=True)
        chunk_pixel = self.vae.decode_to_pixel_stream(denoised_pred)
        return (chunk_pixel * 0.5 + 0.5).clamp(0, 1)

    def _run_sync_vae_vlm(
        self,
        denoised_pred: torch.Tensor,
        chunk_key: str,
        prompt_text: str,
        prompt_id: int,
        chunk_id: int,
        entities: list,
        is_first_chunk: bool,
    ) -> None:
        """Decode and run VLM on the main path for the w/o async ablation."""
        t0 = time.monotonic()
        with torch.no_grad():
            chunk_pixel = self._decode_chunk_pixels_for_vlm(denoised_pred)
        chunk_pixel_cpu = chunk_pixel[0].cpu()
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        self._sync_pixel_store[chunk_key] = chunk_pixel_cpu
        self._sync_pixel_order.append(chunk_key)
        self._sync_vae_decode_times.append(elapsed_ms)

        if getattr(self, "vlm_agent", None) is not None:
            self.vlm_agent.score_chunk_sync(
                pixel_frames=chunk_pixel_cpu,
                prompt_text=prompt_text,
                prompt_id=prompt_id,
                chunk_id=chunk_id,
                entities=entities,
                global_registry=self.agent_memory_bank.global_registry,
                is_first_chunk=is_first_chunk,
            )

    def _get_chunk_pixels(self, chunk_key: str) -> Optional[torch.Tensor]:
        if self._async_vae_vlm is not None:
            return self._async_vae_vlm.get_pixels(chunk_key)
        return self._sync_pixel_store.get(chunk_key)

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        save_mapping: bool = True,
        mapping_path: str = "mapping.json",
        profile: bool = False,
    ):
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # ===== Profiling Setup =====
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            recache_start = torch.cuda.Event(enable_timing=True)
            recache_end = torch.cuda.Event(enable_timing=True)

            # IAM specific timers
            agent_start = torch.cuda.Event(enable_timing=True)
            agent_end = torch.cuda.Event(enable_timing=True)
            memory_start = torch.cuda.Event(enable_timing=True)
            memory_end = torch.cuda.Event(enable_timing=True)

            block_times = []
            agent_times = []  # Per-prompt agent processing time
            memory_times = []  # Per-chunk memory bank timing entries
            recache_times = []

            init_start.record()

        self._reset_agent_state()
        if not self._precomputed_prompt_entities:
            self._precompute_prompt_entities(text_prompts_list)

        if DEBUG:
            print(f"[AgentPipeline] text_prompts_list: {text_prompts_list}")
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        kv_bank1_size = self.bank_size * self.frame_seq_length
        self._initialize_kv_bank(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_bank1_size=kv_bank1_size,
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0
        prompt_start_frame = 0
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if profile:
            agent_start.record()

        self._process_prompt_start(
            prompt_text=text_prompts_list[0][0], prompt_id=1, is_first_prompt=True
        )

        if profile:
            agent_end.record()
            torch.cuda.synchronize()
            agent_time = agent_start.elapsed_time(agent_end)
            agent_times.append(("Prompt 1", agent_time))

        for block_idx, current_num_frames in enumerate(all_num_frames):
            update_bank = False

            if profile:
                block_start.record()

            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                prompt_start_frame = switch_frame_indices[segment_idx - 1]

                if profile:
                    recache_start.record()

                if self.spt_enabled:
                    scheduler = getattr(self, "transition_scheduler", None)
                    if hasattr(scheduler, "update_for_switch"):
                        dist = self._compute_prompt_distance(
                            cond_list[segment_idx - 1], cond_list[segment_idx]
                        )
                        scheduler.update_for_switch(dist)
                    self._soft_switch()
                    if self.kv_bank1 is not None:
                        for blk in self.kv_bank1:
                            blk["local_end_index"].zero_()
                            blk["global_end_index"].zero_()
                        self._iam_bank_length = 0
                        if DEBUG:
                            print(
                                f"[AgentPipeline] Reset kv_bank indices for prompt switch (SPT)"
                            )
                else:
                    self._recache_after_switch(
                        output, current_start_frame, cond_list[segment_idx]
                    )

                if profile:
                    recache_end.record()
                    torch.cuda.current_stream().synchronize()
                    recache_time = recache_start.elapsed_time(recache_end)
                    recache_times.append((f"Prompt {segment_idx + 1}", recache_time))

                if profile:
                    agent_start.record()

                self._process_prompt_start(
                    prompt_text=text_prompts_list[segment_idx][0],
                    prompt_id=segment_idx + 1,
                    is_first_prompt=False,
                )

                if profile:
                    agent_end.record()
                    torch.cuda.current_stream().synchronize()
                    agent_time = agent_start.elapsed_time(agent_end)
                    agent_times.append((f"Prompt {segment_idx + 1}", agent_time))

                if DEBUG:
                    print(
                        f"[AgentPipeline] Switched to segment {segment_idx} at frame {current_start_frame}"
                    )

                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )

            cond_in_use = cond_list[segment_idx]
            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]
            transition_alpha = (
                self._get_current_transition_alpha() if self.spt_enabled else None
            )

            for index, current_timestep in enumerate(self.denoising_step_list):
                if index == 0:
                    q_bank = True
                else:
                    q_bank = False

                timestep = (
                    torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64,
                    )
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_bank=self.kv_bank1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        update_bank=False,
                        q_bank=q_bank,
                        iam_bank_length=self._iam_bank_length,
                        prev_crossattn_cache=self.prev_crossattn_cache,
                        transition_alpha=transition_alpha,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_bank=self.kv_bank1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        update_bank=False,
                        q_bank=q_bank,
                        iam_bank_length=self._iam_bank_length,
                        prev_crossattn_cache=self.prev_crossattn_cache,
                        transition_alpha=transition_alpha,
                    )

            output[
                :, current_start_frame : current_start_frame + current_num_frames
            ] = denoised_pred.to(output.device)

            if self._async_vae_vlm is not None:
                chunk_key = f"p{self.current_prompt_id}_c{self.current_chunk_id + 1}"
                self._async_vae_vlm.submit(
                    denoised_pred=denoised_pred,
                    chunk_key=chunk_key,
                    prompt_text=self.current_prompt_text,
                    prompt_id=self.current_prompt_id,
                    chunk_id=self.current_chunk_id + 1,
                    entities=self.current_entities,
                    global_registry=self.agent_memory_bank.global_registry,
                    is_first_chunk=(self.current_chunk_id == 0),
                )
            elif getattr(self, "vlm_agent", None) is not None:
                chunk_key = f"p{self.current_prompt_id}_c{self.current_chunk_id + 1}"
                self._run_sync_vae_vlm(
                    denoised_pred=denoised_pred,
                    chunk_key=chunk_key,
                    prompt_text=self.current_prompt_text,
                    prompt_id=self.current_prompt_id,
                    chunk_id=self.current_chunk_id + 1,
                    entities=self.current_entities,
                    is_first_chunk=(self.current_chunk_id == 0),
                )

            self.current_chunk_id += 1
            _did_eviction = False

            evicted_chunk_lag = 3
            if self.current_chunk_id >= evicted_chunk_lag + 1 and self.current_entities:
                if profile:
                    wait_count_before = len(self._vlm_wait_log)
                    memory_start.record()

                self._process_chunk_eviction(
                    current_start_frame=current_start_frame,
                    current_num_frames=current_num_frames,
                )
                self._inject_iam_memory_to_bank()
                _did_eviction = True

                if profile:
                    memory_end.record()
                    # NOTE: no synchronize() here — deferred to block_end to avoid
                    # serializing the background VAE thread with the clean context forward.

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_bank=self.kv_bank1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                update_bank=False,
                q_bank=q_bank,
                update_cache=True,
                iam_bank_length=self._iam_bank_length,
                prev_crossattn_cache=self.prev_crossattn_cache,
                transition_alpha=transition_alpha,
            )

            if (
                not _did_eviction
                and self.current_chunk_id >= 1
                and self.current_entities
            ):
                if profile:
                    memory_start.record()

                self._process_chunk_archival(current_start_frame)
                self._inject_iam_memory_to_bank()

                if self.vlm_agent is not None and self.current_chunk_id == 2:
                    corrections = self.vlm_agent.get_attribute_corrections(
                        self.current_prompt_id
                    )
                    if corrections:
                        self.agent_memory_bank.apply_attribute_corrections(
                            self.current_prompt_id, corrections
                        )

                if profile:
                    memory_end.record()
                    # NOTE: no synchronize() here — deferred to block_end.

            if self.spt_enabled:
                self._update_transition_state(current_num_frames)

            if profile:
                block_end.record()
                torch.cuda.current_stream().synchronize()  # main stream only — background VAE continues
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

                # Compute memory time from deferred events (if recorded this block)
                if _did_eviction and self.current_entities:
                    memory_time = memory_start.elapsed_time(memory_end)
                    step_wait_ms = sum(
                        w for _, w, _ in self._vlm_wait_log[wait_count_before:]
                    )
                    memory_times.append(
                        {
                            "label": f"Chunk {block_idx}",
                            "total_ms": memory_time,
                            "wait_ms": step_wait_ms,
                        }
                    )
                elif (
                    not _did_eviction
                    and self.current_chunk_id >= 1
                    and self.current_entities
                ):
                    memory_time = memory_start.elapsed_time(memory_end)
                    memory_times.append(
                        {
                            "label": f"Chunk {block_idx}",
                            "total_ms": memory_time,
                            "wait_ms": 0.0,
                        }
                    )

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)

            vae_start.record()

        if self._async_vae_vlm is not None:
            pixel_chunks = self._async_vae_vlm.get_ordered_chunks()
            self.vae.end_decode_stream()
        else:
            sync_pixel_store = getattr(self, "_sync_pixel_store", {})
            sync_pixel_order = getattr(self, "_sync_pixel_order", [])
            pixel_chunks = [
                sync_pixel_store[k] for k in sync_pixel_order if k in sync_pixel_store
            ]
            if pixel_chunks:
                self.vae.end_decode_stream()

        free_kv_memory = getattr(self, "_free_kv_memory", None)
        if callable(free_kv_memory):
            free_kv_memory()
        else:
            self.clear_kv_cache()
        if hasattr(self, "agent_memory_bank"):
            self.agent_memory_bank.clear_frame_store()

        if pixel_chunks:
            video = torch.cat(pixel_chunks, dim=0).unsqueeze(0)  # [1, T, C, H, W]
            video = video.to(device=noise.device)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)

        if save_mapping:
            self.agent_memory_bank.save_to_json(mapping_path)
            if DEBUG:
                print(f"[AgentPipeline] Saved mapping to {mapping_path}")

        # ===== Profiling Results =====
        if profile:
            total_agent_time = sum(t for _, t in agent_times)
            memory_summary = summarize_memory_timings(memory_times)
            total_memory_time = memory_summary["total_ms"]
            total_recache_time = sum(t for _, t in recache_times)
            # VAE decode is now async (overlapped with DiT); collect wall times from background thread.
            vlm_vae_times: List[float] = []
            if self._async_vae_vlm is not None:
                vlm_vae_times = self._async_vae_vlm.get_vae_decode_times()
            elif self._sync_vae_decode_times:
                vlm_vae_times = list(self._sync_vae_decode_times)
            total_vlm_vae_time = sum(vlm_vae_times) if vlm_vae_times else 0.0

            # VLM wait stats (from _vlm_wait_log accumulated during this run)
            vlm_wait_entries = self._vlm_wait_log
            total_vlm_wait_time = sum(w for _, w, _ in vlm_wait_entries)
            vlm_block_count = sum(1 for _, _, b in vlm_wait_entries if b)
            vlm_total_fetches = len(vlm_wait_entries)

            # VLM async inference stats (from VLMAgent background thread)
            vlm_inference_times: List[float] = []
            if getattr(self, "vlm_agent", None) is not None:
                vlm_inference_times = self.vlm_agent.get_inference_times()
            total_vlm_inference_time = sum(vlm_inference_times)

            pure_memory_time = memory_summary["pure_ms"]

            # Pure diffusion = diffusion - all overhead (including vlm_vae decode)
            # VAE decode is now overlapped with the DiT loop; do not subtract it
            # from diffusion_time (it no longer adds to the wall-clock loop time).
            pure_time, pure_pct = compute_pure_diffusion_time(
                diffusion_time=diffusion_time,
                agent_time=total_agent_time,
                memory_time=total_memory_time,
                recache_time=total_recache_time,
                vlm_vae_time=0.0
                if self._async_vae_vlm is not None
                else total_vlm_vae_time,
            )

            total_time = init_time + diffusion_time + vae_time
            comparable_metrics = build_comparable_metrics(
                total_time=total_time,
                diffusion_time=diffusion_time,
                final_output_time=vae_time,
                pure_denoising_time=pure_time,
                extra_output_decode_time=0.0,  # VAE overlapped, not a sequential cost
            )
            _dp = lambda ms: (
                f"{100 * ms / diffusion_time:5.2f}" if diffusion_time > 0 else " 0.00"
            )
            _tp = lambda ms: (
                f"{100 * ms / total_time:5.2f}" if total_time > 0 else " 0.00"
            )

            print("\n" + "=" * 70)
            print("IAM Agent Pipeline Profiling Results")
            print("=" * 70)

            # --- Overall ---
            print(f"\n[Overall Performance]")
            print(
                f"  Initialization:          {init_time:10.2f} ms ({_tp(init_time)}%)"
            )
            print(
                f"  Diffusion loop:          {diffusion_time:10.2f} ms ({_tp(diffusion_time)}%)"
            )
            vae_label = "Final VAE (concat)" if pixel_chunks else "Final VAE decode"
            print(f"  {vae_label + ':':27s}{vae_time:10.2f} ms ({_tp(vae_time)}%)")
            print(f"  Total:                   {total_time:10.2f} ms")
            print(
                f"  Throughput:              {num_output_frames / (total_time / 1000):10.2f} FPS"
            )

            print(f"\n[Comparable Metrics]")
            print(
                f"  End-to-end total:        {comparable_metrics['end_to_end_ms']:10.2f} ms"
            )
            print(
                f"  Loop (no output decode): {comparable_metrics['loop_no_output_decode_ms']:10.2f} ms"
            )
            print(
                f"  Core DiT time:           {comparable_metrics['core_dit_ms']:10.2f} ms"
            )
            print(
                f"  Method overhead:         {comparable_metrics['method_overhead_ms']:10.2f} ms"
            )
            print(
                f"  Output decode total:     {comparable_metrics['output_decode_ms']:10.2f} ms"
            )

            # --- Diffusion breakdown ---
            print(f"\n[Diffusion Breakdown]")
            print(
                f"  Pure denoising:          {pure_time:10.2f} ms ({_dp(pure_time)}% of diffusion)"
            )
            print(
                f"  LLM Agent (entity):      {total_agent_time:10.2f} ms ({_dp(total_agent_time)}%)"
            )
            print(
                f"  {'Memory Bank (select+inject):':27s}{pure_memory_time:10.2f} ms ({_dp(pure_memory_time)}%)"
            )
            if total_vlm_vae_time > 0:
                if self._async_vae_vlm is not None:
                    print(
                        f"  Async VAE decode:        {total_vlm_vae_time:10.2f} ms ({_dp(total_vlm_vae_time)}%) [overlapped, not in loop wall time]"
                    )
                else:
                    print(
                        f"  Sync VAE decode:         {total_vlm_vae_time:10.2f} ms ({_dp(total_vlm_vae_time)}%) [blocking]"
                    )
            if total_vlm_wait_time > 0:
                print(
                    f"  VLM wait (block):        {total_vlm_wait_time:10.2f} ms ({_dp(total_vlm_wait_time)}%)  [{vlm_block_count}/{vlm_total_fetches} blocked]"
                )
            if total_recache_time > 0:
                print(
                    f"  Recache (SPT):           {total_recache_time:10.2f} ms ({_dp(total_recache_time)}%)"
                )

            # --- Async VAE details ---
            if vlm_vae_times:
                avg_vae = total_vlm_vae_time / len(vlm_vae_times)
                section = (
                    "Async VAE Decode"
                    if self._async_vae_vlm is not None
                    else "Sync VAE Decode"
                )
                mode = "async" if self._async_vae_vlm is not None else "sync"
                suffix = (
                    " [overlapped with DiT]"
                    if self._async_vae_vlm is not None
                    else " [blocking]"
                )
                print(f"\n[{section}]")
                print(
                    f"  Total ({mode}):           {total_vlm_vae_time:10.2f} ms total, {avg_vae:.2f} ms avg x {len(vlm_vae_times)} chunks{suffix}"
                )
                # In per-chunk decode mode the final "VAE" step is just a concat (~0ms).
                if pixel_chunks and vae_time < total_vlm_vae_time:
                    saved = total_vlm_vae_time - vae_time
                    if saved > 0:
                        print(
                            f"  Final VAE saved:         ~{saved:10.2f} ms (skipped full decode)"
                        )

            # --- VLM details (only if VLM enabled) ---
            if self.vlm_agent is not None and (vlm_inference_times or vlm_wait_entries):
                print(f"\n[VLM Details]")
                if vlm_inference_times:
                    avg_inf = total_vlm_inference_time / len(vlm_inference_times)
                    print(
                        f"  VLM inference (async):   {total_vlm_inference_time:10.2f} ms total, {avg_inf:.2f} ms avg x {len(vlm_inference_times)} chunks"
                    )
                if vlm_wait_entries:
                    blocked_keys = [k for k, _, b in vlm_wait_entries if b]
                    print(
                        f"  VLM wait events:         {vlm_block_count}/{vlm_total_fetches} chunks blocked",
                        end="",
                    )
                    if blocked_keys:
                        print(
                            f" ({', '.join(blocked_keys[:5])}"
                            + ("..." if len(blocked_keys) > 5 else "")
                            + ")"
                        )
                    else:
                        print()

            # --- Per-prompt recache ---
            if recache_times:
                print(f"\n[Recache - Per Prompt Switch]")
                for prompt_name, time_ms in recache_times:
                    print(f"  {prompt_name:12s} recache:      {time_ms:8.2f} ms")

            # --- Per-prompt agent ---
            if agent_times:
                print(f"\n[LLM Agent - Per Prompt]")
                for prompt_name, time_ms in agent_times:
                    print(f"  {prompt_name:12s} processing:   {time_ms:8.2f} ms")

            # --- Per-chunk memory bank (show first 5 + last 5 if many) ---
            if memory_times:
                print(f"\n[Memory Bank - Per Chunk]")

                def _print_memory_entry(entry):
                    label = str(entry["label"])
                    total_ms = float(entry["total_ms"])
                    wait_ms = float(entry.get("wait_ms", 0.0))
                    pure_ms = max(0.0, total_ms - wait_ms)
                    if wait_ms > 0:
                        print(
                            f"  {label:12s} select+inject: {pure_ms:8.2f} ms "
                            f"(wait {wait_ms:6.2f} ms, total {total_ms:8.2f} ms)"
                        )
                    else:
                        print(
                            f"  {label:12s} select+inject: {pure_ms:8.2f} ms "
                            f"(total {total_ms:8.2f} ms)"
                        )

                if len(memory_times) <= 10:
                    for entry in memory_times:
                        _print_memory_entry(entry)
                else:
                    for entry in memory_times[:5]:
                        _print_memory_entry(entry)
                    print(f"  ... ({len(memory_times) - 10} chunks omitted)")
                    for entry in memory_times[-5:]:
                        _print_memory_entry(entry)
                print(
                    f"  Average per chunk (pure):  {memory_summary['avg_pure_ms']:8.2f} ms"
                )
                if memory_summary["wait_ms"] > 0:
                    print(
                        f"  Average per chunk (wait):  {memory_summary['avg_wait_ms']:8.2f} ms"
                    )
                print(
                    f"  Average per chunk (total): {memory_summary['avg_total_ms']:8.2f} ms"
                )

            # --- Per-block diffusion ---
            print(f"\n[Diffusion - Per Block]")
            if len(block_times) <= 10:
                for i, bt in enumerate(block_times):
                    print(f"  Block {i:3d}:                 {bt:8.2f} ms ({_dp(bt)}%)")
            else:
                for i in range(5):
                    print(
                        f"  Block {i:3d}:                 {block_times[i]:8.2f} ms ({_dp(block_times[i])}%)"
                    )
                print(f"  ... ({len(block_times) - 10} blocks omitted)")
                for i in range(len(block_times) - 5, len(block_times)):
                    print(
                        f"  Block {i:3d}:                 {block_times[i]:8.2f} ms ({_dp(block_times[i])}%)"
                    )
            avg_block_time = diffusion_time / len(block_times)
            print(f"  Average per block:         {avg_block_time:8.2f} ms")

            print("=" * 70 + "\n")

        if return_latents:
            return video, output
        return video


    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        super()._recache_after_switch(output, current_start_frame, new_conditional_dict)

        if self.kv_bank1 is not None:
            for blk in self.kv_bank1:
                blk["local_end_index"].zero_()
                blk["global_end_index"].zero_()
            self._iam_bank_length = 0
            self._last_injected_memory_key = None
            if DEBUG:
                print(f"[AgentPipeline] Reset kv_bank indices for prompt switch")

    def _reset_agent_state(self) -> None:
        self.current_prompt_id = 0
        self.current_chunk_id = 0
        self.current_entities = []
        self.current_prompt_text = ""
        self.agent_memory_bank.clear()
        self._iam_bank_length = 0
        self.llm_agent.id_manager._next_id = 1
        if getattr(self, "vlm_agent", None) is not None:
            self.vlm_agent.clear()
        async_vae_vlm = getattr(self, "_async_vae_vlm", None)
        if async_vae_vlm is not None:
            self._async_vae_vlm.clear()
        vae = getattr(self, "vae", None)
        if vae is not None and hasattr(vae, "reset_decode_stream"):
            vae.reset_decode_stream()
        if hasattr(self, "_chunk_pixel_store"):
            self._chunk_pixel_store = {}
        if hasattr(self, "_pixel_chunks"):
            self._pixel_chunks = []
        self._sync_pixel_store = {}
        self._sync_pixel_order = []
        self._sync_vae_decode_times = []
        self._vlm_wait_log = []

    def _precompute_prompt_entities(self, text_prompts_list: List[List[str]]) -> None:
        """Run all prompt LLM work up front, then release the LLM before DiT."""
        if self.llm_agent is None:
            return

        self._precomputed_prompt_entities = {}
        self.agent_memory_bank.clear()
        self.llm_agent.id_manager._next_id = 1

        self.llm_agent.preload()
        for prompt_id, prompt_group in enumerate(text_prompts_list, start=1):
            prompt_text = prompt_group[0]
            entities, registry_update = self.llm_agent.process_prompt(
                prompt=prompt_text,
                prompt_id=prompt_id,
                global_registry=self.agent_memory_bank.global_registry,
            )
            self.agent_memory_bank.register_entities(
                entities, prompt_id, registry_update
            )
            self._precomputed_prompt_entities[prompt_id] = entities

        self.llm_agent.unload()
        self.agent_memory_bank.clear()
        self.llm_agent.id_manager._next_id = 1

    def _process_prompt_start(
        self, prompt_text: str, prompt_id: int, is_first_prompt: bool
    ) -> None:
        self.current_prompt_id = prompt_id
        self.current_chunk_id = 0
        self.current_prompt_text = (
            prompt_text
        )

        if prompt_id in self._precomputed_prompt_entities:
            entities = self._precomputed_prompt_entities[prompt_id]
            registry_update = None
        elif self.llm_agent is None:
            entities, registry_update = [], None
        else:
            entities, registry_update = self.llm_agent.process_prompt(
                prompt=prompt_text,
                prompt_id=prompt_id,
                global_registry=self.agent_memory_bank.global_registry,
            )

        self.current_entities = entities

        if DEBUG:
            print(f"[AgentPipeline] Prompt {prompt_id} entities:")
            for e in entities:
                print(f"  - {e.entity} (ID: {e.global_id}): {e.attrs}")

        self.agent_memory_bank.register_entities(entities, prompt_id, registry_update)

        memory_enabled = getattr(self, "memory_enabled", True)
        memory_selection_mode = getattr(self, "memory_selection_mode", "entity")
        should_retrieve = (
            memory_enabled
            and not is_first_prompt
            and (bool(entities) or memory_selection_mode == "prompt")
        )
        if should_retrieve:
            entity_ids = self.agent_memory_bank.get_entity_ids(entities)

            self.agent_memory_bank.retrieve_initial_frames(
                entity_ids,
            )

            if DEBUG:
                print(
                    f"[AgentPipeline] Retrieved initial frames: {self.agent_memory_bank.frame_active_memory}"
                )

            self._inject_iam_memory_to_bank()

    def _fetch_vlm_scores(
        self, prompt_id: int, chunk_id: int
    ) -> Optional[Dict[int, float]]:
        """Fetch VLM scores with wait-time tracking for profiling."""
        if getattr(self, "vlm_agent", None) is None:
            return None
        chunk_key = f"p{prompt_id}_c{chunk_id}"
        # Check if the future is already done before we call get_visual_scores
        with self.vlm_agent._lock:
            future = self.vlm_agent._futures.get(chunk_key)
            already_done = (chunk_key in self.vlm_agent._score_cache) or (
                future is not None and future.done()
            )
        t0 = time.monotonic()
        scores = self.vlm_agent.get_visual_scores(prompt_id, chunk_id)
        wait_ms = (time.monotonic() - t0) * 1000.0
        did_block = not already_done and wait_ms > 1.0
        self._vlm_wait_log.append((chunk_key, wait_ms, did_block))
        return scores

    def _process_chunk_eviction(
        self, current_start_frame: int, current_num_frames: int
    ) -> None:
        memory_enabled = getattr(self, "memory_enabled", True)
        memory_selection_mode = getattr(self, "memory_selection_mode", "entity")
        if not memory_enabled:
            return
        if not self.current_entities and memory_selection_mode != "prompt":
            return

        evicted_chunk_lag = 3
        evicted_chunk_kv = self._get_evicted_chunk_kv(evicted_chunk_lag)

        if evicted_chunk_kv is None:
            return

        evicted_cid = self.current_chunk_id - evicted_chunk_lag
        visual_scores = None
        chunk_pixels = None

        if getattr(self, "vlm_agent", None) is not None:
            visual_scores = self._fetch_vlm_scores(self.current_prompt_id, evicted_cid)
            chunk_key = f"p{self.current_prompt_id}_c{evicted_cid}"
            chunk_pixels = self._get_chunk_pixels(chunk_key)

        entity_ids = self.agent_memory_bank.get_entity_ids(self.current_entities)
        frame_id, score = self.agent_memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=evicted_chunk_kv,
            crossattn_cache=self.crossattn_cache,
            prompt_id=self.current_prompt_id,
            chunk_id=evicted_cid,
            current_entity_ids=entity_ids,
            current_entities=self.current_entities,
            prompt_text=self.current_prompt_text,
            visual_scores=visual_scores,
            pixel_frames=chunk_pixels,
            visual_weight=getattr(self, "vlm_score_weight", 0.3),
        )

        frame_info = self.agent_memory_bank.frame_archive[frame_id]
        if memory_selection_mode == "prompt":
            self.agent_memory_bank.update_active_memory(
                frame_id, frame_info.entity_score
            )
        else:
            self.agent_memory_bank.update_id_memory(frame_id, frame_info.entity_score)

        if DEBUG:
            print(
                f"[AgentPipeline] IAM selected frame {frame_id} with score {score:.4f}"
            )
            print(
                f"[AgentPipeline] Active memory: {self.agent_memory_bank.frame_active_memory}"
            )

    def _process_chunk_archival(self, current_start_frame: int) -> None:
        if not self.current_entities or self.kv_cache1 is None:
            return

        chunk_length = self.num_frame_per_block * self.frame_seq_length
        sink_tokens = (
            getattr(self.args.model_kwargs, "sink_size", 3) * self.frame_seq_length
        )

        cache = self.kv_cache1[0]
        local_end = cache["local_end_index"].item()
        chunk_start = max(sink_tokens, local_end - chunk_length)
        chunk_end = local_end

        if chunk_end <= chunk_start:
            return

        all_blocks_kv = []
        for block_cache in self.kv_cache1:
            all_blocks_kv.append(
                {
                    "k": block_cache["k"][:, chunk_start:chunk_end],
                    "v": block_cache["v"][:, chunk_start:chunk_end],
                }
            )

        visual_scores = None
        chunk_pixels = None
        if getattr(self, "vlm_agent", None) is not None:
            chunk_key = f"p{self.current_prompt_id}_c{self.current_chunk_id}"
            visual_scores = self._fetch_vlm_scores(
                self.current_prompt_id, self.current_chunk_id
            )
            chunk_pixels = self._get_chunk_pixels(chunk_key)

        entity_ids = self.agent_memory_bank.get_entity_ids(self.current_entities)
        frame_id, score = self.agent_memory_bank.select_frame_from_chunk(
            evicted_chunk_kv=all_blocks_kv,
            crossattn_cache=self.crossattn_cache,
            prompt_id=self.current_prompt_id,
            chunk_id=self.current_chunk_id,
            current_entity_ids=entity_ids,
            current_entities=self.current_entities,
            prompt_text=self.current_prompt_text,
            visual_scores=visual_scores,
            pixel_frames=chunk_pixels,
            visual_weight=getattr(self, "vlm_score_weight", 0.3),
        )

        frame_info = self.agent_memory_bank.frame_archive[frame_id]
        self.agent_memory_bank.update_id_memory(frame_id, frame_info.entity_score)

        if DEBUG:
            print(
                f"[AgentPipeline] Archived frame {frame_id} (score={score:.4f}) from chunk {self.current_chunk_id}"
            )

    def _get_evicted_chunk_kv(
        self, chunk_lag: int = 3
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        if self.kv_cache1 is None or len(self.kv_cache1) == 0:
            return None

        cache = self.kv_cache1[0]
        k = cache["k"]

        local_end = cache["local_end_index"].item()

        chunk_length = self.num_frame_per_block * self.frame_seq_length

        chunk_start = max(0, local_end - chunk_lag * chunk_length)
        chunk_end = max(0, local_end - (chunk_lag - 1) * chunk_length)

        if chunk_end <= chunk_start or chunk_end > k.shape[1]:
            return None

        all_blocks_kv = []
        for block_idx in range(len(self.kv_cache1)):
            block_cache = self.kv_cache1[block_idx]
            all_blocks_kv.append(
                {
                    "k": block_cache["k"][:, chunk_start:chunk_end],
                    "v": block_cache["v"][:, chunk_start:chunk_end],
                }
            )

        return all_blocks_kv

    def _inject_iam_memory_to_bank(self) -> None:
        if self.kv_bank1 is None:
            return

        memory_kv = self.agent_memory_bank.get_memory_kv(
            device=self.kv_bank1[0]["k"].device
        )

        if memory_kv is None:
            return

        cache_key = self.agent_memory_bank._memory_kv_cache_key
        if cache_key is not None and cache_key == self._last_injected_memory_key:
            return

        memory_length = memory_kv[0]["k"].shape[1]
        num_frames_in_memory = memory_length // self.frame_seq_length

        for block_idx in range(len(self.kv_bank1)):
            bank = self.kv_bank1[block_idx]
            bank_size = bank["k"].shape[1]

            write_length = min(memory_length, bank_size)
            bank["k"][:, :write_length] = memory_kv[block_idx]["k"][:, :write_length]
            bank["v"][:, :write_length] = memory_kv[block_idx]["v"][:, :write_length]

            if write_length < bank_size:
                bank["k"][:, write_length:].zero_()
                bank["v"][:, write_length:].zero_()

            bank["local_end_index"].fill_(write_length)
            bank["global_end_index"].fill_(write_length)

        self._iam_bank_length = write_length

        self._last_injected_memory_key = cache_key

        if DEBUG:
            print(
                f"[AgentPipeline] Injected {memory_length} tokens from IAM to kv_bank (all {len(self.kv_bank1)} blocks)"
            )


    def get_agent_status(self) -> Dict[str, Any]:
        return {
            "current_prompt_id": self.current_prompt_id,
            "current_chunk_id": self.current_chunk_id,
            "current_entities": [e.to_dict() for e in self.current_entities],
            "global_registry": self.agent_memory_bank.global_registry,
            "frame_archive_count": len(self.agent_memory_bank.frame_archive),
            "active_memory": self.agent_memory_bank.frame_active_memory,
        }

    def save_agent_state(self, path: str) -> None:
        self.agent_memory_bank.save_to_json(path)

    def load_agent_state(self, path: str) -> None:
        self.agent_memory_bank.load_from_json(path)




def create_agent_pipeline(
    config,
    device: torch.device,
    llm_model_path: str = "../Qwen3-4B-Instruct-2507",
    max_memory_frames: int = 3,
    save_dir: str = "data/agent_frames",
    use_vllm: bool = True,
    gpu_memory_utilization: float = 0.2,
    vlm_model_path: str = "../Qwen3-VL-2B-Instruct",
    vlm_enabled: bool = False,
    vlm_gpu_memory_utilization: float = 0.15,
    vlm_score_weight: float = 0.3,
    async_vae_enabled: bool = True,
) -> AgentCausalInferencePipeline:
    pipeline = AgentCausalInferencePipeline(
        args=config,
        device=device,
        llm_model_path=llm_model_path,
        max_memory_frames=max_memory_frames,
        save_dir=save_dir,
        use_vllm=use_vllm,
        gpu_memory_utilization=gpu_memory_utilization,
        vlm_model_path=vlm_model_path,
        vlm_enabled=vlm_enabled,
        vlm_gpu_memory_utilization=vlm_gpu_memory_utilization,
        vlm_score_weight=vlm_score_weight,
        async_vae_enabled=async_vae_enabled,
    )
    return pipeline

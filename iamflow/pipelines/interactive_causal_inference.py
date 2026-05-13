# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from typing import List, Optional, Union
import torch

from iamflow.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from iamflow.utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from iamflow.utils.transition_scheduler import create_scheduler, TransitionScheduler, AdaptiveScheduler
from iamflow.pipelines.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from iamflow.utils.debug_option import DEBUG


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    _TRANSITION_STRATEGY_ALIASES = {
        "soft_prompt_transition": "soft_prompt_transition",
        "soft": "soft_prompt_transition",
        "spt": "soft_prompt_transition",
        "kv_recache": "kv_recache",
        "recache": "kv_recache",
        "no_transition": "no_transition",
        "none": "no_transition",
        "keep_old_cache": "no_transition",
    }

    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
        use_lightvae: bool = False,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae, use_lightvae=use_lightvae)
        self.global_sink = getattr(args, "global_sink", False)
        self.transition_strategy = self._resolve_transition_strategy(args)

        # SPT (Soft Prompt Transition) state
        self.spt_enabled = self.transition_strategy == "soft_prompt_transition"
        self.prev_crossattn_cache = None
        self.frames_since_switch = None
        self.transition_scheduler: Optional[Union[TransitionScheduler, AdaptiveScheduler]] = None

        if self.spt_enabled:
            spt_config = getattr(args, "spt", None)
            if spt_config:
                scheduler_type = getattr(spt_config, "scheduler_type", "cosine")
                if scheduler_type == "adaptive":
                    self.transition_scheduler = AdaptiveScheduler(
                        base_scheduler_type=getattr(spt_config, "base_scheduler", "cosine"),
                        min_window=getattr(spt_config, "min_window", 3),
                        max_window=getattr(spt_config, "max_window", 15),
                        delay_frames=getattr(spt_config, "delay_frames", 3),
                        steepness=getattr(spt_config, "steepness", 6.0),
                        frames_per_chunk=getattr(spt_config, "frames_per_chunk", 3),
                    )
                else:
                    self.transition_scheduler = create_scheduler(
                        scheduler_type=scheduler_type,
                        window_frames=getattr(spt_config, "window_frames", 9),
                        delay_frames=getattr(spt_config, "delay_frames", 3),
                        steepness=getattr(spt_config, "steepness", 6.0),
                        frames_per_chunk=getattr(spt_config, "frames_per_chunk", 3),
                    )
            else:
                self.transition_scheduler = create_scheduler()
            print(f"[SPT] Enabled with {type(self.transition_scheduler).__name__}")
        else:
            print(f"[Transition] Strategy: {self.transition_strategy}")

    @classmethod
    def _resolve_transition_strategy(cls, args) -> str:
        raw_strategy = getattr(args, "transition_strategy", None)
        if raw_strategy is None:
            raw_strategy = (
                "soft_prompt_transition"
                if getattr(args, "spt_enabled", False)
                else "kv_recache"
            )
        key = str(raw_strategy).strip().lower()
        if key not in cls._TRANSITION_STRATEGY_ALIASES:
            valid = ", ".join(sorted(set(cls._TRANSITION_STRATEGY_ALIASES.values())))
            raise ValueError(
                f"Unsupported transition_strategy: {raw_strategy}. Valid values: {valid}"
            )
        return cls._TRANSITION_STRATEGY_ALIASES[key]

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_bank=self.kv_bank1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                update_bank=False,
                update_cache=True,
                q_bank=True,
                is_recache=True,
            )
        
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def _soft_switch(self):
        if self.prev_crossattn_cache is not None and len(self.prev_crossattn_cache) == len(self.crossattn_cache):
            for prev_blk, blk in zip(self.prev_crossattn_cache, self.crossattn_cache):
                prev_blk["k"].copy_(blk["k"])
                prev_blk["v"].copy_(blk["v"])
        else:
            self.prev_crossattn_cache = [
                {"k": blk["k"].clone(), "v": blk["v"].clone()}
                for blk in self.crossattn_cache
            ]
        self.frames_since_switch = 0

        for blk in self.crossattn_cache:
            blk["is_init"] = False

        print(f"[SPT] Soft switch initiated, transition window: {self.transition_scheduler.total_frames} frames")

    def _hard_switch_without_transition(self):
        """
        Switch to the new prompt without SPT blending or temporal recache.
        Preserve temporal KV cache and only invalidate prompt-specific cross-attention cache.
        """
        self.frames_since_switch = None
        if self.crossattn_cache is None:
            return
        for blk in self.crossattn_cache:
            blk["is_init"] = False
        if getattr(self, "kv_bank1", None) is not None:
            for blk in self.kv_bank1:
                if "local_end_index" in blk:
                    blk["local_end_index"].zero_()
                if "global_end_index" in blk:
                    blk["global_end_index"].zero_()
        self._iam_bank_length = 0
        self._last_injected_memory_key = None
        print("[Transition] No-transition switch: preserving temporal KV cache")

    def _handle_prompt_switch(self, output, current_start_frame, cond_old, cond_new):
        if self.transition_strategy == "soft_prompt_transition":
            if isinstance(self.transition_scheduler, AdaptiveScheduler):
                semantic_distance = self._compute_prompt_distance(cond_old, cond_new)
                self.transition_scheduler.update_for_switch(semantic_distance)
            self._soft_switch()
            return

        if self.transition_strategy == "kv_recache":
            self._recache_after_switch(output, current_start_frame, cond_new)
            return

        if self.transition_strategy == "no_transition":
            self._hard_switch_without_transition()
            return

        raise ValueError(f"Unsupported transition_strategy: {self.transition_strategy}")

    def _get_current_transition_alpha(self) -> Optional[float]:
        if self.frames_since_switch is None:
            return None
        return self.transition_scheduler.get_alpha(self.frames_since_switch)

    def _update_transition_state(self, num_frames: int):
        if self.frames_since_switch is None:
            return

        self.frames_since_switch += num_frames

        if self.transition_scheduler.is_complete(self.frames_since_switch):
            self.frames_since_switch = None
            print("[SPT] Transition complete")

    @staticmethod
    def _compute_prompt_distance(cond_old: dict, cond_new: dict) -> float:
        import torch.nn.functional as F
        # prompt_embeds shape: [B, 512, D]
        emb_old = cond_old["prompt_embeds"].mean(dim=(0, 1))  # [D]
        emb_new = cond_new["prompt_embeds"].mean(dim=(0, 1))  # [D]
        cos_sim = F.cosine_similarity(emb_old.unsqueeze(0), emb_new.unsqueeze(0))
        return (1.0 - cos_sim.item())

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        self.frames_since_switch = None
        
        # encode all prompts
        print(text_prompts_list)
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        kv_bank1_size = self.bank_size * self.frame_seq_length
        self._initialize_kv_bank(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device, kv_bank1_size=kv_bank1_size
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        for current_num_frames in all_num_frames:
            if (current_start_frame)%(3*self.record_interval)==0:
                update_bank=True
            else:
                update_bank=False
            
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._handle_prompt_switch(
                    output,
                    current_start_frame,
                    cond_list[segment_idx - 1],
                    cond_list[segment_idx],
                )
                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]

            transition_alpha = self._get_current_transition_alpha() if self.spt_enabled else None

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):

                if index==0:
                    q_bank=True
                else:
                    q_bank=False
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
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
                        prev_crossattn_cache=self.prev_crossattn_cache,
                        transition_alpha=transition_alpha,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
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
                        prev_crossattn_cache=self.prev_crossattn_cache,
                        transition_alpha=transition_alpha,
                    )

            # Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # rerun with clean context to update cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_bank=self.kv_bank1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                update_bank=update_bank,
                q_bank=q_bank,
                update_cache=True,
                prev_crossattn_cache=self.prev_crossattn_cache,
                transition_alpha=transition_alpha,
            )

            if self.spt_enabled:
                self._update_transition_state(current_num_frames)

            # Update frame pointer
            current_start_frame += current_num_frames

        # Standard decoding
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        self.clear_kv_cache()
        if return_latents:
            return video, output
        return video 

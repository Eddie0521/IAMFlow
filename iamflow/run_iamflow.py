# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import argparse
import os
import sys
from typing import List, Optional

# Ensure package imports are resolvable in vLLM spawn workers.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_THIS_DIR)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from einops import rearrange

from iamflow.utils.misc import set_seed
from iamflow.utils.distributed import barrier
from iamflow.utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from iamflow.utils.quantized_checkpoint import load_generator_for_inference, normalize_quantization_config

from iamflow.pipelines.agent_causal_inference import AgentCausalInferencePipeline
from iamflow.utils.dataset import MultiTextDataset


def _align_switch_indices(indices: List[int], block: int) -> List[int]:
    """Align switch frames to block boundaries to avoid mid-block semantic jumps."""
    if block <= 1:
        return indices
    aligned = []
    for idx in indices:
        snapped = (idx // block) * block
        aligned.append(snapped)
    return aligned


def _parse_seed_list(raw: Optional[str], default_seed: int) -> List[int]:
    """Parse comma- or space-separated seeds while preserving order."""
    if raw is None or not str(raw).strip():
        return [int(default_seed)]
    seeds = [int(item) for item in str(raw).replace(",", " ").split()]
    if not seeds:
        raise ValueError("No valid seeds were provided")
    return seeds


# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("IAM Agent causal inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
# Path override arguments
parser.add_argument("--data_path", type=str, default=None, help="Override data_path in config")
parser.add_argument("--output_folder", type=str, default=None, help="Override output_folder in config")
parser.add_argument("--generator_ckpt", type=str, default=None, help="Override generator_ckpt in config")
parser.add_argument("--lora_ckpt", type=str, default=None, help="Override lora_ckpt in config")
parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma- or space-separated seeds to run sequentially in one initialized process")
parser.add_argument("--dit_quantized_ckpt", type=str, default=None, help="Override dit_quantized_ckpt in config")
parser.add_argument("--dit_quant_scheme", type=str, default=None, help="Override dit_quant_scheme in config")
parser.add_argument("--disable_lora_ckpt", action="store_true", help="Disable lora_ckpt for pre-quantized inference")
# IAM-specific arguments
parser.add_argument("--llm_model_path", type=str, default="pretrained/Qwen3-4B-Instruct-2507",
                    help="Path to the LLM model for entity extraction")
parser.add_argument("--vlm_model_path", type=str, default=None,
                    help="Override vlm_model_path in config")
parser.add_argument("--max_memory_frames", type=int, default=3,
                    help="Maximum number of memory frames to keep")
parser.add_argument("--save_dir", type=str, default="data/agent_frames",
                    help="Directory to save frame data")
parser.add_argument("--mapping_path", type=str, default=None,
                    help="Path to save mapping.json (default: output_folder/mapping.json)")
parser.add_argument("--save_frames_to_disk", action="store_true",
                    help="Save frame KV to disk (default: False, keep in memory only for better performance)")
parser.add_argument("--use_tinyvae", "--use_lightvae", dest="use_tinyvae", action="store_true",
                    help="Use TinyVAE (tinyvaew2_1.pth) instead of standard VAE")
parser.add_argument("--llm_device", type=str, default=None,
                    help="GPU device for LLM/VLM (e.g. 'cuda:1'). Default: same as DiT device.")
parser.add_argument("--ablation_mode", type=str, default=None,
                    help="Ablation mode: full | no_vlm | no_llm | no_memory")
parser.add_argument("--transition_strategy", type=str, default=None,
                    help="Prompt switch strategy: soft_prompt_transition | kv_recache | no_transition")
parser.add_argument("--sample_index_offset", type=int, default=None,
                    help="Add an offset to dataset idx when naming outputs and mapping files")


def main():
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # Override config with command line arguments
    if args.data_path: config.data_path = args.data_path
    if args.output_folder: config.output_folder = args.output_folder
    if args.generator_ckpt: config.generator_ckpt = args.generator_ckpt
    if args.lora_ckpt: config.lora_ckpt = args.lora_ckpt
    if args.seed is not None: config.seed = args.seed
    if args.dit_quantized_ckpt: config.dit_quantized_ckpt = args.dit_quantized_ckpt
    if args.dit_quant_scheme: config.dit_quant_scheme = args.dit_quant_scheme
    if args.vlm_model_path: config.vlm_model_path = args.vlm_model_path
    if args.ablation_mode:
        config.ablation_mode = args.ablation_mode
    if args.transition_strategy:
        config.transition_strategy = args.transition_strategy
    if args.disable_lora_ckpt:
        config.lora_ckpt = None
    if args.sample_index_offset is not None:
        config.sample_index_offset = args.sample_index_offset
    normalize_quantization_config(config)

    seed_list = _parse_seed_list(args.seeds, int(getattr(config, "seed", 0)))
    config.seed = seed_list[0]
    multi_seed_mode = args.seeds is not None
    base_output_folder = str(config.output_folder)

    # ----------------------------- Distributed setup -----------------------------
    if "LOCAL_RANK" in os.environ:
        os.environ["NCCL_CROSS_NIC"] = "1"
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))

        # Set device first
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Initialize process group with backend and timeout
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout
            )

        set_seed(config.seed)
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.seed)
        print(f"Single GPU mode on device {device}")

    torch.set_grad_enabled(False)

    # ----------------------------- Create IAM Pipeline -----------------------------
    # Resolve IAM parameters: command line args override config file
    llm_model_path = args.llm_model_path if args.llm_model_path != "pretrained/Qwen3-4B-Instruct-2507" else getattr(config, "llm_model_path", "pretrained/Qwen3-4B-Instruct-2507")
    max_memory_frames = args.max_memory_frames if args.max_memory_frames != 3 else getattr(config, "max_memory_frames", 3)
    save_dir = args.save_dir if args.save_dir != "data/agent_frames" else getattr(config, "save_dir", "data/agent_frames")
    save_frames_to_disk = args.save_frames_to_disk or getattr(config, "save_frames_to_disk", False)
    use_tinyvae = args.use_tinyvae or getattr(config, "use_tinyvae", getattr(config, "use_lightvae", False))
    use_vllm = bool(getattr(config, "use_vllm", True))
    gpu_memory_utilization = float(getattr(config, "gpu_memory_utilization", 0.2))

    # VLM config
    vlm_enabled = getattr(config, "vlm_enabled", False)
    vlm_model_path = getattr(config, "vlm_model_path", "pretrained/Qwen3-VL-2B-Instruct")
    vlm_score_weight = float(getattr(config, "vlm_score_weight", 0.3))
    vlm_gpu_memory_utilization = float(getattr(config, "vlm_gpu_memory_utilization", 0.15))
    async_vae_enabled = getattr(config, "async_vae_enabled", True)
    ablation_mode = str(getattr(config, "ablation_mode", "full")).lower()
    memory_allocation_mode = str(getattr(config, "memory_allocation_mode", "dynamic")).lower()
    frame_selection_score_mode = str(getattr(config, "frame_selection_score_mode", "fused")).lower()
    fixed_id_memory_frames = getattr(config, "fixed_id_memory_frames", None)
    max_id_memory_frames = int(getattr(config, "max_id_memory_frames", 4))

    effective_vlm_enabled = bool(vlm_enabled) and ablation_mode not in {"no_vlm", "no_llm", "no_memory"}

    # ----------------------------- Device placement (multi-GPU) -----------------------------
    def _parse_device_id(raw) -> Optional[int]:
        """Parse 'cuda:N' or 'N' to int device_id, None if unset."""
        if raw is None:
            return None
        s = str(raw)
        if s.startswith("cuda:"):
            return int(s.split(":")[1])
        return int(s)

    def _parse_torch_device(raw, fallback: torch.device) -> torch.device:
        """Parse 'cuda:N' to torch.device, fallback if unset."""
        if raw is None:
            return fallback
        return torch.device(str(raw))

    # DiT / VAE / TextEncoder → torch.device (for .to() calls)
    dit_device = _parse_torch_device(
        getattr(config, "dit_device", None), fallback=device)
    vae_device = _parse_torch_device(
        getattr(config, "vae_device", None), fallback=device)
    text_encoder_device = _parse_torch_device(
        getattr(config, "text_encoder_device", None), fallback=device)

    # LLM / VLM → int device_id (for vLLM CUDA_VISIBLE_DEVICES pinning)
    # CLI --llm_device overrides config
    llm_device_id = _parse_device_id(
        args.llm_device or getattr(config, "llm_device", None))
    vlm_device_id = _parse_device_id(
        getattr(config, "vlm_device", None))
    # If vlm_device not specified, fall back to llm_device
    if vlm_device_id is None:
        vlm_device_id = llm_device_id

    if local_rank == 0:
        print("=" * 60)
        print("IAM Agent Causal Inference Pipeline")
        print("=" * 60)
        print(f"LLM Model Path: {llm_model_path}")
        print(f"Max Memory Frames: {max_memory_frames}")
        print(f"Save Directory: {save_dir}")
        print(f"Seeds: {seed_list}")
        print(f"Save Frames to Disk: {save_frames_to_disk}")
        print(f"LLM Backend: {'vllm' if use_vllm else 'hf'}")
        print(f"Ablation Mode: {ablation_mode}")
        print(
            "Memory Allocation: "
            f"{memory_allocation_mode}"
            + (
                f" (fixed_id_memory_frames={fixed_id_memory_frames})"
                if memory_allocation_mode == "fixed"
                else ""
            )
        )
        print(f"Max ID Memory Frames: {max_id_memory_frames}")
        print(f"Frame Selection Score Mode: {frame_selection_score_mode}")
        print(f"Use TinyVAE: {use_tinyvae}")
        print(f"VLM Enabled: {effective_vlm_enabled}")
        if effective_vlm_enabled:
            print(f"VLM Model Path: {vlm_model_path}")
            print(f"VLM Score Weight: {vlm_score_weight}")
        print(f"Async VAE: {async_vae_enabled}")
        print(f"DiT Quantized: {getattr(config, 'dit_quantized', False)}")
        print(f"DiT Quant Scheme: {getattr(config, 'dit_quant_scheme', 'none')}")
        print(f"DiT Quantized Ckpt: {getattr(config, 'dit_quantized_ckpt', None)}")
        print(f"Device Placement: DiT={dit_device}, VAE={vae_device}, "
              f"TextEncoder={text_encoder_device}, "
              f"LLM={'cuda:'+str(llm_device_id) if llm_device_id is not None else device}, "
              f"VLM={'cuda:'+str(vlm_device_id) if vlm_device_id is not None else device}")
        print("=" * 60)

    pipeline = AgentCausalInferencePipeline(
        config,
        device=device,
        llm_model_path=llm_model_path,
        max_memory_frames=max_memory_frames,
        save_dir=save_dir,
        save_frames_to_disk=save_frames_to_disk,
        use_vllm=use_vllm,
        gpu_memory_utilization=gpu_memory_utilization,
        use_lightvae=use_tinyvae,
        vlm_model_path=vlm_model_path,
        vlm_enabled=vlm_enabled,
        vlm_gpu_memory_utilization=vlm_gpu_memory_utilization,
        vlm_score_weight=vlm_score_weight,
        async_vae_enabled=async_vae_enabled,
        llm_device_id=llm_device_id,
        vlm_device_id=vlm_device_id,
    )

    # ----------------------------- Load generator checkpoint -----------------------------
    checkpoint_route = load_generator_for_inference(pipeline.generator, config)
    if local_rank == 0:
        print(f"Generator checkpoint route: {checkpoint_route}")

    # --------------------------- LoRA support (optional) ---------------------------
    from iamflow.utils.lora_utils import configure_lora_for_model
    import peft

    pipeline.is_lora_enabled = False
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if checkpoint_route == "lightx2v_fp8":
        if local_rank == 0:
            print("Using pre-quantized LightX2V FP8 checkpoint; skipping LoRA attach/load.")
    elif getattr(config, "adapter", None) and configure_lora_for_model is not None:
        if local_rank == 0:
            print(f"LoRA enabled with config: {config.adapter}")
            print("Applying LoRA to generator (inference)...")
        # After loading base weights, apply LoRA wrapper to the generator's transformer model
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=(local_rank == 0),
        )

        # Load LoRA weights (if lora_ckpt is provided)
        if lora_ckpt_path:
            if local_rank == 0:
                print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            # Support both formats: containing the `generator_lora` key or a raw LoRA state dict
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
            if local_rank == 0:
                print("LoRA weights loaded for generator")
        else:
            if local_rank == 0:
                print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

        pipeline.is_lora_enabled = True

    # Move pipeline to appropriate dtype and device
    print("dtype", pipeline.generator.model.dtype)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    low_memory = get_cuda_free_memory_gb(dit_device) < 40 if torch.cuda.is_available() else False
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=text_encoder_device)
    pipeline.generator.to(device=dit_device)
    pipeline.vae.to(device=vae_device)

    # ----------------------------- Build dataset -----------------------------
    # Parse switch_frame_indices
    switch_frame_indices: List[int] = [int(x) for x in config.switch_frame_indices.split(",") if x.strip()]
    block_size = int(getattr(config, "num_frame_per_block", 1))
    aligned_switch_indices = _align_switch_indices(switch_frame_indices, block_size)
    if aligned_switch_indices != switch_frame_indices and local_rank == 0:
        print(
            f"[Warning] switch_frame_indices {switch_frame_indices} are not aligned to "
            f"num_frame_per_block={block_size}, using {aligned_switch_indices}."
        )
    switch_frame_indices = aligned_switch_indices

    # Create dataset
    dataset = MultiTextDataset(config.data_path)

    # Validate number of segments & switch_frame_indices length
    num_segments = len(dataset[0]["prompts_list"])
    assert len(switch_frame_indices) == num_segments - 1, (
        "The number of switch_frame_indices should be the number of prompt segments minus 1"
    )

    print("Number of segments:", num_segments)
    print("Switch frame indices:", switch_frame_indices)

    num_prompts_total = len(dataset)
    sample_index_offset = int(getattr(config, "sample_index_offset", 0))
    print(f"Number of prompt lines: {num_prompts_total}")
    print(f"Sample index offset: {sample_index_offset}")

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    # Create output directory
    if local_rank == 0:
        os.makedirs(base_output_folder, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Determine model type for filename
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "iam_lora"
    elif getattr(config, 'use_ema', False):
        model_type = "iam_ema"
    else:
        model_type = "iam"

    # ----------------------------- Inference loop -----------------------------
    for seed_run_idx, run_seed in enumerate(seed_list, start=1):
        config.seed = int(run_seed)
        set_seed(config.seed)

        if multi_seed_mode:
            current_output_folder = os.path.join(
                base_output_folder,
                f"seed_{config.seed}_samples_{config.num_samples}",
            )
            current_save_dir = os.path.join(save_dir, f"seed_{config.seed}")
        else:
            current_output_folder = base_output_folder
            current_save_dir = save_dir

        if local_rank == 0:
            os.makedirs(current_output_folder, exist_ok=True)
            os.makedirs(current_save_dir, exist_ok=True)
            print("=" * 60)
            print(f"[IAM] Seed run {seed_run_idx}/{len(seed_list)}: seed={config.seed}")
            print(f"[IAM] Output folder: {current_output_folder}")
            print("=" * 60)

        if hasattr(pipeline, "save_dir"):
            pipeline.save_dir = current_save_dir
        if hasattr(pipeline, "agent_memory_bank"):
            pipeline.agent_memory_bank.save_dir = current_save_dir

        if dist.is_initialized():
            dist.barrier()

        progress = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=(local_rank != 0),
            desc=f"seed={config.seed}",
        )
        for i, batch_data in progress:
            idx = batch_data["idx"].item()
            effective_idx = idx + sample_index_offset
            prompts_list: List[str] = batch_data["prompts_list"]  # type: ignore

            sampled_noise = torch.randn(
                [
                    config.num_samples,
                    config.num_output_frames,
                    16,
                    60,
                    104,
                ],
                device=dit_device,
                dtype=torch.bfloat16,
            )

            # Determine mapping path for this sample
            if args.mapping_path and not multi_seed_mode:
                mapping_path = args.mapping_path
            else:
                mapping_path = os.path.join(current_output_folder, f"mapping_{effective_idx}.json")

            # IAM Agent inference
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts_list=prompts_list,
                switch_frame_indices=switch_frame_indices,
                return_latents=False,
                low_memory=low_memory,
                save_mapping=True,
                mapping_path=mapping_path,
                profile=True,  # Enable profiling
            )

            current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

            for seed_idx in range(config.num_samples):
                if config.save_with_index:
                    output_path = os.path.join(current_output_folder, f"rank{rank}-{effective_idx}-{seed_idx}_{model_type}.mp4")
                else:
                    # Use the first prompt segment as the filename prefix to avoid overly long names
                    short_name = prompts_list[0][0][:100].replace("/", "_")
                    output_path = os.path.join(current_output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")
                write_video(output_path, current_video[seed_idx].to(torch.uint8), fps=16)

            if local_rank == 0:
                print(f"[IAM] Saved video to {output_path}")
                print(f"[IAM] Saved mapping to {mapping_path}")

                # Print agent status
                status = pipeline.get_agent_status()
                print(f"[IAM] Final registry entities: {len(status['global_registry'])}")
                print(f"[IAM] Frame archive count: {status['frame_archive_count']}")
                print(f"[IAM] Active memory frames: {status['active_memory']}")

            if config.inference_iter != -1 and i >= config.inference_iter:
                break

    if dist.is_initialized():
        dist.destroy_process_group()

    if local_rank == 0:
        print("=" * 60)
        print("IAM Agent Inference Complete!")
        print("=" * 60)


if __name__ == '__main__':
    main()

import argparse
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_THIS_DIR)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from iamflow.utils.lora_utils import configure_lora_for_model
from iamflow.utils.quantized_checkpoint import (
    build_lightx2v_fp8_checkpoint,
    load_calibration_prompts,
    load_float_generator_checkpoint,
)
from iamflow.utils.quantize import collect_quantizable_linear_names
from iamflow.utils.wan_wrapper import WanDiffusionWrapper


def _collect_calibration_stats(
    generator,
    text_encoder,
    prompts,
    *,
    device,
    num_frames: int,
    timestep_value: int,
    batch_size: int,
):
    target_names = set(collect_quantizable_linear_names(generator))
    stats = {
        name: {"input_absmax": 0.0, "num_batches": 0}
        for name in target_names
    }
    handles = []

    module_map = dict(generator.named_modules())

    def _make_hook(name):
        def _hook(_module, inputs):
            if not inputs:
                return
            tensor = inputs[0]
            if not torch.is_tensor(tensor):
                return
            absmax = float(tensor.detach().float().abs().amax().item())
            stats[name]["input_absmax"] = max(stats[name]["input_absmax"], absmax)
            stats[name]["num_batches"] += 1
        return _hook

    for name in target_names:
        module = module_map.get(name)
        if module is not None:
            handles.append(module.register_forward_pre_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            for prompt_batch in _batched(prompts, batch_size):
                cond = text_encoder(text_prompts=prompt_batch)
                noise = torch.randn(
                    [len(prompt_batch), num_frames, 16, 60, 104],
                    device=device,
                    dtype=torch.bfloat16,
                )
                timestep = torch.ones(
                    [len(prompt_batch), num_frames],
                    device=device,
                    dtype=torch.int64,
                ) * timestep_value
                generator(
                    noisy_image_or_video=noise,
                    conditional_dict=cond,
                    timestep=timestep,
                )
    finally:
        for handle in handles:
            handle.remove()

    return stats


def _merge_lora_if_needed(generator, config, lora_ckpt_path):
    if not lora_ckpt_path:
        return
    if not getattr(config, "adapter", None):
        raise ValueError("LoRA checkpoint provided, but config.adapter is missing.")

    import peft

    generator.model = configure_lora_for_model(
        generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )

    lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
    if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
        peft.set_peft_model_state_dict(generator.model, lora_checkpoint["generator_lora"])
    else:
        peft.set_peft_model_state_dict(generator.model, lora_checkpoint)

    if not hasattr(generator.model, "merge_and_unload"):
        raise RuntimeError("Current PEFT model does not support merge_and_unload().")
    generator.model = generator.model.merge_and_unload()


def main():
    parser = argparse.ArgumentParser("Export merged LightX2V-style FP8 generator checkpoint")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--generator_ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--merged_output_path", type=str, default=None)
    parser.add_argument("--quant_scheme", type=str, default="fp8-vllm")
    parser.add_argument("--calibration_prompts", type=str, default=None)
    parser.add_argument("--calibration_max_prompts", type=int, default=256)
    parser.add_argument("--calibration_batch_size", type=int, default=4)
    parser.add_argument("--calibration_num_frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.quant_scheme not in {"fp8-vllm", "fp8-sgl"}:
        raise ValueError("Phase 1 only supports fp8-vllm and fp8-sgl.")
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 export requires CUDA.")

    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config_path)
    if args.generator_ckpt:
        config.generator_ckpt = args.generator_ckpt
    if args.lora_ckpt:
        config.lora_ckpt = args.lora_ckpt

    if not getattr(config, "generator_ckpt", None):
        raise ValueError("generator_ckpt must be set before export.")

    device = torch.device("cuda")

    generator = WanDiffusionWrapper(
        **getattr(config, "model_kwargs", {}),
        is_causal=True,
    ).eval()
    load_float_generator_checkpoint(
        generator,
        config.generator_ckpt,
        use_ema=bool(getattr(config, "use_ema", False)),
    )
    _merge_lora_if_needed(generator, config, getattr(config, "lora_ckpt", None))
    generator = generator.to(device=device, dtype=torch.bfloat16).eval()

    if args.merged_output_path:
        os.makedirs(os.path.dirname(args.merged_output_path) or ".", exist_ok=True)
        torch.save({"generator": generator.state_dict()}, args.merged_output_path)

    calibration_stats = None
    prompts = []
    if args.calibration_prompts:
        from iamflow.utils.wan_wrapper import WanTextEncoder

        prompts = load_calibration_prompts(
            args.calibration_prompts,
            max_prompts=args.calibration_max_prompts,
        )
        if not prompts:
            raise ValueError("No calibration prompts loaded.")

        text_encoder = WanTextEncoder().to(device=device).eval()
        calibration_stats = _collect_calibration_stats(
            generator,
            text_encoder,
            prompts,
            device=device,
            num_frames=args.calibration_num_frames,
            timestep_value=int(getattr(config, "denoising_step_list", [1000])[0]),
            batch_size=args.calibration_batch_size,
        )

    checkpoint = build_lightx2v_fp8_checkpoint(
        generator,
        metadata={
            "route": "lightx2v_fp8",
            "quant_scheme": args.quant_scheme,
        },
    )
    if prompts:
        checkpoint["quantization"]["calibration_prompt_count"] = len(prompts)
        checkpoint["quantization"]["calibration_num_frames"] = args.calibration_num_frames

    metadata = {
        key: str(value)
        for key, value in checkpoint["quantization"].items()
    }
    if calibration_stats is not None:
        metadata["calibration_stats"] = str(
            {
                name: {
                    "input_absmax": round(stat["input_absmax"], 6),
                    "num_batches": stat["num_batches"],
                }
                for name, stat in calibration_stats.items()
            }
        )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_file(checkpoint["generator"], args.output_path, metadata=metadata)
    print(f"Saved LightX2V-style FP8 checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()

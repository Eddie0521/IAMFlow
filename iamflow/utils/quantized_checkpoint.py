from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch

from .quantize import collect_quantizable_linear_names, prepare_model_for_lightx2v_fp8_load


SUPPORTED_FP8_SCHEMES = {
    "none",
    "fp8-vllm",
    "fp8-sgl",
    "fp8-q8f",
    "fp8-b128-deepgemm",
}


@dataclass(frozen=True)
class GeneratorCheckpointRoute:
    kind: str
    path: str


def resolve_quant_scheme(config: Any) -> str:
    scheme = getattr(config, "dit_quant_scheme", None)
    if scheme not in (None, ""):
        return str(scheme)

    legacy = getattr(config, "quantize_linear", None)
    if legacy in (None, "", "none"):
        return "none"
    if legacy == "fp8":
        return "fp8-vllm"
    return str(legacy)


def normalize_quantization_config(config: Any) -> Any:
    scheme = resolve_quant_scheme(config)
    setattr(config, "dit_quant_scheme", scheme)

    quantized_flag = bool(getattr(config, "dit_quantized", False))
    if scheme != "none" or bool(getattr(config, "dit_quantized_ckpt", None)):
        quantized_flag = True
    setattr(config, "dit_quantized", quantized_flag)

    if not hasattr(config, "dit_quantized_ckpt"):
        setattr(config, "dit_quantized_ckpt", None)

    return config


def detect_generator_checkpoint_route(path: str | None) -> GeneratorCheckpointRoute:
    if not path:
        return GeneratorCheckpointRoute(kind="float_pt", path="")
    if Path(path).suffix.lower() == ".safetensors":
        return GeneratorCheckpointRoute(kind="lightx2v_fp8", path=path)
    return GeneratorCheckpointRoute(kind="float_pt", path=path)


def validate_lora_compatibility(route_kind: str, lora_ckpt_path: str | None) -> None:
    if route_kind == "lightx2v_fp8" and lora_ckpt_path:
        raise ValueError(
            "LoRA checkpoint loading is incompatible with a pre-quantized LightX2V FP8 generator checkpoint. "
            "Please merge LoRA before PTQ export."
        )


def extract_generator_state_dict(checkpoint: dict[str, Any], *, use_ema: bool) -> dict[str, torch.Tensor]:
    if use_ema and "generator_ema" in checkpoint:
        return checkpoint["generator_ema"]
    if "generator" in checkpoint:
        return checkpoint["generator"]
    if "model" in checkpoint:
        return checkpoint["model"]
    raise ValueError("Generator state dict not found in checkpoint.")


def clean_generator_state_dict_for_ema(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("_fsdp_wrapped_module.", ""): value
        for key, value in state_dict.items()
    }


def load_safetensors_state_dict(path: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)
    return state_dict


def load_calibration_prompts(path: str, *, max_prompts: int) -> list[str]:
    prompts: list[str] = []
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record.get("prompt"), str):
                prompts.append(record["prompt"])
            elif isinstance(record.get("prompts"), list):
                prompts.extend(p for p in record["prompts"] if isinstance(p, str))
            elif isinstance(record.get("prompts_list"), list):
                prompts.extend(p for p in record["prompts_list"] if isinstance(p, str))
            if len(prompts) >= max_prompts:
                break
    else:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= max_prompts:
                break
    return prompts[:max_prompts]


def _quantize_weight_fp8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.detach().float()
    max_val = weight_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = (max_val / finfo.max).to(torch.float32)
    scaled = torch.clamp(weight_fp32 / scale, min=finfo.min, max=finfo.max)
    try:
        from qtorch.quant import float_quantize

        quantized = float_quantize(scaled, 4, 3, rounding="nearest").to(torch.float8_e4m3fn)
    except Exception:
        quantized = scaled.to(torch.float8_e4m3fn)
    return quantized, scale


def build_lightx2v_fp8_checkpoint(
    model: torch.nn.Module,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_dict = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    for module_name in collect_quantizable_linear_names(model):
        weight_key = f"{module_name}.weight"
        if weight_key not in state_dict:
            continue
        quantized_weight, weight_scale = _quantize_weight_fp8_per_channel(state_dict[weight_key])
        state_dict[weight_key] = quantized_weight
        state_dict[f"{module_name}.weight_scale"] = weight_scale

    quantization = {"route": "lightx2v_fp8"}
    if metadata:
        quantization.update(metadata)
    return {"generator": state_dict, "quantization": quantization}


def load_lightx2v_fp8_checkpoint(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    quant_scheme: str,
    linear_cls=None,
) -> str:
    route = prepare_model_for_lightx2v_fp8_load(
        model,
        quant_scheme=quant_scheme,
        linear_cls=linear_cls,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys while loading LightX2V FP8 checkpoint: {unexpected}")

    missing = [name for name in missing if not name.endswith(".weight_scale")]
    if missing:
        raise ValueError(f"Missing keys while loading LightX2V FP8 checkpoint: {missing}")
    return route


def load_float_generator_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    use_ema: bool,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_generator_state_dict(checkpoint, use_ema=use_ema)
    if use_ema:
        state_dict = clean_generator_state_dict_for_ema(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys while loading float generator checkpoint: {unexpected}")
        if missing:
            missing_preview = ", ".join(missing[:8])
            print(f"[Warning] {len(missing)} parameters missing while loading float checkpoint: {missing_preview}")
    else:
        model.load_state_dict(state_dict)
    return "float_pt"


def load_lightx2v_fp8_checkpoint_from_path(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    quant_scheme: str,
    linear_cls=None,
) -> str:
    state_dict = load_safetensors_state_dict(checkpoint_path)
    return load_lightx2v_fp8_checkpoint(
        model,
        state_dict,
        quant_scheme=quant_scheme,
        linear_cls=linear_cls,
    )


def load_generator_for_inference(
    model: torch.nn.Module,
    config: Any,
    *,
    linear_cls=None,
) -> str:
    normalize_quantization_config(config)

    if getattr(config, "dit_quantized", False):
        candidate_path = getattr(config, "dit_quantized_ckpt", None)
        if not candidate_path:
            generator_ckpt = getattr(config, "generator_ckpt", None)
            route = detect_generator_checkpoint_route(generator_ckpt)
            if route.kind == "lightx2v_fp8":
                candidate_path = route.path
        if not candidate_path:
            raise ValueError(
                "dit_quantized=true but no LightX2V FP8 checkpoint path was provided. "
                "Set dit_quantized_ckpt or point generator_ckpt to a .safetensors file."
            )
        validate_lora_compatibility("lightx2v_fp8", getattr(config, "lora_ckpt", None))
        return load_lightx2v_fp8_checkpoint_from_path(
            model,
            candidate_path,
            quant_scheme=getattr(config, "dit_quant_scheme"),
            linear_cls=linear_cls,
        )

    checkpoint_path = getattr(config, "generator_ckpt", None)
    if not checkpoint_path:
        return "none"
    return load_float_generator_checkpoint(
        model,
        checkpoint_path,
        use_ema=bool(getattr(config, "use_ema", False)),
    )

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BaseFP8Linear(nn.Module):
    quant_scheme = "fp8-base"

    # Buffers whose dtype must NOT change when the module is cast
    # (e.g. model.to(bfloat16)).  weight is float8 and naturally immune;
    # weight_scale must stay float32 for CUTLASS kernels.
    _fp8_protected_buffers = frozenset({"weight", "weight_scale"})

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, 1, dtype=torch.float32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    def _apply(self, fn, recurse=True):
        # Snapshot protected buffers before the generic _apply (which calls
        # fn on every parameter and buffer, potentially changing dtype).
        saved = {
            name: getattr(self, name).clone()
            for name in self._fp8_protected_buffers
            if hasattr(self, name) and getattr(self, name) is not None
        }
        result = super()._apply(fn, recurse=recurse)
        # Restore original dtype / data for protected buffers, but keep the
        # device change if _apply moved them (e.g. .cuda()).
        for name, original in saved.items():
            buf = getattr(self, name)
            if buf.dtype != original.dtype:
                setattr(self, name, original.to(device=buf.device))
        return result

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        quantize: bool,
        quant_scheme: str,
    ) -> "BaseFP8Linear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        if linear.bias is not None:
            module.bias.copy_(linear.bias.detach().to(torch.bfloat16))
        if quantize:
            weight, scale = _quantize_weight_fp8_per_channel(linear.weight.detach())
            module.weight.copy_(weight)
            module.weight_scale.copy_(scale)
        return module

    def _dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return self.weight.float().mul(self.weight_scale).to(dtype)

    def _cpu_fallback(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self._dequantized_weight(x.dtype), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self._cpu_fallback(x)
        if getattr(self, "_backend_failed", False):
            return self._cpu_fallback(x)
        try:
            return self._backend_forward(x)
        except Exception as exc:
            self._backend_failed = True
            warnings.warn(
                f"{self.quant_scheme} backend unavailable, falling back to dequantized linear: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._cpu_fallback(x)

    def _backend_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FP8VllmLinear(BaseFP8Linear):
    quant_scheme = "fp8-vllm"

    def _backend_forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm import _custom_ops as ops

        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]).contiguous()
        x_quant, x_scale = ops.scaled_fp8_quant(
            x_2d,
            None,
            scale_ub=None,
            use_per_token_if_dynamic=True,
        )
        out = torch.empty(
            (x_2d.shape[0], self.out_features),
            dtype=x.dtype,
            device=x.device,
        )
        torch.ops._C.cutlass_scaled_mm(
            out,
            x_quant,
            self.weight.t(),
            x_scale,
            self.weight_scale.float().t(),
            self.bias.to(x.dtype) if self.bias is not None else None,
        )
        return out.reshape(*x_shape[:-1], self.out_features)


class FP8SglLinear(BaseFP8Linear):
    quant_scheme = "fp8-sgl"

    def _backend_forward(self, x: torch.Tensor) -> torch.Tensor:
        import sgl_kernel

        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1]).contiguous()
        m, k = x_2d.shape
        x_quant = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device, requires_grad=False)
        x_scale = torch.empty((m, 1), dtype=torch.float32, device=x.device, requires_grad=False)
        sgl_kernel.sgl_per_token_quant_fp8(x_2d, x_quant, x_scale)
        out = sgl_kernel.fp8_scaled_mm(
            x_quant,
            self.weight.t(),
            x_scale,
            self.weight_scale.float().t(),
            x.dtype,
            self.bias.to(x.dtype) if self.bias is not None else None,
        )
        return out.reshape(*x_shape[:-1], self.out_features)


def load_fp8_linear_class(quant_scheme: str):
    if quant_scheme == "fp8-vllm":
        return FP8VllmLinear
    if quant_scheme == "fp8-sgl":
        return FP8SglLinear
    if quant_scheme in {"fp8-q8f", "fp8-b128-deepgemm"}:
        raise NotImplementedError(f"{quant_scheme} is planned but not implemented yet.")
    raise ValueError(f"Unsupported FP8 quantization scheme: {quant_scheme}")

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


_ATTN_NAMES = {"q", "k", "v", "o"}
_FFN_NAMES = {"fc1", "fc2"}


def _iter_quantizable_linear_names(model: nn.Module) -> Iterable[str]:
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        parts = name.split(".")
        if len(parts) < 4:
            continue

        offset = 1 if parts[0] == "model" else 0
        if len(parts) <= offset + 3:
            continue
        if parts[offset] != "blocks":
            continue
        if not parts[offset + 1].isdigit():
            continue

        group_name = parts[offset + 2]
        leaf_name = parts[offset + 3]

        if group_name in {"self_attn", "cross_attn"} and leaf_name in _ATTN_NAMES:
            yield name
        elif group_name == "ffn" and leaf_name in _FFN_NAMES:
            yield name


def collect_quantizable_linear_names(model: nn.Module) -> list[str]:
    return sorted(_iter_quantizable_linear_names(model))


def _resolve_parent_module(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def replace_generator_blocks_with_fp8(
    model: nn.Module,
    *,
    quant_scheme: str,
    quantize: bool,
    linear_cls=None,
) -> int:
    if linear_cls is None:
        try:
            from .fp8_linear import load_fp8_linear_class
        except ImportError:
            from iamflow.utils.fp8_linear import load_fp8_linear_class

        linear_cls = load_fp8_linear_class(quant_scheme)

    target_names = collect_quantizable_linear_names(model)
    for qualified_name in target_names:
        parent, leaf = _resolve_parent_module(model, qualified_name)
        original = getattr(parent, leaf)
        replacement = linear_cls.from_linear(
            original,
            quantize=quantize,
            quant_scheme=quant_scheme,
        )
        setattr(parent, leaf, replacement)
    return len(target_names)


def prepare_model_for_lightx2v_fp8_load(
    model: nn.Module,
    *,
    quant_scheme: str,
    linear_cls=None,
) -> str:
    replace_generator_blocks_with_fp8(
        model,
        quant_scheme=quant_scheme,
        quantize=False,
        linear_cls=linear_cls,
    )
    return "lightx2v_fp8"

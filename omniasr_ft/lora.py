"""Minimal, dependency-free LoRA for fairseq2 / omnilingual-asr models.

Works with any module that exposes a 2D ``weight`` (out_features, in_features)
and behaves like a linear layer (``fairseq2.nn.Linear`` and ``torch.nn.Linear``).
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

# Default targets: attention projections of the Llama decoder.
DEFAULT_TARGET_PATTERNS = [
    r"llama_decoder\..*\.(q_proj|k_proj|v_proj|output_proj)$",
]


class LoRALinear(nn.Module):
    """Wraps a frozen linear-like module with a trainable low-rank adapter."""

    def __init__(
        self,
        base: nn.Module,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not hasattr(base, "weight") or base.weight.dim() != 2:
            raise ValueError(f"Cannot apply LoRA to {type(base).__name__}")

        out_features, in_features = base.weight.shape
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        device = base.weight.device
        # Keep adapters in fp32 for training stability.
        self.lora_a = nn.Parameter(torch.empty(r, in_features, device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        x32 = self.lora_dropout(x.to(self.lora_a.dtype))
        update = (x32 @ self.lora_a.T) @ self.lora_b.T * self.scaling
        return result + update.to(result.dtype)

    @torch.no_grad()
    def merge(self) -> nn.Module:
        """Folds the adapter into the base weight and returns the base module."""
        delta = (self.lora_b.to(torch.float32) @ self.lora_a.to(torch.float32)) * self.scaling
        self.base.weight.data += delta.to(self.base.weight.dtype)
        return self.base


def _resolve_parent(model: nn.Module, qual_name: str) -> tuple[nn.Module, str]:
    parts = qual_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_patterns: Iterable[str] = DEFAULT_TARGET_PATTERNS,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> List[str]:
    """Replaces matching submodules with :class:`LoRALinear` wrappers in-place.

    Returns the list of module names that were wrapped.
    """
    compiled = [re.compile(p) for p in target_patterns]
    targets = [
        name
        for name, module in model.named_modules()
        if any(p.search(name) for p in compiled)
        and hasattr(module, "weight")
        and getattr(module, "weight").dim() == 2
        and not isinstance(module, LoRALinear)
    ]
    for name in targets:
        parent, attr = _resolve_parent(model, name)
        base = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
    if not targets:
        raise ValueError(f"No modules matched LoRA target patterns: {list(target_patterns)}")
    return targets


def mark_only_lora_as_trainable(model: nn.Module, extra_trainable: Iterable[str] = ()) -> int:
    """Freezes everything except LoRA params (and optional extra module prefixes).

    Returns the number of trainable parameters.
    """
    extra = tuple(extra_trainable)
    for name, p in model.named_parameters():
        is_lora = "lora_a" in name or "lora_b" in name
        is_extra = any(name.startswith(e) for e in extra)
        p.requires_grad_(is_lora or is_extra)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_a" in k or "lora_b" in k
    }


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected]
    if unexpected:
        raise ValueError(f"Unexpected keys in LoRA state dict: {unexpected[:5]}")
    loaded = [k for k in state]
    if not loaded:
        raise ValueError("Empty LoRA state dict")


def merge_lora(model: nn.Module) -> nn.Module:
    """Merges all LoRA adapters back into their base modules (in-place)."""
    lora_names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in lora_names:
        parent, attr = _resolve_parent(model, name)
        wrapper: LoRALinear = getattr(parent, attr)
        setattr(parent, attr, wrapper.merge())
    return model

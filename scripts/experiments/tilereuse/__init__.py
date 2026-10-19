"""Experimental sparse-attention reuse; importing this package is CPU safe."""

from .config import FIXED_CONFIGS, SHAPE_POLICY, TUNED_BASELINE, select
from .veomni import attention_context, make_attention


__all__ = [
    "FIXED_CONFIGS",
    "SHAPE_POLICY",
    "TUNED_BASELINE",
    "attention_context",
    "make_attention",
    "select",
]

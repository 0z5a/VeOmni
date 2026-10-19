"""Frozen H8/H16/H32 dispatch configurations from the attention experiment."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Configuration:
    group: int
    backward_block: int
    backward_head_tile: int
    pad_heads: bool


TUNED_BASELINE = "tuned-both"
SHAPE_POLICY = "shape-policy-both"
FIXED_CONFIGS = MappingProxyType(
    {
        TUNED_BASELINE: Configuration(1, 64, 32, True),
        **{f"g{group}-b32-m64-both": Configuration(group, 32, 64, False) for group in (2, 4, 8)},
    }
)
_SHAPE_RULE = MappingProxyType({8: "g8-b32-m64-both", 16: "g4-b32-m64-both", 32: "g2-b32-m64-both"})


def select(heads):
    """Select by the actual per-rank head count, with the tuned fallback.

    The fallback is a configuration choice, not qualification of unseen shapes.
    """
    return _SHAPE_RULE.get(heads, TUNED_BASELINE)

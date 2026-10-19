"""Opt-in bindings for a caller-supplied VeOmni native attention module."""

from contextlib import contextmanager
from types import SimpleNamespace

from .config import FIXED_CONFIGS, SHAPE_POLICY, select


def _require_gpu():
    from veomni.ops.kernels.deepseek_v4 import _require_tilelang_sm90

    _require_tilelang_sm90()


def _fixed_attention(native, name, profile):
    # The numerical adapter and GPU dependencies are loaded only on construction.
    from ._adapter import bind_chunk_attention

    config = FIXED_CONFIGS[name]
    proxy = SimpleNamespace(eager_attention_forward=native.eager_attention_forward)
    if hasattr(native, "mask_sparse_attention_indices"):
        proxy.mask_sparse_attention_indices = native.mask_sparse_attention_indices
    bind_chunk_attention(
        proxy,
        "tuned_full_gather" if config.group == 1 else "grouped_reuse",
        group_size=config.group,
        chunk_tokens=1024,
        profile=profile,
        backward_block_size=config.backward_block,
        backward_max_heads=config.backward_head_tile,
        pad_grouped_heads=config.pad_heads,
        elide_singleton_cat=True,
        elide_zero_padding=True,
    )
    return proxy.eager_attention_forward


def make_attention(native, configuration=SHAPE_POLICY, *, profile=False):
    """Build a callable without changing ``native.eager_attention_forward``.

    This is the auxiliary-disabled attention experiment, not the later FP64
    full-model diagnostic. Construction requires the frozen GPU dependencies.
    """
    if configuration != SHAPE_POLICY and configuration not in FIXED_CONFIGS:
        raise ValueError(f"Unknown TileReuse configuration: {configuration!r}")
    _require_gpu()
    if configuration != SHAPE_POLICY:
        return _fixed_attention(native, configuration, profile)

    functions = {name: _fixed_attention(native, name, profile) for name in FIXED_CONFIGS}
    counts = {}

    def attention(module, query, key, value, attention_mask, scaling, **kwargs):
        selected = select(query.shape[1])
        counts[selected] = counts.get(selected, 0) + 1
        return functions[selected](module, query, key, value, attention_mask, scaling, **kwargs)

    attention.choice_counts = counts
    return attention


@contextmanager
def attention_context(native, configuration=SHAPE_POLICY, *, profile=False):
    """Temporarily install TileReuse and restore the previous binding on exit.

    Use in a single serialized owner of the module. Nested contexts restore in
    stack order; concurrent module patches are not supported. Complete forward,
    backward and checkpoint recomputation inside the context.
    """
    original = native.eager_attention_forward
    attention = make_attention(native, configuration, profile=profile)
    native.eager_attention_forward = attention
    try:
        yield attention
    finally:
        native.eager_attention_forward = original

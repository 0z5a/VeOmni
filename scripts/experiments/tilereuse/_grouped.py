"""Group nearby queries and reuse their sparse KV union, including dKV math."""

import torch
import triton
import triton.language as tl

from . import _backward as backward
from . import _forward as forward


@triton.jit
def _segment_or(left_bits, left_head, right_bits, right_head):
    return tl.where(right_head, right_bits, left_bits | right_bits), left_head | right_head


@triton.jit
def group_plan_kernel(
    Ids,
    Union,
    Membership,
    Counts,
    K: tl.constexpr,
    G: tl.constexpr,
    WIDTH: tl.constexpr,
    INPUT_BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    offsets = tl.arange(0, INPUT_BLOCK)
    ids = tl.load(Ids + group * WIDTH + offsets, offsets < WIDTH, other=-1)
    active = (offsets < WIDTH) & (ids >= 0)
    # Eight low bits carry query membership. Fixed 16K traces fit the ID range.
    keys = tl.where(active, (ids << 8) | (1 << (offsets // K)), 2147483647)
    keys = tl.sort(keys, descending=False)
    row = keys >> 8
    bits = keys & 255
    live = keys != 2147483647
    previous = tl.gather(row, tl.maximum(offsets - 1, 0), axis=0)
    following = tl.gather(row, tl.minimum(offsets + 1, INPUT_BLOCK - 1), axis=0)
    head = live & ((offsets == 0) | (row != previous))
    tail = live & ((offsets == INPUT_BLOCK - 1) | (row != following))
    combined, _ = tl.associative_scan((bits, head), axis=0, combine_fn=_segment_or)
    compact = tl.cumsum(head.to(tl.int32), 0) - 1
    tl.store(Union + group * WIDTH + compact, row, mask=tail)
    tl.store(Membership + group * WIDTH + compact, combined, mask=tail)
    tl.store(Counts + group, tl.sum(head.to(tl.int32), 0))


def group_plan(indices, group_size):
    assert indices.is_contiguous() and indices.dtype == torch.int32
    batch, queries, width = indices.shape
    assert batch == 1 and queries % group_size == 0 and group_size in (2, 4, 8)
    groups = queries // group_size
    capacity = group_size * width
    assert capacity % 64 == 0
    input_block = triton.next_power_of_2(capacity)
    union = torch.full((1, groups, capacity), -1, dtype=torch.int32, device=indices.device)
    membership = torch.zeros_like(union)
    counts = torch.empty((1, groups), dtype=torch.int32, device=indices.device)
    group_plan_kernel[(groups,)](
        indices,
        union,
        membership,
        counts,
        width,
        group_size,
        capacity,
        input_block,
        num_warps=4,
    )
    return union, membership, counts


class GroupedSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, indices, group_size):
        assert q.is_contiguous() and kv.is_contiguous()
        assert kv.shape[1] < (1 << 23) - 1, "packed planner ID range"
        batch, queries, heads, dim = q.shape
        assert heads * group_size in (16, 32, 64, 128)
        union, membership, counts = group_plan(indices, group_size)
        grouped = q.reshape(batch, queries // group_size, heads * group_size, dim)
        sinks = sink.repeat(group_size).contiguous()
        kernel = forward.sparse_mqa_fwd(heads * group_size, dim, union.shape[-1], query_heads=heads)
        output, lse = kernel(grouped, kv, sinks, union, membership, counts)
        ctx.save_for_backward(grouped, kv, sinks, union, membership, counts, output, lse)
        ctx.original_shape = q.shape
        ctx.group_size = group_size
        ctx.query_heads = heads
        return output.reshape_as(q)

    @staticmethod
    def backward(ctx, grad):
        q, kv, sink, union, membership, counts, output, lse = ctx.saved_tensors
        batch, groups, heads, dim = q.shape
        delta = backward.preprocess(batch, groups, heads, dim)(output, grad.contiguous().reshape_as(q))
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        dsink = torch.zeros_like(sink)
        kernel = backward.bwd(
            batch,
            groups,
            kv.shape[1],
            heads,
            dim,
            union.shape[-1],
            query_heads=ctx.query_heads,
        )
        dq = kernel(
            q,
            kv,
            grad.contiguous().reshape_as(q),
            sink,
            union,
            membership,
            counts,
            lse,
            delta,
            dkv,
            dsink,
        )
        converted = backward.postprocess(batch, kv.shape[1], dim)(dkv)
        return (
            dq.reshape(ctx.original_shape),
            converted,
            dsink.reshape(ctx.group_size, ctx.query_heads).sum(0),
            None,
            None,
        )


def grouped_sparse_attention(q, kv, sink, indices, group_size=4):
    return GroupedSparseAttention.apply(q, kv, sink, indices, group_size)

"""Isolated copy ablation of the 1024 adapter; inputs stay immutable to backward.

The legacy path is preserved when both options are false. No global buffer pool,
communication change, planner change or kernel change is introduced.
"""

from contextlib import nullcontext


def bind_chunk_attention(
    native,
    mode,
    group_size=2,
    chunk_tokens=1024,
    profile=False,
    backward_block_size=32,
    backward_max_heads=32,
    pad_grouped_heads=True,
    elide_singleton_cat=False,
    elide_zero_padding=False,
):
    import torch

    from veomni.ops.kernels.deepseek_v4 import tilelang_sparse_mla_bwd as original_bwd
    from veomni.ops.kernels.deepseek_v4 import tilelang_sparse_mla_fwd as original_fwd

    from . import _backward as reused_bwd
    from . import _forward as reused_fwd
    from ._grouped import group_plan

    assert mode in (
        "original_full_gather",
        "chunk_full_gather",
        "grouped_reuse",
        "tuned_full_gather",
    )
    original = native.eager_attention_forward
    if mode == "original_full_gather":
        return original

    def marker(name):
        return torch.cuda.nvtx.range(name) if profile else nullcontext()

    def join(values):
        # Preserve the canonical contiguous layout on non-contiguous outputs.
        if elide_singleton_cat and len(values) == 1 and values[0].is_contiguous():
            return values[0]
        return torch.cat(values, dim=1)

    class ChunkAttention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, kv, sink, indices):
            batch, queries, heads, dim = q.shape
            assert batch == 1 and queries > 0 and chunk_tokens == 1024
            assert indices.shape[:2] == (batch, queries)
            input_queries = queries
            # The final partial chunk contributes its actual rows only. Dummy
            # rows have no selected KV or upstream gradient; complete1024-row
            # chunks keep the exact existing execution path.
            extra_queries = (-queries) % chunk_tokens
            if extra_queries:
                q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, extra_queries))
                indices = torch.nn.functional.pad(indices, (0, 0, 0, extra_queries), value=-1)
                queries += extra_queries
            assert heads >= 8 and heads & (heads - 1) == 0
            input_heads = heads
            reuse = mode == "grouped_reuse"
            # A TP8 shard of DS4.1 has eight heads. The MMA kernel needs 16;
            # dummy heads have no upstream gradient and are stripped below.
            if heads < 16 and (not reuse or pad_grouped_heads):
                q = torch.nn.functional.pad(q, (0, 0, 0, 16 - heads)).contiguous()
                sink = torch.nn.functional.pad(sink, (0, 16 - heads))
                heads = 16
            assert kv.shape[1] < (1 << 23) - 1
            actual_kv = kv.shape[1]
            # The original backward also pads KV to this specialization boundary.
            kv_padding = (-actual_kv) % 1024
            if kv_padding or not elide_zero_padding:
                kv = torch.nn.functional.pad(kv, (0, 0, 0, kv_padding))
            kv = kv.contiguous()
            id_padding = (-indices.shape[-1]) % 64
            if id_padding or not elide_zero_padding:
                indices = torch.nn.functional.pad(indices, (0, id_padding), value=-1)
            assert heads >= 16 or (reuse and heads * group_size >= 16)
            tuned = mode == "tuned_full_gather"
            g = group_size if reuse else 1
            assert queries % g == 0 and chunk_tokens % g == 0
            grouped = q.reshape(batch, queries // g, heads * g, dim)
            sinks = sink.repeat(g).contiguous()
            outs, lses, unions, memberships, counts = [], [], [], [], []
            # Batch the independent groups from all logical 1024-query chunks.
            # Divisibility prevents a group from crossing a chunk boundary.
            span = queries if reuse or tuned else chunk_tokens
            for begin in range(0, queries, span):
                selected = indices[:, begin : begin + span].contiguous()
                rows = grouped[:, begin // g : (begin + span) // g].contiguous()
                if reuse:
                    with marker("reuse_planner"):
                        union, membership, count = group_plan(selected, g)
                    with marker("reuse_attention_forward_kernel"):
                        out, lse = reused_fwd.sparse_mqa_fwd(heads * g, dim, union.shape[-1], query_heads=heads)(
                            rows, kv, sinks, union, membership, count
                        )
                    memberships.append(membership)
                    counts.append(count)
                else:
                    union = selected
                    out, lse = original_fwd.sparse_mqa_fwd(heads, dim, union.shape[-1])(rows, kv, sinks, union)
                outs.append(out)
                lses.append(lse)
                unions.append(union)
            output = join(outs)
            lse = join(lses)
            union = join(unions)
            membership = join(memberships) if reuse else indices.new_empty(0)
            count = join(counts) if reuse else indices.new_empty(batch, queries)
            ctx.save_for_backward(grouped, kv, sinks, union, membership, count, output, lse)
            ctx.q_shape = q.shape
            ctx.input_heads = input_heads
            ctx.input_queries = input_queries
            ctx.actual_kv = actual_kv
            ctx.group = g
            ctx.reuse = reuse
            ctx.tuned = tuned
            return output.reshape_as(q)[:, :input_queries, :input_heads]

        @staticmethod
        def backward(ctx, grad):
            q, kv, sink, union, membership, count, output, lse = ctx.saved_tensors
            batch, groups, heads, dim = q.shape
            if grad.shape[1] != ctx.q_shape[1]:
                grad = torch.nn.functional.pad(grad, (0, 0, 0, 0, 0, ctx.q_shape[1] - grad.shape[1]))
            if ctx.input_heads != ctx.q_shape[2]:
                grad = torch.nn.functional.pad(grad, (0, 0, 0, ctx.q_shape[2] - ctx.input_heads))
            do = grad.contiguous().reshape_as(q)
            with marker("attention_backward_preprocess"):
                delta = original_bwd.preprocess(batch, groups, heads, dim)(output, do)
            # All query chunks contribute before the single BF16 conversion.
            dkv = torch.zeros_like(kv, dtype=torch.float32)
            dsink = torch.zeros_like(sink)
            dqs = []
            size = groups if ctx.reuse or ctx.tuned else chunk_tokens // ctx.group
            for begin in range(0, groups, size):
                rows = q[:, begin : begin + size].contiguous()
                dout = do[:, begin : begin + size].contiguous()
                ids = union[:, begin : begin + size].contiguous()
                local_lse = lse[:, begin : begin + size].contiguous()
                local_delta = delta[:, begin : begin + size].contiguous()
                if ctx.reuse or ctx.tuned:
                    kernel = reused_bwd.bwd(
                        batch,
                        size,
                        kv.shape[1],
                        heads,
                        dim,
                        ids.shape[-1],
                        query_heads=ctx.q_shape[2],
                        block_size=backward_block_size,
                        has_membership=ctx.reuse,
                        max_block_heads=backward_max_heads,
                    )
                    with marker(
                        "reuse_attention_backward_kernel" if ctx.reuse else "tuned_full_gather_backward_kernel"
                    ):
                        dq = kernel(
                            rows,
                            kv,
                            dout,
                            sink,
                            ids,
                            membership[:, begin : begin + size].contiguous() if ctx.reuse else ids,
                            count[:, begin : begin + size].contiguous(),
                            local_lse,
                            local_delta,
                            dkv,
                            dsink,
                        )
                else:
                    kernel = original_bwd.bwd(batch, size, kv.shape[1], heads, dim, ids.shape[-1])
                    dq = kernel(rows, kv, dout, sink, ids, local_lse, local_delta, dkv, dsink)
                dqs.append(dq)
            with marker("attention_backward_postprocess"):
                converted = original_bwd.postprocess(batch, kv.shape[1], dim)(dkv)
            return (
                join(dqs).reshape(ctx.q_shape)[:, : ctx.input_queries, : ctx.input_heads],
                converted[:, : ctx.actual_kv],
                dsink.reshape(ctx.group, ctx.q_shape[2]).sum(0)[: ctx.input_heads],
                None,
            )

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        assert query.shape[0] == 1 and key.shape[1] == 1 and dropout == 0
        assert kwargs.get("indexer_target_width") is None, "auxiliary-disabled benchmark"
        indices = kwargs.get("sparse_topk_indices")
        assert indices is not None
        if attention_mask is not None:
            indices = native.mask_sparse_attention_indices(attention_mask, indices)
        rows = query.transpose(1, 2).contiguous()
        kv = key[:, 0].contiguous()
        sink = kwargs.get("s_aux", module.sinks).float().contiguous()
        assert scaling == rows.shape[-1] ** -0.5
        assert rows.dtype == kv.dtype == torch.bfloat16
        return ChunkAttention.apply(rows, kv, sink, indices.contiguous()), None

    native.eager_attention_forward = attention
    return original

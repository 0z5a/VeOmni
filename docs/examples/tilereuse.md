# TileReuse: experimental sparse attention training

**Backward-Tile-Aware KV Reuse for Sparse Attention Training**

TileReuse groups nearby queries, builds a sorted union of their selected KV
rows, and retains per-query membership bits. This permits KV reuse in forward
and shared dKV accumulation in backward while retaining each query's sparse
support. Full-KV communication remains unchanged.

This is a source-checkout experiment under `scripts/experiments/tilereuse`,
excluded from the installed package. It is not a registered training backend.
The packaged entry point has CPU integration coverage only. Earlier full-model
qualification failed, and qualified full-model timing windows were not run.
Do not infer numerical equivalence or training speedup from this extraction.

## Source and provenance

| File | Responsibility |
| --- | --- |
| `config.py` | Immutable configurations and frozen shape lookup |
| `veomni.py` | Callable construction, hardware guard and scoped binding |
| `_adapter.py` | Layout, padding, autograd state and FP32 dKV accumulator |
| `_grouped.py` | Triton union and membership planner |
| `_forward.py` | TileLang forward kernel |
| `_backward.py` | TileLang backward kernel |
| `provenance.json` | Source commit, SHA256 values and numerical AST fingerprints |

The four numerical files were extracted from research commit
`c929327ed8d71889c1f4bf75195b3e90133bcbb3`. Package-relative imports and repository
formatting are the only changes to those files; their numerical ASTs were
compared against the frozen sources. The manifest records the source and target
SHA256 values, the normalized AST fingerprint, the frozen dispatch specification,
and VeOmni base `8791a71f2ee2a732453b44392375e6a06aa00324`.

The source commit identifies the original research checkout; its full history,
raw traces and experiment harness are not included here. The provenance test
checks the extracted files against the recorded fingerprints; it does not
independently fetch the original checkout or reproduce historical measurements.
The TileLang files retain the Miles/ByteDance Apache-2.0 notices. Measurements
in inherited kernel comments are not new measurements of this entry point.

## Explicit use from a source checkout

The public API and configuration import without importing PyTorch, Triton,
TileLang or VeOmni. Constructing a callable requires the compatible VeOmni
runtime, PyTorch, Triton, TileLang and an NVIDIA SM90+ GPU. This change does not
establish a new runtime-version compatibility matrix.

```python
from scripts.experiments.tilereuse import attention_context, make_attention, select
from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as native

assert select(16) == "g4-b32-m64-both"

# run_existing_attention_training_step is supplied by the caller's experiment.
# Use the eager attention route and disable the indexer auxiliary loss.
# Keep the complete forward/backward and checkpoint recomputation in this scope.
with attention_context(native) as attention:
    loss = run_existing_attention_training_step()
    loss.backward()
    choices = dict(attention.choice_counts)

# The previous native.eager_attention_forward binding is restored on exit.
# A fixed callable can also be constructed without installing it:
fixed = make_attention(native, "g4-b32-m64-both")
```

The caller owns installation and execution. VeOmni does not automatically call
these helpers or select this experiment from a configuration setting:

- [select](../../scripts/experiments/tilereuse/config.py#L26) accepts the actual
  per-rank head count and returns a frozen configuration name.
- [make_attention](../../scripts/experiments/tilereuse/veomni.py#L38) accepts a
  native module, optional configuration name and keyword-only `profile` flag.
  It constructs a native-signature callable without replacing the module's
  binding. Invalid names fail before GPU imports. Shape-policy callables expose
  `choice_counts`; fixed callables do not.
- [attention_context](../../scripts/experiments/tilereuse/veomni.py#L63) builds
  the callable before installation, yields it, and restores the previous
  binding on normal or exceptional exit. Nested contexts restore in stack order.
- The existing [DeepSeek-V4 attention selection](../../veomni/models/transformers/deepseek_v4/generated/patched_modeling_deepseek_v4_gpu.py#L1994)
  obtains `eager_attention_forward` only when the eager route is selected.
  Other registered attention implementations bypass this binding.

One attention call reads the current query shape, selects the fixed kernel,
applies native masking, pads the inputs and constructs group membership. The
forward saves its inputs and outputs for backward. Backward accumulates dKV in
FP32 and performs the final conversion once. The return value is the native
pair `(output, None)`, with output layout `[1, Q, H, D]`.

The module binding requires a single serialized owner. It affects every model
using that module in the process during the context; concurrent model calls or
patches are unsupported. Saved inputs must remain immutable until backward.
Finish checkpoint recomputation before exiting the context.

## Frozen configurations

The shape policy makes a lookup on every attention call; there is no online
fitting. Query grouping, KV tile size, backward head tile and logical query
chunk length are distinct quantities.

| Per-rank heads H | Query group g | KV tile B | Backward head tile M |
| --- | ---: | ---: | ---: |
| 8 | 8 | 32 | 64 |
| 16 | 4 | 32 | 64 |
| 32 | 2 | 32 | 64 |
| Other | 1 | 64 | 32 |

The fallback is `tuned-both`; it does not qualify unseen shapes. Fixed grouped
choices are `g2-b32-m64-both`, `g4-b32-m64-both` and `g8-b32-m64-both`. Both the
tuned and grouped paths elide singleton concatenation and zero-width padding;
non-contiguous outputs retain their canonical copy path. Logical query chunks
remain 1024 rows, while the kernel launch batches the rank's groups.

## Input and evidence boundaries

- Batch size 1, BF16 Q/KV, a single shared KV head, FP32 sink computation,
  zero dropout, auxiliary-disabled sparse MLA and scaling `D**-0.5`.
- Query layout `[1, H, Q, D]`, key layout `[1, 1, S, D]`, and int32 selected
  IDs `[1, Q, K]`. The key tensor also supplies values, as in the native MLA
  path. A separate value tensor is not supported by this experiment.
- Valid IDs must be unique within each query and belong to its KV domain;
  negative IDs denote padding. Duplicate IDs are folded by the planner, so
  multiset attention is unsupported. KV length must be below `(1 << 23) - 1`.
- The adapter retains its original tail-query, head, KV and selected-ID
  padding. Callers must satisfy the native selection/masking contract and
  keep assertions enabled; the extraction is not a general input validator.
- Historical attention experiments used H100, two native ID graphs and
  synthetic projection inputs/targets. They do not establish universal
  optimality, new-corpus generalization, firstness or full-model speedup.
- The later native-forward FP64 full-model diagnostic is not part of this
  entry point. Previous negative results and unfulfilled numerical gates
  remain valid; CPU packaging tests do not supersede them.

## CPU validation

From the repository root, with pytest installed:

```sh
python -m pytest tests/ops/test_tilereuse.py -q
make quality
```

The tests cover source fingerprints, dependency-free public import, frozen
configuration parity, shape routing, argument and mask-helper forwarding,
fixed-path construction, nested restoration, construction failure and the
hardware-guard boundary. GPU execution and hardware detection are mocked in
binding tests. The GPU unit workflow already discovers `tests/ops` wholesale;
this CPU test adds no accelerator qualification. A future GPU run must qualify
the extracted entry point against the original numerical and full-step gates.

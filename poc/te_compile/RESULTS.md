# Can torch.compile see through Transformer Engine? (MFSDP v2 compiler-route PoC, stage 2)

Date: 2026-10-28. Single GB200 node, NeMo 26.08 container, TE 2.20.2, branch `te-compile-poc`
(based on `fab7790cb`). Probe: `te_compile_probe.py`, driver: `run_probe.sh`.

Modes: `explain` = `torch._dynamo.explain` on forward; `aot_eager` / `inductor` =
`torch.compile(fullgraph=True)` forward + backward, compared with eager; `partial` = same with
`fullgraph=False`. Precisions: `bf16`; `mxfp8` = bf16 weights + `te.autocast(MXFP8BlockScaling)`;
`mxfp8_param` = `te.quantized_model_init` MXFP8 primary weights (the v2 setup).

## 1. The container's PyTorch is too old

With the container's torch `2.13.0a0+8145d630e8.nv26.06`, TE 2.20.2 disables its compile support:

    Transformer Engine torch.compile support is disabled: could not register OpaqueValueBundle as an
    opaque type (cannot import name 'is_opaque_constant_type' from 'torch._library.opaque_object')

Every TE module then falls back to eager (`fullgraph=True` fails everywhere). Results: jobs 8428542,
8429862. All numbers below use PyTorch nightly `2.16.0.dev20261028+cu130` + TE 2.20.2 built against
it (`setup_nightly.sh`, jobs 8431546, 8431937).

## 2. Results with torch nightly

| Case | fullgraph=True (aot_eager and inductor) | Notes |
|---|---|---|
| `te.Linear` bf16 / mxfp8 / mxfp8_param | **pass**, bit-exact fwd + bwd | one `transformer_engine_compile::linear(_base)` op fwd, `linear_backward(_base)` bwd |
| SimpleFSDP-style sharded `te.Linear` (bf16 shard, all-gather inside the graph) bf16 / mxfp8 | **pass**, bit-exact | fwd graph `all_gather_into_tensor -> wait -> te::linear`; bwd graph `te::linear_backward -> reduce_scatter_tensor -> wait` |
| v2-style sharded MXFP8 planes, unshard Function returns an `MXFP8Tensor` | fail | the tracer gives the Function's backward an `MXFP8Tensor` fake gradient; `reduce_scatter` on it touches the data pointer. `__coerce_tangent_metadata__` does not help |
| `te.LayerNormLinear`, `te.LayerNormMLP`, `te.GroupedLinear` | fail | modules are `@no_torch_dynamo`; with `fullgraph=False` they run eager (graph break) |
| `te.DotProductAttention` (cuDNN fused) | fail | `FusedAttention.forward` is `torch.compiler.disable`d |
| `te.TransformerLayer` | fail | 3 to 7 breaks (LayerNormLinear, FusedAttention, ...) |
| MCore `TransformerLayer`, TE spec, bf16 | fail; `partial` 5 graphs / 4 breaks | `TELinear` (proj, fc2) compiled; `TELayerNormColumnParallelLinear` and attention break |
| MCore TE spec, mxfp8 | fail | additionally `te.Linear` falls back: **FP8 weight caching (`is_first_microbatch`)**, which MCore always passes |
| MCore TE spec, mxfp8_param | fail | plus `tex.dequantize` pybind call traced from MCore code |
| MCore local spec, bf16 | fail; 11 graphs / 10 breaks | MCore's own code: `get_rng_state` (RNG tracker), `warnings.warn` |

`partial` (fullgraph=False) runs are numerically equal to eager in every case (TE pieces bit-exact;
MCore layer ~1e-3 rel. err in bf16 from attention, same with or without compile).

## 3. What this means for the compiler route

1. **Needs a newer PyTorch than the current NGC containers ship** (`torch._library.opaque_object`
   opaque-constant support). Nothing else in TE 2.20.2 is a blocker for `te.Linear`.
2. **SimpleFSDP mechanics work with TE today** for weights that are gathered in high precision and
   quantized inside TE's op (bf16 shard -> all-gather -> `te::linear`): one graph with the collective
   and the GEMM, reduce-scatter from autograd, bit-exact.
3. **v2's "gather pre-quantized MXFP8 planes" does not map onto a differentiable unshard op yet**:
   an autograd.Function cannot output a QuantizedTensor whose gradient is a plain tensor. Options:
   (a) a custom op / TE API that takes the gathered planes as inputs and returns the wgrad w.r.t. the
   high-precision master (TE's own `linear_base` already takes quantized-weight planes as op inputs,
   so this is a small TE-side extension); (b) keep the MXFP8 unshard eager and compile only the
   compute (the `te.Linear mxfp8_param` case passes, the weight is a graph input).
4. **Coverage gaps are in TE modules, not in FSDP**: LayerNormLinear / LayerNormMLP / GroupedLinear
   / fused attention are not yet custom ops. An MCore layer gets at most per-module graphs today.
   Short-term options: an MCore spec with separate norm + `TELinear` (no LN fusion), and
   `is_first_microbatch=None` under compile.
5. MCore's own code also breaks graphs (RNG tracker, warnings) independent of TE.

## 4. Whole-graph capture (make_fx joint fwd + bwd) and FSDP passes

`joint_capture_probe.py` traces one training step (forward, loss, `torch.autograd.grad`) of a
4-block SimpleFSDP-style model (bf16 `Shard(0)` weights, unshard autograd.Function, te.Linear)
into one FX graph with make_fx on fake tensors (the torchtitan graph_trainer approach), 4 GB200
ranks, hidden 2048, ffn 8192, 8192 tokens. Jobs 8434334 (te.Linear block) and 8434579
(te.LayerNormLinear block).

* **TE's custom-op path must be switched on explicitly.** It only activates when
  `torch.compiler.is_compiling()`, which make_fx does not set; without
  `torch.compiler._compile_session_context()` TE calls pybind kernels on fake tensors
  ("Cannot access data pointer"). With it, the whole step is one graph.
* **te.LayerNormLinear without TE changes**: `te_opaque.py` swaps `_LayerNormLinear` for a proxy
  that, while tracing, calls a generic `mfsdp_te::opaque_fwd` / `opaque_bwd` custom-op pair running
  TE's own forward/backward with a stub ctx. The op takes `Tensor[]` (not `Tensor?[]`, which autograd
  treats as non-differentiable). Saved tensors stay outside the graph (memory passes cannot see them).
* All graphs and pass pipelines are **bit-exact** vs eager (loss and every gradient), bf16 and mxfp8.

Passes (`fsdp_graph_passes.py`), each replacing a v2 hook mechanism:

| pass | replaces in v2 | effect in the graph |
|---|---|---|
| `dedup_unshard` | per-phase `_unshard_events` reuse | te.Linear reads `weight` twice per forward -> 16 all-gathers for 8 weights; dedup -> 8 |
| `prefetch_forward` | forward prefetch over `ExecutionOrder` | block k+1's all-gather issued before block k's compute |
| `bucket_by_block` | one DBuffer per FsdpModule | 1 AG bucket + 1 RS bucket per block |
| `sink_rs_waits` | RS stream + final autograd callback | RS of block k overlaps backward of block k-1; waits at graph end |

Step time (FX graph executed op by op, no inductor, no CUDA graphs, so CPU overhead is included;
mean of 10 steps) and peak allocated memory:

| block | prec | eager | graph, no pass | graph + dedup+prefetch+blockbucket+sinkrs |
|---|---|---|---|---|
| te.Linear | bf16 | 7.81 ms / 2.15 GB | 8.10 ms / 1.98 GB | 7.24 ms / 1.98 GB (dedup+prefetch only: 6.96 ms) |
| te.Linear | mxfp8 | 9.36 ms / 2.10 GB | 11.24 ms / 1.97 GB | 12.22 ms / 1.97 GB |
| te.LayerNormLinear (opaque) | bf16 | 7.64 ms / 2.15 GB | 7.69 ms / 1.98 GB | 6.97 ms / 1.98 GB |
| te.LayerNormLinear (opaque) | mxfp8 | 8.75 ms / 2.10 GB | 9.68 ms / 1.97 GB | 8.38 ms / 1.97 GB |

torch's own `bucket_all_gather`/`bucket_reduce_scatter` need a `stable_topological_sort` afterwards
(inductor does it in post_grad); they merge everything into one bucket (6.69 ms, +0.07 GB).
The mxfp8 te.Linear graph is slower than eager because of the per-call Python glue of TE's custom op
(output-plan parsing); timings need CUDA graphs / inductor to be meaningful.

## 5. Real MCore TransformerLayers (TE spec) as one graph

`mcore_capture_probe.py`: 4 MCore `TransformerLayer`s from `get_gpt_layer_with_transformer_engine_spec()`
(TP=1, hidden 2048, ffn 5632, 16 heads, seq 2048 x batch 2, RMSNorm, SwiGLU, no bias), every 2-D
weight (16) sharded Shard(0) over 4 ranks behind a property calling the unshard Function
(`apply_simple_fsdp`), one training step traced with make_fx. Jobs 8437077 (default) and 8437076
(`deterministic_mode` + `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`).

What it took (all outside MCore/TE source, in the probe):

| obstacle | where | PoC workaround | real fix |
|---|---|---|---|
| `LayerNormLinear` (linear_qkv, linear_fc1) not a torch op | TE | `te_opaque` opaque op pair | TE custom op |
| `FusedAttnFunc` not a torch op; its args dataclass carries tensors | TE | `te_opaque` with dataclass tensor fields as op inputs | TE custom op |
| TE asserts q/k/v may be views of a packed buffer under compile | MCore/TE | `.contiguous()` q/k/v while tracing | MCore passes `qkv_layer=` to TE |
| backend selection sees `type(q) is FakeTensor`, disables fused/flash, silently uses unfused attention (+1.1 GB, different numerics) | TE bug (make_fx only; Dynamo reports torch.Tensor) | treat FakeTensor as torch.Tensor in `get_attention_backend` | TE fix |
| MCore fused `SwiGLUFunction` calls a non-op kernel | MCore | `bias_activation_fusion=False` (plain silu*mul) | make it an op or let inductor fuse |
| FP8 `is_first_microbatch` weight caching rejected by te.Linear's op | MCore/TE | `disable_parameter_transpose_cache=True` | caching as an explicit graph buffer |
| MCore `jit_fuser` (torch.compile'd helpers) | MCore | `error_on_nested_fx_trace=False` (as graph_trainer) | -- |

Result: one graph per step with 24 all-gathers, 16 reduce-scatters, 8 `te::linear(_backward)`,
12 opaque TE fwd + 12 bwd (qkv, fc1, attention per layer). Loss bit-exact; gradients differ from
eager exactly as much as eager differs from itself (cuDNN attention bwd is non-deterministic):

| prec | eager vs eager rerun (max grad rel) | graph vs eager | deterministic mode: graph vs eager |
|---|---|---|---|
| bf16 | 5.0e-3 | 5.0e-3 | **0 (bit-exact)** |
| mxfp8 | 2.4e-2 | 2.2e-2 .. 2.5e-2 | **0 (bit-exact)** |

Step time / peak memory (FX graph run op by op, no CUDA graphs; job 8437077):

| prec | eager | graph | graph + dedup+prefetch+blockbucket+sinkrs |
|---|---|---|---|
| bf16 | 19.17 ms / 2.30 GB | 16.75 ms / 2.21 GB | **13.43 ms (-30%)** / 2.21 GB |
| mxfp8 | 22.46 ms / 2.67 GB | 21.84 ms / 2.59 GB | **17.74 ms (-21%)** / 2.59 GB |

Eager here is the SimpleFSDP-style model run eagerly (gather on use, no prefetch), not MFSDP v2.

## 6. MCore MoE TransformerLayers (TE spec) as one graph

`mcore_capture_probe.py --moe-experts 8`: 4 MoE layers, 8 experts, top-2, dropless, EP=1/TP=1,
`moe_ffn_hidden_size=1408`, TE grouped GEMM, AllToAll dispatcher, aux-loss balancing (coeff 1e-2),
same attention part as section 5. 76 of 84 parameters sharded (2-D weights incl. router and every
expert's fc1/fc2). Deterministic mode.

Additional obstacles beyond section 5:

| obstacle | where | PoC workaround | real fix |
|---|---|---|---|
| `GroupedLinear` (experts) not a torch op | TE | `te_opaque.patch_grouped_linear`; `m_splits` is a tensor op input, output shape = total tokens, so dropless stays static-shaped | TE custom op |
| `TEGroupedMLP` calls `tokens_per_expert.tolist()` | MCore | **source edit** `experts.py`: skip `tolist()` while compiling (TE accepts tensor m_splits) | same edit upstream |
| MoE spec's pre-MLP norm is TE op-fuser `TEFusedResidualRMSNorm` | TE | spec swap to MCore `WrappedTorchNorm` | wrap te.ops fuser |
| `RouterGatingLinearFunction` calls TE pybind `general_gemm` | MCore | opaque op proxy (`patch_mcore_router_gemm`) | make it an op |
| `MoEMetricsTracker.record` mutates a global real tensor in place | MCore | skipped while tracing (logging only) | buffers as explicit graph state |
| AllToAll dispatcher D2H copy on a side stream + `record_stream` | MCore | skipped while tracing (EP=1 only) | EP>1: symbolic or capacity splits |
| `MoEAuxLossAutoScaler.main_loss_backward_scale` is global state | MCore | passed as an explicit graph input (`inputs["aux_scale"]`) | same, graph state |
| real module buffers (dispatcher sort indices) used by ops | -- | tracer uses its own `FakeTensorMode(allow_non_fake_inputs=True, ShapeEnv())` like graph_trainer | -- |
| TE/MCore backward returning an input grad unchanged (custom-op alias rule) | -- | clone aliased outputs in the opaque ops | -- |

Results (FX graph run op by op, no CUDA graphs; loss / max grad rel. error vs eager):

| run | permute | nodes | host syncs | numerics | eager | graph | graph + 4 passes |
|---|---|---|---|---|---|---|---|
| 8438493 | unfused | 2239 | 96 | bit-exact | 67.95 ms / 2.86 GB | 54.37 ms / 2.84 GB | 36.56 ms / 2.85 GB |
| 8438886 | `moe_permute_fusion` | 1539 | **0** | bit-exact | 64.75 ms / 2.86 GB | 59.85 ms / 2.84 GB | 60.93 ms / 2.85 GB |

* 84 all-gathers / 76 reduce-scatters per step (before passes), 4+4 after block bucketing.
* Host syncs = `aten._local_scalar_dense` nodes (counted from the graph dump for 8438493). The
  unfused AllToAll path re-sorts tokens by expert with `split` on host-side sizes (identity at
  EP=1, still executed): 8 unbacked sizes per sort, fwd and bwd. TE's fused permute/chunk-sort are
  torch.library ops taking split sizes as tensors -> no host syncs, CUDA-graph-capturable.
* Open question: with the fused permute the four passes give nothing (60.9 vs 59.9 ms) while they
  give -33% on the unfused graph; not investigated yet.
* Dense regression after switching the tracer to its own FakeTensorMode (8438889): bf16 and mxfp8
  still bit-exact (bf16 graph+passes 14.44 ms vs eager 18.62; mxfp8 17.47 vs 21.18).

**MXFP8 MoE does not trace yet** (8438689 fused, 8438691 unfused). MXFP8 grouped GEMM needs every
expert segment padded to a multiple of 32 tokens; with dropless routing the padded total is
data-dependent (between T*k and T*k + E*31). Fused: TE's `permute_mask_map_fwd` gets
`num_out_tokens` as a tensor (padded routing-map sum) where it needs an int; unfused: "Could not
extract specialized integer from data-dependent expression u0". Whole-graph MXFP8 MoE needs a
static padding budget (pad to the upper bound, with masked rows) or symbolic sizes through TE.

## 7. Parameter-group FSDP: MFSDP v2's flat-buffer layout as graph ops

`fsdp_group_ops.py`, `mcore_capture_probe.py --fsdp group`. Replaces the per-parameter `_Unshard`
of sections 4-6 with v2's unit of communication:

* every parameter of a TransformerLayer goes into one flat buffer **per dtype** (as v2's
  `FsdpParameterGroup`; MoE routers keep an fp32 group), laid out by v2's own
  `layout_builder.build_row_atomic_layout` (RowAtomic, `ShardLayout.PADDED` or `UNPADDED`); each
  rank holds its segment as a 1-D parameter `fsdp_flat_shard_<dtype>`;
* the layer's forward calls `UnshardGroup` **once** per group (an explicit data dependency, not a
  hook) and publishes per-parameter views; submodule attributes (`weight`, ...) are properties
  returning those views. Backward packs the per-parameter grads into the flat layout (zeros for
  gaps/padding) and reduce-scatters them: one autograd node and one RS per group per layer;
* collectives: equal segments use the functional collectives (launch and wait are separate,
  schedulable nodes). Unequal (UNPADDED) segments either `opaque` -- custom ops
  `mfsdp::all_gather_uneven` / `mfsdp::reduce_scatter_uneven` running v2's list collectives
  exactly -- or `pad` -- pad each segment to the largest, functional collectives, drop the padding.

Results (4 layers, 4 GB200 ranks, deterministic mode; FX graph run op by op, no CUDA graphs;
times are eager / graph / graph + dedup+prefetch+blockbucket+sinkrs). Every graph and every pass
pipeline is **bit-exact** vs eager (loss and all gradients), 0 host syncs.

| model | layout | uneven | AG/RS per step | padding / layer | bf16 (ms) | mxfp8 (ms) | job |
|---|---|---|---|---|---|---|---|
| dense | PADDED | -- | 4 / 4 | 69,632 el. | 12.07 / 12.91 / 12.04 | 14.85 / 17.35 / 15.94 | 8444928 |
| dense | UNPADDED | pad | 4 / 4 | 0 | 13.01 / 14.59 / 13.35 | 15.27 / 18.41 / 17.04 | 8444928 |
| dense | UNPADDED | opaque | 4u / 4u | 0 | 10.98 / 12.50 / 12.19 | 13.55 / 17.25 / 15.86 | 8444928 |
| MoE 8e | PADDED | -- | 8 / 8 | 55,296 el. (bf16 group) | 31.84 / 32.89 / 33.52 | -- | 8445358 |
| MoE 8e | UNPADDED | pad | 8 / 8 | 0 | 33.57 / 39.11 / 32.20 | -- | 8445358 |

(dense layer group: 6 params, 51.4 M elements; MoE layer: bf16 group of 20 params, 86.0 M elements,
plus a 1-param fp32 router group. UNPADDED rank segments are within 2,048 elements of each other.
Job 8444930 is the first MoE attempt, which failed on the one-dtype-per-layer assumption; 8445094
is the rerun whose pass pipeline failed in block bucketing, fixed below.)

Findings:
* v2's DBuffer semantics map onto the graph unchanged: v2's layout builder, uneven segments and
  per-dtype groups all trace, and the reduce-scatter of one packed flat gradient per group is
  bit-exact. Compared with per-parameter sharding (section 6, MoE: 84 AG / 76 RS, eager
  64.8 ms) the group unit cuts collectives 10x and eager time in half.
* `opaque` uneven collectives are correct but complete inside their node, so passes can only
  change when they are issued: prefetch moved them, nothing overlapped (no RS wait to sink).
  To be schedulable, an uneven collective needs a launch/wait split like the functional
  collectives (or the `pad` emulation).
* `pad` keeps UNPADDED layouts schedulable at the cost of (max - own) extra elements per rank
  (<= 2,048 here) plus pad/compaction copies.
* The block-bucketing pass had to key buckets by dtype (the inductor merge helpers require one
  dtype per bucket); with one group per (layer, dtype) there is nothing left for it to merge.
* Timings are FX-interpreted and mostly noise-level at this size; they say nothing about v2 yet.

## 8. Schedulable uneven collectives (launch / wait split)

`--uneven-impl async` (`fsdp_group_ops.py`): `mfsdp::all_gather_uneven_start` /
`mfsdp::reduce_scatter_uneven_start` run v2's list collectives with `async_op=True` and register the
Work on their output with `torch._C._distributed_c10d._register_work` (the pattern PyTorch's own
symmetric-memory ops use); the graph then waits with the standard
`_c10d_functional.wait_tensor`. Launch and wait are separate nodes, so the existing prefetch and
RS-wait-sinking passes apply unchanged (they now also recognise `mfsdp.*` launches).

Correctness: every async graph and pass pipeline is bit-exact vs eager (dense bf16 + mxfp8, MoE
bf16; jobs 8445970, 8445975, 8446374, 8446379).

Overlap (`--overlap`: NCCL vs compute kernels on the rank-0 GPU timeline, per step, graph +
dedup+prefetch+blockbucket+sinkrs), dense bf16, UNPADDED, job 8445970:

| uneven impl | NCCL time | overlapped with compute | exposed |
|---|---|---|---|
| opaque | 12.44 ms | 0.00 ms | 12.44 ms |
| pad | 8.89 ms | 3.62 ms | 5.27 ms |
| async | 7.10 ms | 2.65 ms | 4.45 ms |

Without the passes all three expose 100% of their communication; `opaque` stays at 0 overlap with
them, `async` and `pad` overlap once prefetch / RS-wait sinking move the waits.

**Caveat -- these runs are host-bound, so overlap is not yet quantifiable.** Kernel-only analysis
of the saved traces (`trace_overlap.py`, jobs 8446374 / 8446379; traces under `results/*/graphs/`,
not committed) shows GPU busy time of 6-21 ms against a per-step span of 24-63 ms: the GPU idles
while Python dispatches the FX graph node by node (plus TE custom-op glue and profiler overhead).
NCCL kernel durations then mostly measure peers' host skew (the same mxfp8 `pad` graph shows
1.3 ms without and 15.0 ms with the passes), so MXFP8 and MoE overlap numbers vary run to run and
the dense bf16 table above is only qualitative. Quantifying overlap needs the graph replayed
without host overhead (CUDA graphs / inductor), the next step.

(The in-probe statistics of 8446374 / 8446379 are wrong: with CPU profiling enabled, Kineto
mirrors user annotations onto the GPU timeline and `prof.events()` counted them as kernels.
`overlap_stats` now parses the chrome trace and counts only `cat == "kernel"` events.)

## Environment notes
* slurm-broker jobs need explicit Lustre mounts (`mounts=[/lustre/fs1/...:..., /lustre/fsw/...:...]`).
* Nightly overlay: `/lustre/fsw/portfolios/coreai/users/xiaodaz/envs/torch216-nightly-te2.20.2`,
  used via `PYTHONPATH` in front of `/opt/venv`. uv refuses to replace the container's torch; pip
  downloads on compute nodes stalled, so wheels come from `.wheelhouse` (downloaded on the login node).
  `nightly_site/sitecustomize.py` hides the container's flash-attn and `fla` builds, which are ABI-
  or Triton-incompatible with the nightly.

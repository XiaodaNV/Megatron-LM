"""Whole-graph capture of real MCore TransformerLayers (TE spec) with SimpleFSDP-style sharding.

Builds `--layers` MCore TransformerLayers from get_gpt_layer_with_transformer_engine_spec()
(TP=1), shards every 2-D weight Shard(0) over all ranks behind a property that calls the
unshard autograd.Function (all-gather fwd / reduce-scatter bwd), and traces one training step
(fwd + loss + bwd) into one FX graph with make_fx -- see joint_capture_probe.py for the method.

TE pieces that are not torch.library ops yet go through te_opaque.py (LayerNormLinear for
linear_qkv / linear_fc1, FusedAttnFunc for core_attention); TERowParallelLinear (te.Linear)
uses TE's own custom op. 1-D parameters (layernorm weights) stay replicated (their gradients
are not all-reduced: the PoC compares eager vs graph of the same computation).

Run: torchrun --nproc-per-node 4 mcore_capture_probe.py --out r.jsonl
"""

import argparse
import contextlib
import json
import os
import time
import traceback

import torch
import torch.distributed as dist
from torch.fx.traceback import annotate

import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

import te_opaque
from joint_capture_probe import (
    MODULE_FQN,
    PARAM_FQN,
    _Unshard,
    apply_pass,
    bench,
    compare,
    rank0,
    summarize,
    trace,
)


def patch_mcore_attention_for_tracing():
    """Under torch.compile TE cannot detect whether q/k/v are views into one packed buffer
    (it reads data pointers) and asserts. MCore splits q/k/v out of linear_qkv's output as
    views, so make them contiguous while tracing. A real fix passes the packed buffer to TE
    via DotProductAttention(qkv_layer=..., qkv_interleave_dim=...)."""
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    orig = TEDotProductAttention.forward
    if getattr(orig, "_poc_patched", False):
        return

    def forward(self, query, key, value, *args, **kwargs):
        if torch.compiler.is_compiling():
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        return orig(self, query, key, value, *args, **kwargs)

    forward._poc_patched = True
    TEDotProductAttention.forward = forward


def patch_mcore_router_gemm():
    """MCore's RouterGatingLinearFunction calls TE's pybind general_gemm directly; route it
    through the same opaque op pair as the TE modules."""
    import megatron.core.transformer.moe.moe_utils as moe_utils

    if isinstance(moe_utils.RouterGatingLinearFunction, te_opaque.OpaqueFunction):
        return

    def out_meta(inp, weight, bias, router_dtype):
        return [(tuple(inp.shape[:-1]) + (weight.shape[0],), router_dtype, inp.device)]

    moe_utils.RouterGatingLinearFunction = te_opaque.OpaqueFunction(
        moe_utils.RouterGatingLinearFunction, out_meta, returns_tuple=False)


def patch_mcore_moe_metrics_for_tracing():
    """MoEMetricsTracker.record accumulates aux-loss values in place into a global real tensor
    (logging only), which a traced graph cannot mutate. Skip it while tracing; the aux loss
    itself still enters backward. A real fix threads the tracker buffers through the graph as
    explicit state (graph_trainer's `graph_state`)."""
    from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker

    orig = MoEMetricsTracker.record
    if getattr(orig, "_poc_patched", False):
        return

    def record(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return None
        return orig(self, *args, **kwargs)

    record._poc_patched = True
    MoEMetricsTracker.record = record


def patch_mcore_alltoall_dtoh_for_tracing():
    """The AllToAll dispatcher copies tokens_per_expert (and EP split sizes) to the host on a
    side stream with record_stream; make_fx cannot take torch.Stream arguments. With EP=1 the
    copy only serves the expert GEMM splits, which stay a device tensor under capture (see
    experts.py), so skip it while tracing. EP>1 needs host split sizes for the all-to-all, which
    are data-dependent: that requires symbolic splits or capacity padding (not covered here)."""
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    orig = MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize
    if getattr(orig, "_poc_patched", False):
        return

    def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
        if torch.compiler.is_compiling():
            assert self.ep_size == 1 and self.tp_size == 1, "PoC: EP/TP=1 only"
            return tokens_per_expert
        return orig(self, point, tokens_per_expert)

    _maybe_dtoh_and_synchronize._poc_patched = True
    MoEAlltoAllTokenDispatcher._maybe_dtoh_and_synchronize = _maybe_dtoh_and_synchronize


def apply_simple_fsdp(model):
    """Replace each 2-D parameter `name` of every submodule by `name_shard` (Shard(0) over
    WORLD) and make `name` a property returning the unsharded weight."""
    ws, r = dist.get_world_size(), dist.get_rank()
    sharded = []
    for mod_name, mod in list(model.named_modules()):
        names = [n for n, p in mod._parameters.items()
                 if p is not None and p.dim() >= 2 and p.shape[0] % ws == 0]
        if not names:
            continue
        props = {}
        for n in names:
            p = mod._parameters.pop(n)
            mod.register_parameter(n + "_shard",
                                   torch.nn.Parameter(p.detach().chunk(ws, 0)[r].clone()))
            fqn = f"{mod_name}.{n}"
            sharded.append(fqn)

            def prop(self, n=n, fqn=fqn):
                with annotate({PARAM_FQN: fqn}):
                    return _Unshard.apply(self._parameters[n + "_shard"], fqn)

            props[n] = property(prop)
        mod.__class__ = type("FSDP" + type(mod).__name__, (type(mod),), props)
    return sharded


class LayerStack(torch.nn.Module):
    def __init__(self, config, spec, n):
        super().__init__()
        from megatron.core.transformer.transformer_layer import TransformerLayer

        self.layers = torch.nn.ModuleList(
            TransformerLayer(config, spec.submodules, layer_number=i + 1) for i in range(n))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            with annotate({MODULE_FQN: f"layers.{i}"}):
                out = layer(hidden_states=x, attention_mask=None)
                x = out[0] if isinstance(out, tuple) else out
        return x


def build(args, prec):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.transformer_config import TransformerConfig

    fp8 = {} if prec == "bf16" else dict(fp8="e4m3", fp8_recipe="mxfp8",
                                          # no FP8 weight caching: TE's te.Linear op path
                                          # rejects is_first_microbatch
                                          disable_parameter_transpose_cache=True)
    moe = {}
    if args.moe_experts:
        # Dropless top-k MoE, EP=1, TE grouped GEMM, AllToAll dispatcher, aux-loss balancing.
        moe = dict(num_moe_experts=args.moe_experts, moe_router_topk=args.moe_topk,
                   moe_ffn_hidden_size=args.moe_ffn, moe_grouped_gemm=True,
                   moe_token_dispatcher_type="alltoall", moe_router_load_balancing_type="aux_loss",
                   moe_aux_loss_coeff=1e-2,
                   # TE fused permute / chunk-sort are torch.library ops taking split sizes as
                   # tensors; the unfused path splits by host values (data-dependent syncs).
                   moe_permute_fusion=args.moe_permute_fusion,
                   # FP8/MXFP8 grouped GEMM needs 32-aligned expert segments: pad in the router
                   # (routing map) instead of the host-side Fp8Padding in the experts.
                   moe_router_padding_for_quantization=(prec != "bf16"))
    cfg = TransformerConfig(
        num_layers=args.layers, hidden_size=args.hidden, ffn_hidden_size=args.ffn,
        num_attention_heads=args.heads, normalization="RMSNorm", gated_linear_unit=True,
        # bias_activation_fusion=False: MCore's fused SwiGLUFunction calls a kernel that is not
        # a torch op (fails on fake tensors); plain silu*mul traces and inductor can fuse it.
        activation_func=torch.nn.functional.silu, bias_activation_fusion=False,
        add_bias_linear=False, hidden_dropout=0.0, attention_dropout=0.0, bf16=True,
        params_dtype=torch.bfloat16, deterministic_mode=args.deterministic, **fp8, **moe)
    torch.manual_seed(0)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(123)
    if args.moe_experts:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm

        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=args.moe_experts,
                                                          moe_grouped_gemm=True)
        # The MoE spec's pre-MLP norm is TE's op-fuser TEFusedResidualRMSNorm, whose autograd
        # Function is not a torch op; use MCore's torch RMSNorm (inductor can fuse it).
        spec.submodules.pre_mlp_layernorm = WrappedTorchNorm
    else:
        spec = get_gpt_layer_with_transformer_engine_spec()
    model = LayerStack(cfg, spec, args.layers).cuda()
    return model


def make_step(model, prec):
    recipe = MXFP8BlockScaling() if prec == "mxfp8" else None
    names = [n for n, _ in model.named_parameters()]

    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    def step(params, inputs):
        """inputs: {"x": hidden states, "aux_scale": MoE aux-loss backward scale}. The scale is
        global state MCore keeps on MoEAuxLossAutoScaler (set_loss_scale each step); it is
        passed as a graph input instead of being baked into the graph as a constant."""
        saved = MoEAuxLossAutoScaler.main_loss_backward_scale
        MoEAuxLossAutoScaler.main_loss_backward_scale = inputs["aux_scale"]
        try:
            ctx = te.autocast(enabled=True, recipe=recipe) if recipe else contextlib.nullcontext()
            with ctx:
                out = torch.func.functional_call(model, params, (inputs["x"],))
            loss = out.float().pow(2).mean()
            grads = torch.autograd.grad(loss, [params[n] for n in names])
        finally:
            MoEAuxLossAutoScaler.main_loss_backward_scale = saved
        return [loss, *grads]

    return step, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=5632)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--precs", default="bf16,mxfp8")
    ap.add_argument("--moe-experts", type=int, default=0, help="0 = dense MLP")
    ap.add_argument("--moe-topk", type=int, default=2)
    ap.add_argument("--moe-ffn", type=int, default=1408)
    ap.add_argument("--moe-permute-fusion", action="store_true")
    ap.add_argument("--fsdp", default="param", choices=["param", "group"],
                    help="param: per-parameter Shard(0) unshard; group: v2 flat buffer per layer")
    ap.add_argument("--shard-layout", default="padded", choices=["padded", "unpadded"])
    ap.add_argument("--uneven-impl", default="pad", choices=["pad", "opaque"])
    ap.add_argument("--deterministic", action="store_true",
                    help="TransformerConfig.deterministic_mode (deterministic cuDNN attention bwd)")
    ap.add_argument("--passes", default="none,dedup+prefetch+blockbucket+sinkrs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(1, 1)
    te_opaque.patch_all()
    patch_mcore_attention_for_tracing()
    patch_mcore_router_gemm()
    patch_mcore_moe_metrics_for_tracing()
    patch_mcore_alltoall_dtoh_for_tracing()

    def emit(rec):
        rec.update(te=transformer_engine.__version__, torch=torch.__version__,
                   world=dist.get_world_size(), layers=args.layers,
                   model="mcore_te_layer" + (f"_moe{args.moe_experts}" if args.moe_experts else ""),
                   deterministic=args.deterministic, permute_fusion=args.moe_permute_fusion,
                   fsdp=args.fsdp, shard_layout=args.shard_layout,
                   uneven_impl=args.uneven_impl if args.fsdp == "group" else None)
        if rank0():
            print("RESULT " + json.dumps(rec), flush=True)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    for prec in args.precs.split(","):
        model = build(args, prec)
        layout_info = None
        if args.fsdp == "group":
            from fsdp_group_ops import apply_group_fsdp, describe
            from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
                ShardLayout,
            )

            specs = apply_group_fsdp(
                {f"layers.{i}": layer for i, layer in enumerate(model.layers)},
                shard_layout=ShardLayout[args.shard_layout.upper()],
                uneven_impl=args.uneven_impl)
            sharded = [f for _, s in specs for f in s.names]
            layout_info = [d for d in describe(specs) if d["unit"] == "layers.0"]  # same per layer
        else:
            sharded = apply_simple_fsdp(model)
        params = {n: p for n, p in model.named_parameters()}
        torch.manual_seed(1 + dist.get_rank())
        x = {"x": torch.randn(args.seq, args.batch, args.hidden, device="cuda", dtype=torch.bfloat16),
             "aux_scale": torch.ones((), device="cuda")}
        step, names = make_step(model, prec)
        rec = dict(prec=prec, stage="eager", sharded=len(sharded), params=len(names),
                   layout=layout_info)
        try:
            ref = [t.detach().clone() for t in step(params, x)]
            # run-to-run noise of eager itself (cuDNN attention backward uses atomics)
            rec.update({f"eager_rerun_{k}": v for k, v in compare(step(params, x), ref).items()})
            torch.cuda.reset_peak_memory_stats()
            eager_ms = bench(lambda: step(params, x))
            eager_peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            rec.update(ok=True, eager_ms=eager_ms, eager_peak_gb=eager_peak)
        except Exception as e:  # pylint: disable=broad-except
            rec.update(ok=False, error=" ".join(str(e).split())[:1500])
            if rank0():
                traceback.print_exc()
            emit(rec)
            continue
        emit(rec)

        rec = dict(prec=prec, stage="trace")
        try:
            t0 = time.time()
            gm = trace(step, params, x, te_custom_ops=True)
            rec.update(ok=True, trace_s=round(time.time() - t0, 1), **summarize(gm))
        except Exception as e:  # pylint: disable=broad-except
            rec.update(ok=False, error=" ".join(str(e).split())[:1500])
            if rank0():
                traceback.print_exc()
            emit(rec)
            continue
        emit(rec)

        flat = [params[n] for n in names]

        def run(g):
            with torch.no_grad():
                return g(dict(zip(names, flat)), x)

        for pname in args.passes.split(","):
            rec = dict(prec=prec, stage="pass", pass_name=pname, eager_ms=eager_ms,
                       eager_peak_gb=eager_peak)
            try:
                g, stats = apply_pass(gm, pname)
                rec.update(pass_stats=stats, **summarize(g))
                rec.update(**compare(run(g), ref))
                torch.cuda.reset_peak_memory_stats()
                rec["graph_ms"] = bench(lambda: run(g))
                rec["graph_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                rec["ok"] = True
                if args.dump and rank0():
                    os.makedirs(args.dump, exist_ok=True)
                    path = os.path.join(args.dump, f"mcore-{prec}-{pname.replace('+', '_')}.py")
                    with open(path, "w") as fh:
                        fh.write(g.print_readable(print_output=False))
            except Exception as e:  # pylint: disable=broad-except
                rec.update(ok=False, error=" ".join(str(e).split())[:1500])
                if rank0():
                    traceback.print_exc()
            emit(rec)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Probe how far torch.compile can see through Transformer Engine modules.

Each case builds a module (TE building blocks, an MCore TransformerLayer, or a
SimpleFSDP-style sharded te.Linear) and, for each precision, runs:

  explain   torch._dynamo.explain on the forward: graph count, graph breaks + reasons
  aot_eager torch.compile(fullgraph=True, backend=aot_autograd capture) fwd+bwd,
            numerics vs eager, op histogram of the captured fw/bw graphs
  partial   same as aot_eager but fullgraph=False (counts Dynamo graphs when breaks exist)
  inductor  torch.compile(fullgraph=True, backend="inductor") fwd+bwd numerics

Precisions: bf16 (no FP8), mxfp8 (bf16 weights + te.autocast(MXFP8BlockScaling)),
mxfp8_param (te.quantized_model_init MXFP8 primary weights + autocast, the v2 setup).

Run with torchrun (dist cases need >= 2 ranks; other cases only report on rank 0):
  torchrun --nproc-per-node 2 te_compile_probe.py --case te_linear --out results.jsonl
"""

import argparse
import collections
import contextlib
import copy
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

H, F, S, B, NH = 1024, 4096, 1024, 2, 8
T = S * B
PRECISIONS = ["bf16", "mxfp8", "mxfp8_param"]


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def log(*a):
    if rank() == 0:
        print(*a, flush=True)


# ----------------------------------------------------------------------------- helpers
def short_exc(e, n=6):
    lines = [l for l in traceback.format_exception_only(type(e), e) if l.strip()]
    msg = " ".join(" ".join(lines).split())
    return msg[:1500]


def te_autocast(prec):
    if prec == "bf16":
        return contextlib.nullcontext()
    return te.autocast(enabled=True, recipe=RECIPE)


def model_init(prec):
    if prec == "mxfp8_param":
        return te.quantized_model_init(recipe=RECIPE, preserve_high_precision_init_val=True)
    return contextlib.nullcontext()


def op_histogram(gms):
    hist = collections.Counter()
    for gm in gms:
        for n in gm.graph.nodes:
            if n.op == "call_function":
                t = n.target
                name = getattr(t, "name", None)
                name = name() if callable(name) else str(t)
                ns = name.split("::")[0] if "::" in name else (
                    "python:" + getattr(t, "__module__", "?").split(".")[0])
                hist[ns] += 1
    return dict(hist)


def interesting_ops(gms, limit=40):
    names = collections.Counter()
    for gm in gms:
        for n in gm.graph.nodes:
            if n.op == "call_function":
                t = n.target
                name = getattr(t, "name", None)
                name = name() if callable(name) else str(t)
                if not name.startswith("aten::") and not name.startswith("<built-in"):
                    names[name] += 1
    return dict(names.most_common(limit))


def capture_backend(store):
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    def cap(kind):
        def f(gm, example_inputs):
            store[kind].append(gm)
            return make_boxed_func(gm.forward)

        return f

    aot = aot_autograd(fw_compiler=cap("fw"), bw_compiler=cap("bw"))

    def backend(gm, example_inputs):
        store["dynamo"].append(gm)
        return aot(gm, example_inputs)

    return backend


def rel_err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


# ----------------------------------------------------------------------------- cases
# A case returns (module, make_inputs, run) where run(module, inputs) -> output tensor.


def case_te_linear(prec):
    with model_init(prec):
        m = te.Linear(H, F, params_dtype=torch.bfloat16, device="cuda")
    return m, lambda: [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


def case_te_ln_linear(prec):
    with model_init(prec):
        m = te.LayerNormLinear(H, 3 * H, normalization="RMSNorm",
                               params_dtype=torch.bfloat16, device="cuda")
    return m, lambda: [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


def case_te_ln_mlp(prec):
    with model_init(prec):
        m = te.LayerNormMLP(H, F, normalization="RMSNorm", activation="swiglu",
                            params_dtype=torch.bfloat16, device="cuda")
    return m, lambda: [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


class GroupedWrap(torch.nn.Module):
    def __init__(self, prec):
        super().__init__()
        with model_init(prec):
            self.g = te.GroupedLinear(4, H, F, params_dtype=torch.bfloat16, device="cuda")
        self.splits = [256, 512, 256, 1024]

    def forward(self, x):
        return self.g(x, self.splits)


def case_te_grouped_linear(prec):
    m = GroupedWrap(prec)
    return m, lambda: [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


class DPAWrap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dpa = te.DotProductAttention(NH, H // NH, attn_mask_type="causal", qkv_format="sbhd")

    def forward(self, q, k, v):
        return self.dpa(q, k, v)


def case_te_dpa(prec):
    m = DPAWrap()

    def inputs():
        return [torch.randn(S, B, NH, H // NH, device="cuda", dtype=torch.bfloat16,
                            requires_grad=True) for _ in range(3)]

    return m, inputs


def case_te_transformer_layer(prec):
    with model_init(prec):
        m = te.TransformerLayer(H, F, NH, normalization="RMSNorm", activation="swiglu",
                                hidden_dropout=0.0, attention_dropout=0.0,
                                self_attn_mask_type="causal", params_dtype=torch.bfloat16,
                                fuse_qkv_params=True,
                                device="cuda")
    return m, lambda: [torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


_MCORE_READY = False


def _mcore_init():
    global _MCORE_READY
    if _MCORE_READY:
        return
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    _MCORE_READY = True


class MCoreLayerWrap(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(hidden_states=x, attention_mask=None)
        return out[0] if isinstance(out, tuple) else out


def _mcore_layer(prec, spec_kind):
    _mcore_init()
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    fp8 = {} if prec == "bf16" else dict(fp8="e4m3", fp8_recipe="mxfp8",
                                          fp8_param=(prec == "mxfp8_param"))
    cfg = TransformerConfig(num_layers=1, hidden_size=H, ffn_hidden_size=F,
                            num_attention_heads=NH, normalization="RMSNorm",
                            gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                            bias_activation_fusion=True, add_bias_linear=False,
                            hidden_dropout=0.0, attention_dropout=0.0, bf16=True,
                            params_dtype=torch.bfloat16, **fp8)
    spec = (get_gpt_layer_with_transformer_engine_spec() if spec_kind == "te"
            else get_gpt_layer_local_spec(normalization="RMSNorm"))
    with model_init(prec):
        layer = TransformerLayer(cfg, spec.submodules).cuda()
    return MCoreLayerWrap(layer), lambda: [
        torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


def case_mcore_layer_te(prec):
    return _mcore_layer(prec, "te")


def case_mcore_layer_local(prec):
    if prec != "bf16":
        raise NotImplementedError("local spec has no MXFP8 path")
    return _mcore_layer(prec, "local")


# --- SimpleFSDP-style: the full weight is produced inside the graph from a shard.
def _gather(t):
    """all-gather dim 0 over WORLD and return a plain (waited) tensor."""
    import torch.distributed._functional_collectives as funcol

    out = funcol.wait_tensor(funcol.all_gather_tensor(t.contiguous(), 0, dist.group.WORLD))
    if isinstance(out, funcol.AsyncCollectiveTensor):  # eager may still hand back a wrapper
        out = out.trigger_wait()
    return out


def _reduce_scatter(t):
    import torch.distributed._functional_collectives as funcol

    out = funcol.wait_tensor(
        funcol.reduce_scatter_tensor(t.contiguous(), "sum", 0, dist.group.WORLD))
    if isinstance(out, funcol.AsyncCollectiveTensor):
        out = out.trigger_wait()
    return out


class _UnshardBF16(torch.autograd.Function):
    """Unshard as a differentiable op: forward all-gathers the Shard(0) weight,
    backward reduce-scatters its gradient (what v2's FsdpUnshardGroup would do)."""

    @staticmethod
    def forward(ctx, shard):
        return _gather(shard)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter(grad)


class ShardedTELinear(torch.nn.Module):
    """te.Linear whose `weight` is a property returning _UnshardBF16(shard)."""

    def __init__(self, prec):
        super().__init__()
        ws = dist.get_world_size()
        self.inner = te.Linear(H, F, params_dtype=torch.bfloat16, device="cuda")
        full = self.inner.weight.detach()
        shard = full.chunk(ws, dim=0)[dist.get_rank()].clone()
        self.weight_shard = torch.nn.Parameter(shard)
        del self.inner._parameters["weight"]
        mod = self.inner
        owner = self

        def weight_prop(_self):
            return _UnshardBF16.apply(owner.weight_shard)

        mod.__class__ = type("Sharded" + type(mod).__name__, (type(mod),),
                             {"weight": property(weight_prop)})

    def forward(self, x):
        return self.inner(x)


def case_simplefsdp_te_linear(prec):
    if prec == "mxfp8_param":
        raise NotImplementedError("see case simplefsdp_mxfp8_param")
    m = ShardedTELinear(prec)
    return m, lambda: [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


class _UnshardMXFP8(torch.autograd.Function):
    """v2-style unshard of MXFP8 planes: gather rowwise+columnwise data/scales, wrap
    them in an MXFP8Tensor; backward reduce-scatters the bf16 weight grad."""

    @staticmethod
    def forward(ctx, master_shard, rd, rs, cd, cs, quantizer, shape, swizzled):
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        # row-sharded planes: rowwise data [F/ws, H], rowwise scale [F/ws, H/32],
        # columnwise data [F/ws, H], columnwise scale [F/ws/32, H]
        rd, rs, cd, cs = (_gather(t) for t in (rd, rs, cd, cs))
        return MXFP8Tensor(shape=shape, dtype=torch.bfloat16, rowwise_data=rd,
                           rowwise_scale_inv=rs, columnwise_data=cd,
                           columnwise_scale_inv=cs, fp8_dtype=quantizer.dtype,
                           quantizer=quantizer, with_gemm_swizzled_scales=swizzled,
                           requires_grad=False)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter(grad).float(), None, None, None, None, None, None, None


class ShardedMXFP8Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        ws, r = dist.get_world_size(), dist.get_rank()
        self.inner = te.Linear(H, F, params_dtype=torch.bfloat16, device="cuda")
        full = self.inner.weight.detach()
        try:
            from transformer_engine.pytorch.constants import DType
        except ImportError:
            from transformer_engine_torch import DType
        q = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=True, columnwise=True)
        qt = q(full)
        self.quantizer = q

        def shard(t):
            return t.chunk(ws, dim=0)[r].contiguous()

        # Splitting dim 0 and all-gathering it back reproduces each plane byte for
        # byte (F, H are multiples of the 128x4 scale padding, so no padding here).
        self.swizzled = bool(qt._with_gemm_swizzled_scales)
        self.register_buffer("rd", shard(qt._rowwise_data))
        self.register_buffer("rs", shard(qt._rowwise_scale_inv))
        self.register_buffer("cd", shard(qt._columnwise_data))
        self.register_buffer("cs", shard(qt._columnwise_scale_inv))
        self.master = torch.nn.Parameter(shard(full).float())
        del self.inner._parameters["weight"]
        owner = self

        def weight_prop(_self):
            return _UnshardMXFP8.apply(owner.master, owner.rd, owner.rs, owner.cd,
                                       owner.cs, owner.quantizer, (F, H), owner.swizzled)

        mod = self.inner
        mod.__class__ = type("ShardedMX" + type(mod).__name__, (type(mod),),
                             {"weight": property(weight_prop)})

    def forward(self, x):
        return self.inner(x)


def case_simplefsdp_mxfp8_param(prec):
    if prec != "mxfp8_param":
        raise NotImplementedError("only mxfp8_param")
    m = ShardedMXFP8Linear()
    return m, lambda: [torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)]


def case_simplefsdp_mxfp8_param_coerce(prec):
    """Same as simplefsdp_mxfp8_param, but tell the tracer that the gradient of an
    MXFP8Tensor is a plain high-precision tensor (TE's wgrad really is bf16), via the
    tensor-subclass `__coerce_tangent_metadata__` hook."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    def _coerce(self):
        return torch.empty(self.shape, dtype=self.dtype, device=self.device)

    MXFP8Tensor.__coerce_tangent_metadata__ = _coerce
    return case_simplefsdp_mxfp8_param(prec)


CASES = {k[len("case_"):]: v for k, v in globals().items() if k.startswith("case_")}
DIST_CASES = {"simplefsdp_te_linear", "simplefsdp_mxfp8_param", "simplefsdp_mxfp8_param_coerce"}


# ----------------------------------------------------------------------------- runners
def fwd_bwd(m, inputs, prec, seed=0):
    torch.manual_seed(seed)
    with te_autocast(prec):
        out = m(*inputs)
    if isinstance(out, tuple):
        out = out[0]
    torch.manual_seed(seed + 1)
    gout = torch.randn_like(out)
    out.backward(gout)
    return out


def grads_of(m, inputs):
    gs = {}
    for n, p in m.named_parameters():
        if p.grad is not None:
            gs[n] = p.grad.detach().float().clone()
    for i, x in enumerate(inputs):
        if x.grad is not None:
            gs[f"input{i}"] = x.grad.detach().float().clone()
    return gs


def zero(m, inputs):
    for p in m.parameters():
        p.grad = None
    for x in inputs:
        x.grad = None


def run_mode(case, prec, mode):
    rec = dict(case=case, prec=prec, mode=mode, te=transformer_engine.__version__,
               torch=torch.__version__)
    torch._dynamo.reset()
    torch.manual_seed(1234)
    m, make_inputs = CASES[case](prec)
    inputs = make_inputs()

    if mode == "explain":
        def f(*xs):
            with te_autocast(prec):
                return m(*xs)

        ex = torch._dynamo.explain(f)(*inputs)
        rec.update(graph_count=ex.graph_count, graph_break_count=ex.graph_break_count,
                   op_count=ex.op_count,
                   break_reasons=[" ".join(str(r.reason).split())[:400]
                                  for r in ex.break_reasons][:12])
        rec["ok"] = True
        return rec

    # eager reference
    ref_out = fwd_bwd(m, inputs, prec).detach().float().clone()
    ref_grads = grads_of(m, inputs)
    zero(m, inputs)

    store = collections.defaultdict(list)
    if mode in ("aot_eager", "partial"):
        backend = capture_backend(store)
    else:
        backend = "inductor"
    cm = torch.compile(m, backend=backend, fullgraph=(mode != "partial"))
    t0 = time.time()
    out = fwd_bwd(cm, inputs, prec).detach().float()
    torch.cuda.synchronize()
    rec["compile_s"] = round(time.time() - t0, 1)
    grads = grads_of(m, inputs)
    rec["out_rel_err"] = rel_err(out, ref_out)
    rec["grad_rel_err"] = {k: round(rel_err(grads[k], ref_grads[k]), 5)
                           for k in ref_grads if k in grads}
    rec["missing_grads"] = sorted(set(ref_grads) - set(grads))
    if store:
        rec["dynamo_graphs"] = len(store["dynamo"])
        rec["fw_graphs"] = len(store["fw"])
        rec["bw_graphs"] = len(store["bw"])
        rec["fw_ops"] = op_histogram(store["fw"])
        rec["bw_ops"] = op_histogram(store["bw"])
        rec["fw_nonaten"] = interesting_ops(store["fw"])
        rec["bw_nonaten"] = interesting_ops(store["bw"])
        if os.environ.get("DUMP_GRAPHS") and rank() == 0:
            d = os.path.join(os.environ["DUMP_GRAPHS"], f"{case}-{prec}-{mode}")
            os.makedirs(d, exist_ok=True)
            for kind in ("dynamo", "fw", "bw"):
                for i, gm in enumerate(store[kind]):
                    with open(os.path.join(d, f"{kind}{i}.py"), "w") as fh:
                        fh.write(gm.print_readable(print_output=False))
    rec["ok"] = True
    return rec


def main():
    global RECIPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--precs", default=",".join(PRECISIONS))
    ap.add_argument("--modes", default="explain,aot_eager,partial,inductor")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if "RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    RECIPE = MXFP8BlockScaling()
    if args.case in DIST_CASES and (not dist.is_initialized() or dist.get_world_size() < 2):
        raise SystemExit("dist case needs torchrun with >= 2 ranks")

    for prec in args.precs.split(","):
        for mode in args.modes.split(","):
            t0 = time.time()
            try:
                rec = run_mode(args.case, prec, mode)
            except NotImplementedError as e:
                rec = dict(case=args.case, prec=prec, mode=mode, ok=None, skipped=str(e))
            except Exception as e:  # pylint: disable=broad-except
                rec = dict(case=args.case, prec=prec, mode=mode, ok=False, error=short_exc(e),
                           te=transformer_engine.__version__)
                if os.environ.get("VERBOSE") and rank() == 0:
                    traceback.print_exc()
            rec["wall_s"] = round(time.time() - t0, 1)
            if rank() == 0:
                print("RESULT " + json.dumps(rec), flush=True)
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if dist.is_initialized():
                dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

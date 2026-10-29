"""Whole-graph PoC: capture one training step (fwd + loss + bwd) of a SimpleFSDP-style TE model
as a single FX graph with make_fx (the torchtitan graph_trainer approach), then run FSDP graph
passes on it.

  step 1  trace with make_fx on fake tensors; report op mix, collective/TE-op order, whether TE's
          custom-op path was taken (needs torch.compiler._compile_session_context, because TE
          only uses its custom ops when torch.compiler.is_compiling() is true)
  step 2  run the traced graph on real tensors (numerics + time vs eager), then apply
          torch's FSDP passes (bucketing, overlap scheduling) and re-check numerics + time

Model: `--layers` blocks of  x + fc2(gelu(fc1(rmsnorm(x))))  with fc1/fc2 = te.Linear whose
weight is a bf16 Shard(0) parameter unsharded by an autograd.Function (all-gather fwd,
reduce-scatter bwd). Run: torchrun --nproc-per-node 4 joint_capture_probe.py --out r.jsonl
"""

import argparse
import contextlib
import copy
import json
import os
import time
import traceback

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import annotate, preserve_node_meta

import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

MODULE_FQN = "module_fqn"  # node.meta["custom"] key: owning block
PARAM_FQN = "fsdp_param_fqn"  # node.meta["custom"] key: FSDP parameter of a collective


def rank0():
    return dist.get_rank() == 0


def log(*a):
    if rank0():
        print(*a, flush=True)


# ----------------------------------------------------------------------------- model
def _wait(t):
    out = funcol.wait_tensor(t)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        out = out.trigger_wait()
    return out


class _Unshard(torch.autograd.Function):
    """all-gather dim 0 in forward, reduce-scatter (sum) the gradient in backward."""

    @staticmethod
    def forward(ctx, shard, fqn):
        ctx.fqn = fqn
        return _wait(funcol.all_gather_tensor(shard, 0, dist.group.WORLD))

    @staticmethod
    def backward(ctx, grad):
        with annotate({PARAM_FQN: ctx.fqn}):
            rs = funcol.reduce_scatter_tensor(grad.contiguous(), "sum", 0, dist.group.WORLD)
            return _wait(rs), None


class ShardedTELinear(torch.nn.Module):
    """TE linear module (te.Linear or te.LayerNormLinear) whose `weight` is a property
    returning _Unshard(shard)."""

    def __init__(self, in_f, out_f, fqn, te_cls=None, **te_kwargs):
        super().__init__()
        ws, r = dist.get_world_size(), dist.get_rank()
        te_cls = te_cls or te.Linear
        self.inner = te_cls(in_f, out_f, bias=False, params_dtype=torch.bfloat16, device="cuda",
                            **te_kwargs)
        full = self.inner.weight.detach()
        self.weight_shard = torch.nn.Parameter(full.chunk(ws, dim=0)[r].clone())
        del self.inner._parameters["weight"]
        owner, pfqn = self, fqn + ".weight"

        def weight_prop(_self):
            with annotate({PARAM_FQN: pfqn}):
                return _Unshard.apply(owner.weight_shard, pfqn)

        self.inner.__class__ = type("Sharded" + type(self.inner).__name__,
                                    (type(self.inner),), {"weight": property(weight_prop)})

    def forward(self, x):
        return self.inner(x)


class Block(torch.nn.Module):
    """x + fc2(gelu(fc1(rmsnorm(x)))); with kind="lnlinear" the norm is fused into fc1
    (te.LayerNormLinear, as in MCore's TELayerNormColumnParallelLinear)."""

    def __init__(self, h, f, fqn, kind):
        super().__init__()
        self.fqn = fqn
        if kind == "lnlinear":
            self.norm = None
            self.fc1 = ShardedTELinear(h, f, fqn + ".fc1", te.LayerNormLinear,
                                       normalization="RMSNorm")
        else:
            self.norm = torch.nn.RMSNorm(h, device="cuda", dtype=torch.bfloat16)
            self.fc1 = ShardedTELinear(h, f, fqn + ".fc1")
        self.fc2 = ShardedTELinear(f, h, fqn + ".fc2")

    def forward(self, x):
        with annotate({MODULE_FQN: self.fqn}):
            y = self.norm(x) if self.norm is not None else x
            return x + self.fc2(torch.nn.functional.gelu(self.fc1(y)))


class Model(torch.nn.Module):
    def __init__(self, layers, h, f, kind="linear"):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            Block(h, f, f"layers.{i}", kind) for i in range(layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------------------------------------------------------------------- step + tracing
def make_step(model, prec):
    recipe = MXFP8BlockScaling() if prec == "mxfp8" else None
    names = [n for n, _ in model.named_parameters()]

    def step(params, x):
        ctx = te.autocast(enabled=True, recipe=recipe) if recipe else contextlib.nullcontext()
        with ctx:
            out = torch.func.functional_call(model, params, (x,))
        loss = out.float().pow(2).mean()
        grads = torch.autograd.grad(loss, [params[n] for n in names])
        return [loss, *grads]

    return step, names


def copy_fwd_meta_to_bwd(gm):
    """Backward nodes inherit the forward node's `custom` meta (matched by seq_nr), so passes
    can tell which block a backward op belongs to (from torchtitan graph_trainer)."""
    fwd = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and "seq_nr" in n.meta and not n.meta.get("autograd_backward"):
            fwd.setdefault(n.meta["seq_nr"], n)
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.meta.get("autograd_backward") and "seq_nr" in n.meta:
            f = fwd.get(n.meta["seq_nr"])
            if f is not None and f is not n and f.meta.get("custom"):
                merged = copy.deepcopy(f.meta["custom"])
                merged.update(n.meta.get("custom", {}))
                n.meta["custom"] = merged


def trace(step, params, x, te_custom_ops):
    # error_on_nested_fx_trace=False: inline torch.compile'd helpers (e.g. MCore's jit_fuser
    # kernels) instead of erroring, as graph_trainer does.
    ctxs = [torch.autograd.set_multithreading_enabled(False), preserve_node_meta(),
            torch.compiler._non_strict_tracing_context(), torch.compiler._patch_engine_backward(),
            torch._dynamo.config.patch(error_on_nested_fx_trace=False)]
    if te_custom_ops:
        ctxs.append(torch.compiler._compile_session_context())
    # Own FakeTensorMode, as graph_trainer's minimal_fx_tracer: allow_non_fake_inputs lets real
    # module state reached during tracing (e.g. MoE dispatcher index buffers) become graph
    # constants; the ShapeEnv carries unbacked sizes from data-dependent ops.
    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    import torch.utils._pytree as pytree

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    fake_params, fake_x = pytree.tree_map_only(
        torch.Tensor, lambda t: fake_mode.from_tensor(t, static_shapes=True), (params, x))
    with contextlib.ExitStack() as st:
        st.enter_context(fake_mode)
        for c in ctxs:
            st.enter_context(c)
        gm = make_fx(step, record_stack_traces=False)(fake_params, fake_x)
    copy_fwd_meta_to_bwd(gm)
    return gm


# ----------------------------------------------------------------------------- analysis
def target_name(n):
    t = n.target
    name = getattr(t, "name", None)
    return name() if callable(name) else str(t)


def summarize(gm):
    hist, te_ops, colls, seq = {}, {}, {}, []
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        name = target_name(n)
        ns = name.split("::")[0] if "::" in name else "python"
        hist[ns] = hist.get(ns, 0) + 1
        custom = n.meta.get("custom", {})
        bwd = "B" if n.meta.get("autograd_backward") else "F"
        if ns.startswith("transformer_engine"):
            te_ops[name] = te_ops.get(name, 0) + 1
            seq.append(f"{bwd}:TE({custom.get(MODULE_FQN, '?')})")
        elif ns == "_c10d_functional" and "wait" not in name:
            short = {"all_gather_into_tensor": "AG", "reduce_scatter_tensor": "RS"}.get(
                name.split("::")[1].split(".")[0], name)
            colls[short] = colls.get(short, 0) + 1
            seq.append(f"{bwd}:{short}({custom.get(PARAM_FQN, custom.get('bucket', '?'))})")
        elif ns == "mfsdp_te":
            te_ops[name] = te_ops.get(name, 0) + 1
            seq.append(f"{bwd}:TEopaque({custom.get(MODULE_FQN, '?')})")
        elif ns == "bucketing" or "bucket" in name:
            seq.append(f"{bwd}:{name}")
    # device->host scalar reads (data-dependent sizes): block CUDA-graph capture
    host_syncs = sum(1 for n in gm.graph.nodes if n.op == "call_function"
                     and n.target is torch.ops.aten._local_scalar_dense.default)
    return dict(nodes=len(gm.graph.nodes), ns_hist=hist, te_ops=te_ops, collectives=colls,
                host_syncs=host_syncs, order=" ".join(seq))


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def compare(outs, ref):
    return dict(loss_rel=rel(outs[0], ref[0]),
                max_grad_rel=max(rel(o, r) for o, r in zip(outs[1:], ref[1:])))


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return round(t0.elapsed_time(t1) / iters, 3)


# ----------------------------------------------------------------------------- passes
def apply_pass(gm, name):
    """Return (new graph, pass stats). Pipelines are comma-free names joined by '+'."""
    import fsdp_graph_passes as P
    from torch._inductor.pattern_matcher import stable_topological_sort

    gm = copy.deepcopy(gm)
    stats = {}
    for step in name.split("+"):
        if step == "none":
            pass
        elif step == "dedup":
            stats["dedup_removed"] = P.dedup_unshard(gm)
        elif step == "prefetch":
            stats["prefetch_moved"] = P.prefetch_forward(
                gm, lambda n: n.meta.get("custom", {}).get(MODULE_FQN))
        elif step == "blockbucket":
            stats.update(P.bucket_by_block(gm))
        elif step == "sinkrs":
            stats["rs_wait_moved"] = P.sink_rs_waits(gm)
        elif step == "torchbucket":
            from torch._inductor.fx_passes.bucketing import (
                bucket_all_gather,
                bucket_reduce_scatter,
            )

            bucket_all_gather(gm)
            bucket_reduce_scatter(gm)
            stable_topological_sort(gm.graph)
        elif step == "auto_overlap":
            from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing

            gm = schedule_overlap_bucketing(gm, collective_bucketing=True,
                                            insert_overlap_deps=False)
        else:
            raise ValueError(step)
        gm.graph.lint()
        gm.recompile()
    return gm, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--precs", default="bf16,mxfp8")
    ap.add_argument("--passes", default=(
        "none,dedup,dedup+prefetch,dedup+prefetch+blockbucket+sinkrs,"
        "dedup+torchbucket,dedup+auto_overlap"))
    ap.add_argument("--block", default="linear", choices=["linear", "lnlinear"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(0)
    if args.block == "lnlinear":
        import te_opaque

        te_opaque.patch_layernorm_linear()
    model = Model(args.layers, args.hidden, args.ffn, args.block)
    params = {n: p for n, p in model.named_parameters()}
    torch.manual_seed(1 + dist.get_rank())
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)

    def emit(rec):
        rec.update(te=transformer_engine.__version__, torch=torch.__version__,
                   world=dist.get_world_size(), layers=args.layers, block=args.block)
        if rank0():
            print("RESULT " + json.dumps(rec), flush=True)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    for prec in args.precs.split(","):
        step, names = make_step(model, prec)
        ref = [t.detach().clone() for t in step(params, x)]
        torch.cuda.reset_peak_memory_stats()
        eager_ms = bench(lambda: step(params, x))
        eager_peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

        for te_ops in (False, True):
            rec = dict(prec=prec, stage="trace", te_custom_ops=te_ops)
            try:
                t0 = time.time()
                gm = trace(step, params, x, te_ops)
                rec.update(ok=True, trace_s=round(time.time() - t0, 1), **summarize(gm))
            except Exception as e:  # pylint: disable=broad-except
                rec.update(ok=False, error=" ".join(str(e).split())[:1500])
                if rank0():
                    traceback.print_exc()
                emit(rec)
                continue
            emit(rec)
            if not te_ops:
                continue

            flat = [params[n] for n in names]

            def run(g):
                with torch.no_grad():
                    return g(dict(zip(names, flat)), x)

            for pname in args.passes.split(","):
                rec = dict(prec=prec, stage="pass", pass_name=pname, eager_ms=eager_ms,
                           eager_peak_gb=eager_peak_gb)
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
                        with open(os.path.join(args.dump, f"{prec}-{pname.replace('+', '_')}.py"), "w") as fh:
                            fh.write(g.print_readable(print_output=False))
                except Exception as e:  # pylint: disable=broad-except
                    rec.update(ok=False, error=" ".join(str(e).split())[:1500])
                    if rank0():
                        traceback.print_exc()
                emit(rec)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

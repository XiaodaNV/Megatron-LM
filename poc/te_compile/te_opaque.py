"""Make a TE autograd.Function traceable as an opaque fwd/bwd custom-op pair (no TE changes).

TE 2.20.2 registers only te.Linear as a torch.library op; LayerNormLinear, LayerNormMLP,
GroupedLinear and fused attention call pybind kernels from an autograd.Function, which fails
under fake-tensor tracing. `OpaqueFunction` replaces such a Function (e.g.
`layernorm_linear._LayerNormLinear`) with a proxy whose `apply`, while compiling/tracing, calls

    mfsdp_te::opaque_fwd(Tensor[] tensors, int spec) -> Tensor[]   # [*outputs, handle]
    mfsdp_te::opaque_bwd(Tensor[] grads, Tensor handle, int spec) -> Tensor[]

The ops run the Function's own static forward/backward with a stub ctx. Non-tensor arguments
(quantizers, flags, argument dataclasses) live in a Python registry under `spec`; tensors held
by an argument dataclass (e.g. cu_seqlens in FusedAttnFwdArgs) are passed as op inputs. The ctx
(and therefore the tensors saved for backward) lives in a side table under the runtime `handle`.

Wrapped Functions (`patch_all`): LayerNormLinear, FusedAttnFunc (cuDNN attention) and
GroupedLinear (MoE experts; m_splits is a tensor input, so dropless routing keeps the graph
static-shaped). The same proxy wraps MCore's RouterGatingLinearFunction (mcore_capture_probe).
`patch_attention_backend_for_fake_tensors` works around TE choosing unfused attention when it
sees a FakeTensor query.

Real (non-fake) tensors reaching a proxy while tracing are module state and are kept as
registry constants; outputs that alias an input or another output are cloned (custom-op rule).

Limitations (PoC): saved tensors are invisible to the graph (memory passes cannot see them);
tensor-subclass inputs (MXFP8 primary weights) are not supported; outputs must be described
by an `out_meta_fn` because the fake kernels cannot run TE.
"""

import copy
import dataclasses
import itertools
from typing import List

import torch

_SPECS = {}  # spec id -> registry entry, see OpaqueFunction.apply
_CTXS = {}  # runtime handle -> stub ctx
_spec_ids = itertools.count()
_handles = itertools.count(1)


class _StubCtx:
    """Enough of FunctionCtx for TE's forward/backward."""

    def __init__(self, needs_input_grad):
        self.needs_input_grad = needs_input_grad
        self._saved = ()

    def save_for_backward(self, *tensors):
        self._saved = tensors

    @property
    def saved_tensors(self):
        return self._saved

    def mark_non_differentiable(self, *args):
        pass

    def set_materialize_grads(self, value):
        pass

    def mark_dirty(self, *args):
        pass


# Argument layout entries: ("t", None) one tensor-or-None slot; ("c", value) a constant;
# ("d", (obj, field_names)) a dataclass whose tensor fields occupy len(field_names) slots.


def _no_alias(outputs, inputs):
    """Custom-op outputs may not alias inputs or each other; TE/MCore backward functions
    sometimes return an incoming gradient (or a view of it) unchanged, so clone those."""
    seen = {t.untyped_storage().data_ptr() for t in inputs
            if isinstance(t, torch.Tensor) and t.numel() > 0}
    result = []
    for t in outputs:
        if t.numel() > 0:
            ptr = t.untyped_storage().data_ptr()
            if ptr in seen:
                t = t.clone()
                ptr = t.untyped_storage().data_ptr()
            seen.add(ptr)
        result.append(t)
    return result


def _unflatten(spec, tensors):
    """Rebuild the Function's positional args from the op's tensor list + registry entry."""
    it = iter(tensors)
    args = []
    for kind, payload in spec["layout"]:
        if kind == "t":
            args.append(next(it))
        elif kind == "c":
            args.append(payload)
        else:
            obj, fields = payload
            obj = copy.copy(obj)
            for f in fields:
                setattr(obj, f, next(it))
            args.append(obj)
    return args


@torch.library.custom_op("mfsdp_te::opaque_fwd", mutates_args=())
def opaque_fwd(tensors: List[torch.Tensor], spec: int) -> List[torch.Tensor]:
    # `Tensor[]`, not `Tensor?[]`: autograd treats optional-tensor lists as non-differentiable.
    # None arguments travel as empty placeholders and are restored from spec["is_none"].
    s = _SPECS[spec]
    ctx = _StubCtx(s["needs_grad"])
    # Below the autograd key the inputs arrive without requires_grad, but TE reads it to
    # decide what to save and whether to compute wgrad/dgrad: restore the flags.
    tensors = [None if none else (t.detach().requires_grad_(need) if t.is_floating_point() else t)
               for t, need, none in zip(tensors, s["slot_needs_grad"], s["is_none"])]
    outs = s["fn_cls"].forward(ctx, *_unflatten(s, tensors))
    outs = outs if isinstance(outs, tuple) else (outs,)
    h = next(_handles)
    _CTXS[h] = ctx
    # Only outputs declared by out_meta cross the op boundary (e.g. not GroupedLinear's
    # list of FP8 weight workspaces).
    present = _no_alias([outs[i] for i in s["present_slots"]], tensors)
    return [*present, torch.tensor([h], dtype=torch.int64)]


@opaque_fwd.register_fake
def _(tensors, spec):
    metas = _SPECS[spec]["out_meta"]
    outs = [torch.empty(shape, dtype=dtype, device=dev) for shape, dtype, dev in metas]
    return [*outs, torch.empty(1, dtype=torch.int64)]


@torch.library.custom_op("mfsdp_te::opaque_bwd", mutates_args=())
def opaque_bwd(grads: List[torch.Tensor], handle: torch.Tensor, spec: int) -> List[torch.Tensor]:
    s = _SPECS[spec]
    ctx = _CTXS.pop(int(handle.item()))
    full = [None] * len(s["out_slots"])
    for slot, g in zip(s["present_slots"], grads):
        full[slot] = g.contiguous()
    in_grads = s["fn_cls"].backward(ctx, *full)
    # in_grads has one entry per positional argument; only "t" slots can need a grad.
    return _no_alias([g for g, need in zip(in_grads, s["arg_needs_grad"]) if need],
                     [*grads, handle])


@opaque_bwd.register_fake
def _(grads, handle, spec):
    s = _SPECS[spec]
    return [torch.empty(shape, dtype=dtype, device=dev) for shape, dtype, dev in s["in_grad_meta"]]


def _setup_context(ctx, inputs, output):
    _, spec = inputs
    ctx.spec = spec
    ctx.save_for_backward(output[-1])


def _backward(ctx, grads):
    (handle,) = ctx.saved_tensors
    s = _SPECS[ctx.spec]
    user_grads = [g if g is not None else torch.zeros(()) for g in grads[:-1]]
    gs = iter(opaque_bwd(user_grads, handle, ctx.spec))
    return [next(gs) if need else None for need in s["slot_needs_grad"]], None


torch.library.register_autograd("mfsdp_te::opaque_fwd", _backward, setup_context=_setup_context)


class OpaqueFunction:
    """Drop-in replacement for an autograd.Function class: `apply` goes through the opaque
    ops while compiling/tracing and through the original Function otherwise.

    out_meta_fn(*args) -> per Function output (shape, dtype, device) or None.
    returns_tuple: whether Function.forward returns a tuple (else a single tensor).
    """

    def __init__(self, fn_cls, out_meta_fn, returns_tuple=True):
        self.fn_cls = fn_cls
        self.out_meta_fn = out_meta_fn
        self.returns_tuple = returns_tuple
        self.forward = fn_cls.forward
        self.backward = fn_cls.backward

    def __getattr__(self, name):
        # Other static helpers the module calls on the Function class (e.g. GroupedLinear's
        # _validate_or_alloc_output).
        return getattr(self.fn_cls, name)

    def apply(self, *args):
        if not torch.compiler.is_compiling():
            return self.fn_cls.apply(*args)
        from torch._subclasses.fake_tensor import is_fake

        layout, slots, slot_needs, arg_needs = [], [], [], []
        for a in args:
            if isinstance(a, torch.Tensor) and not is_fake(a):
                # A real tensor reaching the op while tracing is module state captured as a
                # constant (e.g. GroupedLinear's empty dummy biases): keep it out of the graph.
                layout.append(("c", a))
                arg_needs.append(False)
            elif a is None or isinstance(a, torch.Tensor):
                layout.append(("t", None))
                slots.append(a)
                need = a is not None and a.requires_grad
                slot_needs.append(need)
                arg_needs.append(need)
            elif dataclasses.is_dataclass(a) and not isinstance(a, type):
                fields = [f.name for f in dataclasses.fields(a)
                          if isinstance(getattr(a, f.name, None), torch.Tensor)]
                layout.append(("d", (a, fields)))
                slots.extend(getattr(a, f) for f in fields)
                slot_needs.extend([False] * len(fields))
                arg_needs.append(False)
            else:
                layout.append(("c", a))
                arg_needs.append(False)
        is_none = [t is None for t in slots]
        dev = next(t.device for t in slots if t is not None)
        slots = [torch.empty(0, device=dev) if t is None else t for t in slots]
        out_slots = self.out_meta_fn(*args)
        present = [i for i, m in enumerate(out_slots) if m is not None]
        spec = next(_spec_ids)
        _SPECS[spec] = dict(
            fn_cls=self.fn_cls, layout=layout,
            needs_grad=tuple(bool(isinstance(a, torch.Tensor) and a.requires_grad) for a in args),
            slot_needs_grad=slot_needs, arg_needs_grad=arg_needs, is_none=is_none,
            out_slots=out_slots, present_slots=present,
            out_meta=[out_slots[i] for i in present],
            in_grad_meta=[(tuple(t.shape), t.dtype, t.device)
                          for t, need in zip(slots, slot_needs) if need])
        outs = iter(opaque_fwd(slots, spec)[:-1])
        result = tuple(next(outs) if m is not None else None for m in out_slots)
        return result if self.returns_tuple else result[0]


def patch_layernorm_linear():
    """Route te.LayerNormLinear through the opaque ops while compiling/tracing."""
    import transformer_engine.pytorch.module.layernorm_linear as lnl

    if isinstance(lnl._LayerNormLinear, OpaqueFunction):
        return

    def out_meta(inp, ln_weight, ln_bias, weight, weight_workspace, bias, nt):
        activation_dtype, return_ln_out = nt[17], nt[19]
        out_shape = tuple(inp.shape[:-1]) + (weight.shape[0],)
        out = (out_shape, activation_dtype, inp.device)
        ln_out = (tuple(inp.shape), activation_dtype, inp.device) if return_ln_out else None
        return [out, ln_out, None]  # new_weight_workspace: no FP8 weight caching here

    lnl._LayerNormLinear = OpaqueFunction(lnl._LayerNormLinear, out_meta)


def patch_fused_attention():
    """Route TE's cuDNN fused attention (FusedAttnFunc) through the opaque ops."""
    import transformer_engine.pytorch.attention.dot_product_attention.backends as be

    if isinstance(be.FusedAttnFunc, OpaqueFunction):
        return

    def out_meta(q, k, v, attn_bias, softmax_offset, fwd_args):
        assert not fwd_args.return_max_logit and not fwd_args.fp8_output
        return [(tuple(q.shape[:-1]) + (v.shape[-1],), q.dtype, q.device)]

    be.FusedAttnFunc = OpaqueFunction(be.FusedAttnFunc, out_meta, returns_tuple=False)


def patch_grouped_linear():
    """Route te.GroupedLinear (MoE experts) through the opaque ops while compiling/tracing.

    m_splits (tokens per expert) is a tensor input of the op: only the GEMM partitioning
    depends on its values, the output shape is [total tokens, out_features], so the graph stays
    static-shaped under dropless routing; TE reads the splits on the host inside the op."""
    import transformer_engine.pytorch.module.grouped_linear as gl

    if isinstance(gl._GroupedLinear, OpaqueFunction):
        return

    def out_meta(inp, m_splits, nt, out, dgrad_out, *weights_and_biases):
        activation_dtype, is_first_microbatch = nt[14], nt[1]
        assert out is None and dgrad_out is None and is_first_microbatch is None
        w = weights_and_biases[0]
        out_features = w.shape[0] if w.dim() == 2 else w.shape[1]
        out_shape = (inp.shape[0],) + tuple(inp.shape[1:-1]) + (out_features,)
        return [(out_shape, activation_dtype, inp.device), None]  # (out, new_workspaces)

    gl._GroupedLinear = OpaqueFunction(gl._GroupedLinear, out_meta)


def patch_attention_backend_for_fake_tensors():
    """TE's backend selection keys on `type(query)` and only accepts torch.Tensor /
    Float8Tensor. Under make_fx the query is a FakeTensor, so FusedAttention and
    FlashAttention are disabled and TE silently falls back to UnfusedDotProductAttention
    (under Dynamo the type is reported as torch.Tensor, so torch.compile is unaffected).
    Report FakeTensor as torch.Tensor."""
    import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
    from torch._subclasses.fake_tensor import FakeTensor

    orig = dpa_utils.get_attention_backend
    if getattr(orig, "_poc_patched", False):
        return

    def get_attention_backend(params):
        if isinstance(params.qkv_type, type) and issubclass(params.qkv_type, FakeTensor):
            params = dataclasses.replace(params, qkv_type=torch.Tensor)
        return orig(params)

    get_attention_backend._poc_patched = True
    dpa_utils.get_attention_backend = get_attention_backend


def patch_all():
    patch_layernorm_linear()
    patch_fused_attention()
    patch_grouped_linear()
    patch_attention_backend_for_fake_tensors()

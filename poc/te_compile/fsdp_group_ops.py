"""Parameter-group FSDP for whole-graph capture: MFSDP v2's DBuffer layout as graph ops.

Replaces the per-parameter `_Unshard` of the earlier PoC with v2's unit of communication: all
parameters of one FSDP unit (here: one TransformerLayer) live in one flat buffer laid out by
v2's `layout_builder` (RowAtomic, PADDED or UNPADDED), each rank holding its segment. One
autograd Function per unit and forward call gathers the whole buffer and returns per-parameter
views; its backward packs the per-parameter gradients back into the flat layout and
reduce-scatters them into the local segment.

Collectives (1-D data-parallel mesh, the WORLD group):

* equal segments (PADDED layout): PyTorch functional collectives (`_c10d_functional`
  all_gather_into_tensor / reduce_scatter_tensor) -- launch and wait are separate graph nodes,
  so passes can reorder them (prefetch, wait sinking, bucketing).
* unequal segments (UNPADDED layout), selected by `uneven_impl`:
  - "opaque": torch.library custom ops `mfsdp::all_gather_uneven` /
    `mfsdp::reduce_scatter_uneven` running exactly v2's list collectives (grouped NCCL
    broadcasts / reduces). One node each: the collective completes inside the op, so graph
    passes cannot overlap it with compute.
  - "async": the same list collectives launched with async_op=True by
    `mfsdp::all_gather_uneven_start` / `mfsdp::reduce_scatter_uneven_start`, which register the
    Work on their output; the graph waits with PyTorch's `_c10d_functional.wait_tensor`, so
    launch and wait are separate nodes like the functional collectives.
  - "pad": pad every segment to the largest one and use the functional collectives above, then
    drop the padding. Sends (max - own) extra elements per rank; schedulable like the equal case.

Parameter access: `apply_group_fsdp` moves a unit's parameters into the group's flat shard
parameter and turns each original attribute into a property that returns the view produced by
the unit's current forward call (the unit's forward runs the unshard op first and publishes
the views). This is an explicit data dependency in the forward, not a hook.
"""

from typing import List

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.fx.traceback import annotate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import layout_builder
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import ShardLayout

PARAM_FQN = "fsdp_param_fqn"  # node.meta["custom"] key read by fsdp_graph_passes
_WORLD = lambda: dist.group.WORLD  # noqa: E731


def _group_name():
    return _WORLD().group_name


def _wait(t):
    out = funcol.wait_tensor(t)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        out = out.trigger_wait()
    return out


# ----------------------------------------------------------------------------- uneven, opaque
def _resolve(group_name):
    from torch.distributed.distributed_c10d import _resolve_process_group

    return _resolve_process_group(group_name)


@torch.library.custom_op("mfsdp::all_gather_uneven", mutates_args=())
def all_gather_uneven(shard: torch.Tensor, rank_sizes: List[int], group_name: str) -> torch.Tensor:
    """v2's uneven DBuffer.allgather: one list all_gather into per-peer slices of the output."""
    out = shard.new_empty(sum(rank_sizes))
    dist.all_gather(list(out.split(rank_sizes)), shard.contiguous(), group=_resolve(group_name))
    return out


@all_gather_uneven.register_fake
def _(shard, rank_sizes, group_name):
    return shard.new_empty(sum(rank_sizes))


@torch.library.custom_op("mfsdp::reduce_scatter_uneven", mutates_args=())
def reduce_scatter_uneven(
    full: torch.Tensor, rank_sizes: List[int], rank: int, group_name: str
) -> torch.Tensor:
    """v2's uneven DBuffer.reduce_scatter (SUM): list reduce_scatter of per-peer slices."""
    out = full.new_empty(rank_sizes[rank])
    dist.reduce_scatter(out, list(full.contiguous().split(rank_sizes)), op=dist.ReduceOp.SUM,
                        group=_resolve(group_name))
    return out


@reduce_scatter_uneven.register_fake
def _(full, rank_sizes, rank, group_name):
    return full.new_empty(rank_sizes[rank])


# ----------------------------------------------------------------------------- uneven, async
# Same list collectives as above, launched with async_op=True. The Work is registered on the
# output tensor, so PyTorch's functional `_c10d_functional.wait_tensor` waits for it (the pattern
# torch's own symmetric-memory ops use): launch and wait become separate graph nodes that passes
# can move apart, like the functional collectives of the equal case.
def _register(out, work):
    torch._C._distributed_c10d._register_work(out, work)
    return out


@torch.library.custom_op("mfsdp::all_gather_uneven_start", mutates_args=())
def all_gather_uneven_start(
    shard: torch.Tensor, rank_sizes: List[int], group_name: str
) -> torch.Tensor:
    out = shard.new_empty(sum(rank_sizes))
    work = dist.all_gather(list(out.split(rank_sizes)), shard.contiguous(),
                           group=_resolve(group_name), async_op=True)
    return _register(out, work)


@all_gather_uneven_start.register_fake
def _(shard, rank_sizes, group_name):
    return shard.new_empty(sum(rank_sizes))


@torch.library.custom_op("mfsdp::reduce_scatter_uneven_start", mutates_args=())
def reduce_scatter_uneven_start(
    full: torch.Tensor, rank_sizes: List[int], rank: int, group_name: str
) -> torch.Tensor:
    out = full.new_empty(rank_sizes[rank])
    work = dist.reduce_scatter(out, list(full.contiguous().split(rank_sizes)),
                               op=dist.ReduceOp.SUM, group=_resolve(group_name), async_op=True)
    return _register(out, work)


@reduce_scatter_uneven_start.register_fake
def _(full, rank_sizes, rank, group_name):
    return full.new_empty(rank_sizes[rank])


# ----------------------------------------------------------------------------- dispatch
def gather_flat(shard, rank_sizes, rank, uneven_impl):
    """Local segment -> full flat buffer (layout.size elements)."""
    ws = len(rank_sizes)
    if len(set(rank_sizes)) == 1:
        return _wait(funcol.all_gather_tensor(shard, 0, _WORLD()))
    if uneven_impl == "opaque":
        return all_gather_uneven(shard, list(rank_sizes), _group_name())
    if uneven_impl == "async":
        return _wait(all_gather_uneven_start(shard, list(rank_sizes), _group_name()))
    if uneven_impl == "pad":
        m = max(rank_sizes)
        padded = torch.cat([shard, shard.new_zeros(m - rank_sizes[rank])]) \
            if rank_sizes[rank] < m else shard
        g = _wait(funcol.all_gather_tensor(padded, 0, _WORLD()))
        return torch.cat([g.narrow(0, i * m, rank_sizes[i]) for i in range(ws)])
    raise ValueError(uneven_impl)


def reduce_scatter_flat(full, rank_sizes, rank, uneven_impl):
    """Full flat gradient -> this rank's segment, summed over ranks."""
    if len(set(rank_sizes)) == 1:
        return _wait(funcol.reduce_scatter_tensor(full, "sum", 0, _WORLD()))
    if uneven_impl == "opaque":
        return reduce_scatter_uneven(full, list(rank_sizes), rank, _group_name())
    if uneven_impl == "async":
        return _wait(reduce_scatter_uneven_start(full, list(rank_sizes), rank, _group_name()))
    if uneven_impl == "pad":
        m = max(rank_sizes)
        parts = []
        for part, n in zip(full.split(rank_sizes), rank_sizes):
            parts.append(torch.cat([part, part.new_zeros(m - n)]) if n < m else part)
        rs = _wait(funcol.reduce_scatter_tensor(torch.cat(parts), "sum", 0, _WORLD()))
        return rs.narrow(0, 0, rank_sizes[rank])
    raise ValueError(uneven_impl)


# ----------------------------------------------------------------------------- unshard op
class GroupSpec:
    """Static description of one FSDP unit's flat buffer (v2 GlobalLayout + rank)."""

    def __init__(self, fqn, names, shapes, layout, rank, uneven_impl):
        self.fqn = fqn
        self.names = names
        self.shapes = [torch.Size(s) for s in shapes]
        self.layout = layout
        self.rank = rank
        self.uneven_impl = uneven_impl
        self.rank_sizes = [layout.rank_size(r) for r in range(layout.dp_size)]
        # Flat-gradient packing order: tensors by offset, zero fill for gaps/padding.
        self.order = sorted(range(len(shapes)), key=lambda i: layout.tensor_to_offset[i])


class UnshardGroup(torch.autograd.Function):
    """flat local segment -> per-parameter full tensors; backward reduce-scatters."""

    @staticmethod
    def forward(ctx, shard, spec):
        ctx.spec = spec
        with annotate({PARAM_FQN: spec.fqn}):
            full = gather_flat(shard, spec.rank_sizes, spec.rank, spec.uneven_impl)
        return tuple(full.narrow(0, off, s.numel()).view(s)
                     for off, s in zip(spec.layout.tensor_to_offset, spec.shapes))

    @staticmethod
    def backward(ctx, *grads):
        spec = ctx.spec
        layout = spec.layout
        ref = next(g for g in grads if g is not None)
        pieces, pos = [], 0
        for i in spec.order:
            off, n = layout.tensor_to_offset[i], spec.shapes[i].numel()
            if off > pos:
                pieces.append(ref.new_zeros(off - pos))
            g = grads[i]
            pieces.append(g.reshape(-1) if g is not None else ref.new_zeros(n))
            pos = off + n
        if layout.size > pos:
            pieces.append(ref.new_zeros(layout.size - pos))
        with annotate({PARAM_FQN: spec.fqn}):
            shard_grad = reduce_scatter_flat(torch.cat(pieces), spec.rank_sizes, spec.rank,
                                             spec.uneven_impl)
        return shard_grad, None


# ----------------------------------------------------------------------------- model rewrite
def apply_group_fsdp(units, shard_layout=ShardLayout.PADDED, uneven_impl="pad"):
    """Shard every parameter of each unit (a dict fqn -> nn.Module) into v2 flat buffers, one
    per dtype (as v2's FsdpParameterGroup; e.g. MoE routers keep fp32 weights).

    Returns a list of (unit fqn, GroupSpec), one entry per (unit, dtype) group.
    """
    ws, rank = dist.get_world_size(), dist.get_rank()
    specs = []
    for unit_fqn, unit in units.items():
        owners = []  # (submodule, attr name, full fqn, tensor)
        for mod_name, mod in unit.named_modules():
            for n, p in list(mod._parameters.items()):
                if p is not None:
                    owners.append((mod, n, f"{unit_fqn}.{mod_name}.{n}".replace("..", "."), p))
        by_dtype = {}
        for idx, (*_, p) in enumerate(owners):
            by_dtype.setdefault(p.dtype, []).append(idx)

        unit_groups = []  # (flat shard param name, spec, owner indices)
        for g, (dtype, idxs) in enumerate(sorted(by_dtype.items(), key=lambda kv: str(kv[0]))):
            members = [owners[i] for i in idxs]
            shapes = [tuple(p.shape) for *_, p in members]
            layout = layout_builder.build_row_atomic_layout(
                shapes, dp_size=ws, block_size=1, shard_layout=shard_layout)
            layout.validate_for_row_atomic()
            tag = str(dtype).replace("torch.", "")
            spec = GroupSpec(f"{unit_fqn}.__fsdp_group_{tag}__", [f for _, _, f, _ in members],
                             shapes, layout, rank, uneven_impl)
            # Build this rank's segment of the flat buffer from the full parameters.
            full = torch.zeros(layout.size, dtype=dtype, device=members[0][3].device)
            for (_, _, _, p), off in zip(members, layout.tensor_to_offset):
                full.narrow(0, off, p.numel()).copy_(p.detach().reshape(-1))
            start = layout.rank_to_offset[rank]
            name = f"fsdp_flat_shard_{tag}"
            unit.register_parameter(
                name, torch.nn.Parameter(full.narrow(0, start, layout.rank_size(rank)).clone()))
            unit_groups.append((name, spec, idxs))
            specs.append((unit_fqn, spec))
        unit._fsdp_views = None

        # Submodule attributes -> views published by the unit's forward (indexed like owners).
        by_mod = {}
        for idx, (mod, n, _, _) in enumerate(owners):
            del mod._parameters[n]
            by_mod.setdefault(mod, {})[n] = idx
        for mod, attrs in by_mod.items():
            props = {n: property(lambda self, idx=idx, unit=unit: unit._fsdp_views[idx])
                     for n, idx in attrs.items()}
            mod.__class__ = type("Grp" + type(mod).__name__, (type(mod),), props)

        orig_forward = type(unit).forward

        def forward(self, *args, _orig=orig_forward, _groups=tuple(unit_groups),
                    _n=len(owners), **kwargs):
            prev = self._fsdp_views
            views = [None] * _n
            for name, spec, idxs in _groups:
                for i, v in zip(idxs, UnshardGroup.apply(getattr(self, name), spec)):
                    views[i] = v
            self._fsdp_views = views
            try:
                return _orig(self, *args, **kwargs)
            finally:
                self._fsdp_views = prev

        unit.__class__ = type("Unit" + type(unit).__name__, (type(unit),), {"forward": forward})
    return specs


def describe(specs):
    """Per-unit layout summary for the result records."""
    out = []
    for fqn, s in specs:
        logical = sum(sh.numel() for sh in s.shapes)
        out.append(dict(unit=fqn, params=len(s.shapes), logical=logical, size=s.layout.size,
                        padding=s.layout.size - logical, rank_sizes=s.rank_sizes,
                        equal=len(set(s.rank_sizes)) == 1))
    return out

"""FSDP graph passes for a make_fx joint (fwd + bwd) training-step graph.

Each pass replaces something MFSDP v2 does today with module hooks or runtime state:

  dedup_unshard      one all-gather per parameter per direction, however often the module
                     reads `weight` (v2: per-phase `_unshard_events` reuse)
  bucket_by_block    one all-gather bucket per block in forward, one reduce-scatter bucket per
                     block in backward (v2: one DBuffer per FsdpModule)
  prefetch_forward   issue block i+1's all-gather before block i's compute (v2: forward
                     prefetch over `ExecutionOrder` from `pre_forward` hooks)
  sink_rs_waits      wait for reduce-scatters only at the end of the graph, so they overlap the
                     remaining backward (v2: reduce-scatter stream + final autograd callback)

Collectives are identified by node.meta["custom"]["fsdp_param_fqn"] (set with
torch.fx.traceback.annotate when the unshard op is called); the block is the "...layers.<i>"
prefix of the FQN ("layers.3.self_attention.linear_qkv.weight" -> "layers.3").
"""

import collections
import re

import torch
from torch._inductor.pattern_matcher import stable_topological_sort

PARAM_FQN = "fsdp_param_fqn"
_c10d = torch.ops._c10d_functional


def _is(n, op):
    return n.op == "call_function" and n.target in (op.default,)


def is_ag(n):
    return _is(n, _c10d.all_gather_into_tensor)


def is_rs(n):
    return _is(n, _c10d.reduce_scatter_tensor)


def is_wait(n):
    return _is(n, _c10d.wait_tensor)


def is_bwd(n):
    return bool(n.meta.get("autograd_backward"))


def param_fqn(n):
    return n.meta.get("custom", {}).get(PARAM_FQN)


_BLOCK_RE = re.compile(r"^(.*?layers\.\d+)\.")


def block_of(n):
    """Owning block of a collective: the "...layers.<i>" prefix of its parameter FQN."""
    fqn = param_fqn(n)
    m = _BLOCK_RE.match(fqn) if fqn else None
    return m.group(1) if m else None


def _wait_of(coll):
    waits = [u for u in coll.users if is_wait(u)]
    assert len(waits) == 1, coll
    return waits[0]


def _finish(gm):
    stable_topological_sort(gm.graph)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def dedup_unshard(gm):
    """CSE identical all-gathers (same input, same group) within the same direction."""
    seen, removed = {}, 0
    for n in list(gm.graph.nodes):
        if not is_ag(n):
            continue
        key = (n.args, tuple(sorted(n.kwargs.items())), is_bwd(n))
        first = seen.get(key)
        if first is None:
            seen[key] = n
            continue
        # Collectives are side-effecting, so DCE keeps them: erase explicitly.
        w = _wait_of(n)
        w.replace_all_uses_with(_wait_of(first))
        gm.graph.erase_node(w)
        gm.graph.erase_node(n)
        removed += 1
    _finish(gm)
    return removed


def bucket_by_block(gm, mode=None):
    """Merge each block's forward all-gathers, and its backward reduce-scatters."""
    from torch._inductor.fx_passes.bucketing import merge_all_gather, merge_reduce_scatter

    ag, rs = collections.defaultdict(list), collections.defaultdict(list)
    for n in gm.graph.nodes:
        if is_ag(n) and block_of(n):
            ag[(block_of(n), is_bwd(n))].append(n)
        elif is_rs(n) and block_of(n):
            rs[block_of(n)].append(n)
    ag_buckets = [b for b in ag.values() if len(b) > 1]
    rs_buckets = [b for b in rs.values() if len(b) > 1]
    if ag_buckets:
        merge_all_gather(gm, ag_buckets, mode)
    if rs_buckets:
        merge_reduce_scatter(gm, rs_buckets, mode)
    _finish(gm)
    return dict(ag_buckets=len(ag_buckets), rs_buckets=len(rs_buckets))


def _collectives(gm, pred):
    """Collective "start" nodes (any all-gather / reduce-scatter variant, bucketed or not)."""
    out = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or not pred(n):
            continue
        name = str(n.target)
        if "_c10d_functional" in name and ("all_gather" in name or "reduce_scatter" in name):
            out.append(n)
    return out


def _move_before(target, node):
    """Move `node` and the ancestors it needs (that are not already before `target`)
    to just before `target`, preserving their relative order."""
    order = {n: i for i, n in enumerate(target.graph.nodes)}
    tpos = order[target]
    need, stack = set(), [node]
    while stack:
        cur = stack.pop()
        if cur in need or cur.op == "placeholder" or order[cur] < tpos:
            continue
        need.add(cur)
        stack.extend(cur.all_input_nodes)
    for n in sorted(need, key=order.__getitem__):
        target.prepend(n)
    return len(need)


def prefetch_forward(gm, compute_block_fn, depth=1):
    """Move forward all-gathers of block k+depth in front of block k's first compute node.

    compute_block_fn(node) -> block name for compute nodes (e.g. from the module annotation).
    """
    # Run before bucket_by_block: works on the per-parameter all-gathers.
    ag_block = collections.defaultdict(list)
    for n in gm.graph.nodes:
        if is_ag(n) and not is_bwd(n) and block_of(n):
            ag_block[block_of(n)].append(n)
    first_compute = {}
    for n in gm.graph.nodes:
        if n.op != "call_function" or is_bwd(n) or param_fqn(n) or "_c10d" in str(n.target):
            continue
        b = compute_block_fn(n)
        if b and b not in first_compute:
            first_compute[b] = n
    blocks = sorted(first_compute, key=lambda b: int(b.rsplit(".", 1)[1]))
    moved = 0
    for i, b in enumerate(blocks):
        src = blocks[i + depth] if i + depth < len(blocks) else None
        if src is None:
            continue
        for ag in ag_block.get(src, []):
            moved += _move_before(first_compute[b], ag)
    _finish(gm)
    return moved


def sink_rs_waits(gm):
    """Move each backward reduce-scatter's wait (and its consumers) to just before output."""
    out = next(n for n in gm.graph.nodes if n.op == "output")
    moved = 0
    # Reduce-scatters only occur in backward; bucketed ones carry no autograd_backward meta.
    for rs in _collectives(gm, lambda n: True):
        if "reduce_scatter" not in str(rs.target):
            continue
        for w in [u for u in rs.users if is_wait(u)]:
            # move the wait and its (unary) consumer chain that only feeds the output
            chain, stack = [], [w]
            while stack:
                cur = stack.pop()
                chain.append(cur)
                stack.extend(u for u in cur.users if u.op != "output")
            order = {n: i for i, n in enumerate(gm.graph.nodes)}
            for n in sorted(chain, key=order.__getitem__):
                out.prepend(n)
                moved += 1
    _finish(gm)
    return moved

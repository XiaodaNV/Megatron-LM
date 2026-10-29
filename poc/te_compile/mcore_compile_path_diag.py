"""Why does make_fx capture of an MCore TransformerLayer pick a different attention backend?

Wraps TE's get_attention_backend to record its inputs (AttentionParams) and decision, with TE's
backend-selection debug log routed to stdout even while tracing (TE swaps in a no-op logger
under torch.compiler.is_compiling()). Runs one layer eagerly, then traces it with make_fx as
mcore_capture_probe does, and prints both decisions plus the AttentionParams fields that differ.

Run: torchrun --nproc-per-node 1 mcore_compile_path_diag.py
"""

import dataclasses
import logging
import os
import sys

import torch
import torch.distributed as dist

import te_opaque
from joint_capture_probe import trace
from mcore_capture_probe import make_step, patch_mcore_attention_for_tracing


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    os.environ["NVTE_DEBUG"] = "1"
    os.environ["NVTE_DEBUG_LEVEL"] = "2"
    import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    log = logging.getLogger("DotProductAttention")
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    dpa_utils.no_op_logger = log  # see the selection reasoning while tracing too

    calls = []
    orig = dpa_utils.get_attention_backend

    def wrapped(params):
        out = orig(params)
        calls.append((torch.compiler.is_compiling(), params, out))
        print(f"### get_attention_backend(compiling={torch.compiler.is_compiling()}) -> "
              f"flash={out[0]} fused={out[1]} fused_backend={out[2]} unfused={out[3]} "
              f"available={out[4]}", flush=True)
        return out

    dpa_utils.get_attention_backend = wrapped
    import transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention as dpa

    if hasattr(dpa, "get_attention_backend"):
        dpa.get_attention_backend = wrapped

    parallel_state.initialize_model_parallel(1, 1)
    te_opaque.patch_all()
    patch_mcore_attention_for_tracing()
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(123)
    cfg = TransformerConfig(
        num_layers=1, hidden_size=2048, ffn_hidden_size=5632, num_attention_heads=16,
        normalization="RMSNorm", gated_linear_unit=True, activation_func=torch.nn.functional.silu,
        bias_activation_fusion=False, add_bias_linear=False, hidden_dropout=0.0,
        attention_dropout=0.0, bf16=True, params_dtype=torch.bfloat16)

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = TransformerLayer(
                cfg, get_gpt_layer_with_transformer_engine_spec().submodules, layer_number=1)

        def forward(self, x):
            return self.layer(hidden_states=x, attention_mask=None)[0]

    model = One().cuda()
    params = dict(model.named_parameters())
    x = {"x": torch.randn(2048, 2, 2048, device="cuda", dtype=torch.bfloat16),
         "aux_scale": torch.ones((), device="cuda")}
    step, _ = make_step(model, "bf16")

    print("===== eager =====", flush=True)
    step(params, x)
    print("===== make_fx trace =====", flush=True)
    gm = trace(step, params, x, te_custom_ops=True)
    ops = sorted({str(n.target) for n in gm.graph.nodes if n.op == "call_function"
                  and ("te_" in str(n.target) or "mfsdp" in str(n.target))})
    print("TE-ish ops in graph:", ops, flush=True)

    eager = [c for c in calls if not c[0]]
    traced = [c for c in calls if c[0]]
    if eager and traced:
        pe, pt = eager[-1][1], traced[-1][1]
        print("AttentionParams fields that differ (eager vs traced):")
        for f in dataclasses.fields(pe):
            a, b = getattr(pe, f.name), getattr(pt, f.name)
            try:
                same = a == b
                same = bool(same) if not isinstance(same, torch.Tensor) else bool(same.all())
            except Exception:  # pylint: disable=broad-except
                same = repr(a) == repr(b)
            if not same:
                print(f"  {f.name}: eager={a!r} traced={b!r}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

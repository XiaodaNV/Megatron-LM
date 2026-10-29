#!/bin/bash
# Runs inside the NeMo 26.08 container on one GB200 node (submitted via slurm-broker).
# Usage: bash poc/te_compile/run_probe.sh [case ...]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${HERE}/../.." && pwd)
cd "${REPO}"
PY=/opt/venv/bin/python
TE_VERSION=${TE_VERSION:-2.20.2}
OUT=${HERE}/results/${SLURM_JOB_ID:-local}-te${TE_VERSION}${TORCH_NIGHTLY:+-torchnightly}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1

echo "Job ${SLURM_JOB_ID:-?} on $(hostname): $(git log -1 --format='%h %s')"
OVERLAY=""
if [ -n "${TORCH_NIGHTLY:-}" ]; then
    # PyTorch nightly + TE built against it, layered in front of /opt/venv via PYTHONPATH.
    OVERLAY=/lustre/fsw/portfolios/coreai/users/xiaodaz/envs/torch216-nightly-te${TE_VERSION}
    bash "${HERE}/setup_nightly.sh" "${OVERLAY}" "${TE_VERSION}" || exit 1
    OVERLAY="${HERE}/nightly_site:${OVERLAY}"
    export PYTHONPATH="${OVERLAY}"
elif [ "${TE_VERSION}" != "default" ]; then
    export UV_CACHE_DIR="${HOME}/.cache/uv" PATH="${HOME}/.local/bin:${PATH}"
    uv pip install --python "$PY" --no-deps transformer-engine==${TE_VERSION} transformer-engine-cu13==${TE_VERSION}
    MAX_JOBS=$(nproc) NVTE_BUILD_MAX_JOBS=$(nproc) NVTE_CUDA_ARCHS=100 \
        uv pip install --python "$PY" --no-deps --no-build-isolation transformer-engine-torch==${TE_VERSION}
fi
"$PY" -c 'import torch, transformer_engine as te; print("torch", torch.__version__, "te", te.__version__)'
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then
    CASES=(simplefsdp_te_linear simplefsdp_mxfp8_param te_linear te_ln_linear te_ln_mlp te_grouped_linear te_dpa te_transformer_layer
           mcore_layer_te mcore_layer_local)
fi
export PYTHONPATH="${OVERLAY:+${OVERLAY}:}${REPO}" DUMP_GRAPHS="${OUT}/graphs" VERBOSE=1
port=29600
for c in "${CASES[@]}"; do
    nproc=1
    case "$c" in simplefsdp_*) nproc=2 ;; esac
    port=$((port + 1))
    echo "=== case ${c} (nproc ${nproc}) ==="
    timeout ${CASE_TIMEOUT:-900} "$PY" -m torch.distributed.run --nproc-per-node ${nproc} --master-port ${port} \
        "${HERE}/te_compile_probe.py" --case "$c" --out "${OUT}/results.jsonl" \
        > "${OUT}/${c}.log" 2>&1
    echo "exit $? ; $(grep -c '^RESULT' "${OUT}/${c}.log") results"
    grep '^RESULT' "${OUT}/${c}.log" | cut -c1-400
done
echo "results in ${OUT}"

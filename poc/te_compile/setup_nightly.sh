#!/bin/bash
# Build an overlay of PyTorch nightly (cu130, aarch64) + TE built against it, for use
# with the NeMo container's /opt/venv python via PYTHONPATH. Idempotent: reuses the
# overlay if the marker file exists.
# Usage (inside the container): bash setup_nightly.sh <overlay_dir> <te_version>
set -euo pipefail
OVL=$1
TE_VERSION=${2:-2.20.2}
PY=/opt/venv/bin/python
export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/xiaodaz/.cache/uv}"
export PATH="${HOME}/.local/bin:${PATH}"
if [ -f "${OVL}/.ready" ]; then
    echo "overlay ready: ${OVL} ($(cat "${OVL}/.ready"))"
    exit 0
fi
rm -rf "${OVL}" && mkdir -p "${OVL}"
start=$SECONDS
export PIP_CONSTRAINT=  # the container pins torch==2.13 in /etc/pip/constraint.txt
env | grep -E '^(UV_|PIP_)' || true
"$PY" -m pip --version || "$PY" -m ensurepip --upgrade
# pip, not uv: uv in this container treats the container's torch 2.13 as satisfying the
# request. Wheels are pre-downloaded on the login node (compute-node downloads stalled):
#   pip download --pre torch==<ver>+cu130 --index-url .../whl/nightly/cu130 \
#       --extra-index-url https://pypi.org/simple -d .wheelhouse
WHEELHOUSE=${WHEELHOUSE:-$(cd "$(dirname "$0")" && pwd)/.wheelhouse}
"$PY" -m pip install --quiet --no-index --find-links "${WHEELHOUSE}" --target "${OVL}" \
    "$(ls "${WHEELHOUSE}"/torch-*.whl)"
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/nightly_site:${OVL}"
"$PY" -c 'import torch, sys; print("nightly torch", torch.__version__, torch.version.cuda, torch.__file__); sys.exit(0 if torch.__version__.startswith("2.16") else 1)'
uv pip install --python "$PY" --target "${OVL}" --no-deps \
    transformer-engine==${TE_VERSION} transformer-engine-cu13==${TE_VERSION}
# The torch extension wheel installs into transformer_engine/, which already exists in
# the overlay, so `pip install --target` skips it; build the wheel and unzip it instead.
rm -rf "${OVL}/.wheels"
MAX_JOBS=$(nproc) NVTE_BUILD_MAX_JOBS=$(nproc) NVTE_CUDA_ARCHS=100 \
    "$PY" -m pip wheel --no-deps --no-build-isolation --no-cache-dir -w "${OVL}/.wheels" \
    transformer-engine-torch==${TE_VERSION}
"$PY" -m zipfile -e "${OVL}"/.wheels/transformer_engine_torch-*.whl "${OVL}"
ls "${OVL}"/transformer_engine/wheel_lib/*.so
"$PY" -c 'import torch, transformer_engine.pytorch as te, transformer_engine as t, transformer_engine_torch as tex; print("te", t.__version__, tex.__file__)'
echo "torch $("$PY" -c 'import torch; print(torch.__version__)') te ${TE_VERSION}" > "${OVL}/.ready"
echo "overlay built in $((SECONDS - start))s"

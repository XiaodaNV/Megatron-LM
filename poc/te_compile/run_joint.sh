#!/bin/bash
# Whole-graph capture PoC on one GB200 node with the torch-nightly + TE overlay.
# Usage (inside the NeMo container): bash poc/te_compile/run_joint.sh [extra probe args]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "${HERE}/../.." && pwd)
cd "${REPO}"
PY=/opt/venv/bin/python
TE_VERSION=${TE_VERSION:-2.20.2}
OUT=${HERE}/results/${SLURM_JOB_ID:-local}-${TAG:-joint}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/run.log") 2>&1
echo "Job ${SLURM_JOB_ID:-?} on $(hostname): $(git log -1 --format='%h %s')"

OVERLAY=/lustre/fsw/portfolios/coreai/users/xiaodaz/envs/torch216-nightly-te${TE_VERSION}
bash "${HERE}/setup_nightly.sh" "${OVERLAY}" "${TE_VERSION}" || exit 1
export PYTHONPATH="${HERE}/nightly_site:${OVERLAY}:${REPO}"

timeout ${TIMEOUT:-2400} "$PY" -m torch.distributed.run --nproc-per-node ${NPROC:-4} \
    "${HERE}/${PROBE:-joint_capture_probe.py}" --out "${OUT}/results.jsonl" --dump "${OUT}/graphs" "$@"
echo "exit $?"
grep '^RESULT' "${OUT}/run.log" | cut -c1-600

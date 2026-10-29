#!/bin/bash
# Group-FSDP (v2 DBuffer layout as graph ops) experiment matrix on one GB200 node.
# Submit from a login node (outside the Claude sandbox): bash poc/te_compile/submit_group_fsdp.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=/lustre/fsw/portfolios/coreai/users/xiaodaz/images/nvcr.io+nvidia+nemo+26.08.00.sqsh
MOUNTS=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_mcore/users/xiaodaz:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_mcore/users/xiaodaz,/lustre/fsw/portfolios/coreai/users/xiaodaz:/lustre/fsw/portfolios/coreai/users/xiaodaz
mkdir -p "${HERE}/results"

sbatch --parsable --job-name=group-fsdp --account=coreai_dlalgo_mcore --partition=batch \
    --qos=short --nodes=1 --gpus-per-node=4 --time=01:30:00 \
    --output="${HERE}/results/sbatch-group-fsdp-%j.out" \
    --wrap="srun --container-image=${IMAGE} --container-mounts=${MOUNTS} bash -c '
set -u
R=${HERE}/run_joint.sh
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 PROBE=mcore_capture_probe.py
D=\"--deterministic\"
TAG=grp-dense-padded        bash \$R \$D --fsdp group --shard-layout padded
TAG=grp-dense-unpad-pad     bash \$R \$D --fsdp group --shard-layout unpadded --uneven-impl pad
TAG=grp-dense-unpad-opaque  bash \$R \$D --fsdp group --shard-layout unpadded --uneven-impl opaque
TAG=grp-moe-padded          bash \$R \$D --fsdp group --shard-layout padded --moe-experts 8 --moe-permute-fusion --precs bf16
TAG=grp-moe-unpad-pad       bash \$R \$D --fsdp group --shard-layout unpadded --uneven-impl pad --moe-experts 8 --moe-permute-fusion --precs bf16
'"

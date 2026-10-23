#!/bin/bash -l
#SBATCH -A naiss2025-5-643-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=110G
#SBATCH --gpus=1
#SBATCH --requeue
#SBATCH -t 24:00:00
#SBATCH -J ropc_lrim_16
#SBATCH -o /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/ROPC_lrim_16_%A_%a.out
#SBATCH -e /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/ROPC_lrim_16_%A_%a.err
set -Eeuo pipefail
PROJECT="${ROPC_PROJECT_DIR:-}"
if [[ -z "${PROJECT}" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  for candidate in "${SCRIPT_DIR}/../.." "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/../.." \
      "${ROPC_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}/code/ROPC"; do
    if [[ -f "${candidate}/ropc/cli.py" ]]; then
      PROJECT="$(cd -- "${candidate}" && pwd)"; break
    fi
  done
fi
[[ -n "${PROJECT}" ]] || { echo '[ERROR] Set ROPC_PROJECT_DIR to the repository.' >&2; exit 2; }
export ROPC_PROJECT_DIR="${PROJECT}"
exec bash "${PROJECT}/scripts/run_arrhenius.sh" lrim-16 "$@"

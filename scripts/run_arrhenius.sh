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
#SBATCH -J ropc
#SBATCH -o /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/ROPC_%A_%a.out
#SBATCH -e /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/logs/ROPC_%A_%a.err
set -Eeuo pipefail
umask 002

BASE="${ROPC_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}"
CACHE="${ROPC_CACHE_DIR:-${BASE}/cache}"
LOGS="${ROPC_LOGS_DIR:-${BASE}/logs}"
OUTPUTS="${ROPC_OUTPUTS_DIR:-${BASE}/outputs}"
DATA_ROOT="${ROPC_DATA_ROOT:-${BASE}/datasets/ogb_cache}"
VENV="${ROPC_VENV_DIR:-${BASE}/code/venv_arrhenius_gh200}"
PYTHON="${ROPC_PYTHON:-${VENV}/bin/python}"
DATASET="${1:-}"
MODE="${2:-full}"
SEED="${ROPC_SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
DEVICE="${ROPC_DEVICE:-cuda}"
SPLIT="${ROPC_SPLIT_INDEX:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${ROPC_PROJECT_DIR:-}"
if [[ -z "${PROJECT}" ]]; then
  for candidate in "${SCRIPT_DIR}/.." "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/../.." "${BASE}/code/ROPC"; do
    if [[ -f "${candidate}/ropc/cli.py" ]]; then
      PROJECT="$(cd -- "${candidate}" && pwd)"
      break
    fi
  done
fi
PROJECT="${PROJECT:-${BASE}/code/ROPC}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_arrhenius.sh DATASET COMMAND [ARGUMENT]

  init                   Create cluster directories before sbatch submission
  dry-run [full|pilot|smoke]  Print the resolved command without writes
  preflight              Validate geometry and sparse forward/backward support
  prepare                Download/process the selected dataset
  smoke                  Complete workflow on an explicitly synthetic fixture
  pilot                  Real selected dataset with smaller training budgets
  full                   Full selected configuration
  resume /absolute/suite Verify and reuse original source/config/data/seed settings

Submit compute commands with sbatch. Per-dataset wrappers are in experiments/.
Independent seeds: sbatch --array=0-9%2 experiments/ogbn-arxiv/run.sh full
Overrides: ROPC_BASE, ROPC_PROJECT_DIR, ROPC_DATA_ROOT, ROPC_CONFIG, ROPC_SEED,
ROPC_DEVICE, ROPC_SPLIT_INDEX, ROPC_VENV_DIR, ROPC_PYTHON, ROPC_INSTALL_MISSING.
ROPC_ALLOW_LOCAL=1, ROPC_SKIP_MODULES=1 and ROPC_SKIP_ARCH_CHECK=1 enable local checks.
When changing ROPC_BASE, also give sbatch --output/--error existing log paths;
Slurm directives contain literal server paths and do not expand shell variables.
EOF
}

if [[ "${DATASET}" == help || "${DATASET}" == --help || "${DATASET}" == -h || -z "${DATASET}" ]]; then
  usage; exit 0
fi
SLUG="${DATASET,,}"
SLUG="${SLUG//_/-}"
CONFIG_SOURCE="${ROPC_CONFIG:-${PROJECT}/experiments/${SLUG}/config.json}"
[[ "${CONFIG_SOURCE}" == /* ]] || CONFIG_SOURCE="${PWD}/${CONFIG_SOURCE}"
case "${MODE}" in
  help|-h|--help) usage; exit 0 ;;
  init)
    mkdir -p "${LOGS}" "${OUTPUTS}" "${CACHE}/ropc/jobs" "${DATA_ROOT}"
    printf '[READY] logs=%s\noutputs=%s\n' "${LOGS}" "${OUTPUTS}"
    exit 0 ;;
  dry-run)
    PRESET="${3:-full}"
    case "${PRESET}" in full|pilot|smoke) ;; *) echo '[ERROR] Invalid dry-run preset.' >&2; exit 2 ;; esac
    cmd=("${PYTHON}" -m ropc run --dataset "${DATASET}" --preset "${PRESET}" --config "${CONFIG_SOURCE}"
         --data-root "${DATA_ROOT}" --output '<suite>/run' --seeds "${SEED}" --device "${DEVICE}" --split-index "${SPLIT}")
    [[ "${PRESET}" != smoke ]] || cmd+=(--synthetic)
    printf 'account=%s\npartition=%s\nproject=%s\n' naiss2025-5-643-gpu gpu "${PROJECT}"
    printf 'command:'; printf ' %q' "${cmd[@]}"; printf '\n'
    exit 0 ;;
  preflight|prepare|smoke|pilot|full|resume) ;;
  *) usage >&2; exit 2 ;;
esac
[[ "${SEED}" =~ ^[0-9]+$ && "${SPLIT}" =~ ^[0-9]+$ ]] || { echo '[ERROR] Seed/split must be nonnegative integers.' >&2; exit 2; }
case "${DEVICE}" in cpu|cuda|auto) ;; *) echo '[ERROR] Invalid ROPC_DEVICE.' >&2; exit 2 ;; esac
if [[ -z "${SLURM_JOB_ID:-}" && "${ROPC_ALLOW_LOCAL:-0}" != 1 ]]; then
  echo '[ERROR] Submit compute commands using sbatch.' >&2; exit 2
fi
[[ -f "${PROJECT}/ropc/cli.py" ]] || { echo "[ERROR] Missing package under ${PROJECT}; set ROPC_PROJECT_DIR." >&2; exit 2; }
mkdir -p "${LOGS}" "${OUTPUTS}" "${CACHE}/ropc/jobs" "${DATA_ROOT}"
if [[ "${ROPC_SKIP_MODULES:-0}" != 1 ]]; then
  module --force purge
  module load GPU/buildenv-nvhpc/25.9-cu13.0
  module load GPU/Python/3.13.5-bare-gcc-2025b-eb
fi
export ROPC_BASE="${BASE}" ROPC_VENV_DIR="${VENV}" ROPC_CACHE_DIR="${CACHE}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${OMP_NUM_THREADS}" NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export TORCH_HOME="${CACHE}/torch" XDG_CACHE_HOME="${CACHE}/xdg" PIP_CACHE_DIR="${CACHE}/pip"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset PYTHONPATH || true
ROPC_SKIP_MODULES=1 bash "${PROJECT}/scripts/bootstrap_arrhenius.sh"
cd -- "${PROJECT}"
if [[ "${MODE}" == preflight ]]; then
  exec "${PYTHON}" -m ropc validate --device "${DEVICE}" --threads "${OMP_NUM_THREADS}"
elif [[ "${MODE}" == prepare ]]; then
  exec "${PYTHON}" -m ropc prepare --dataset "${DATASET}" --data-root "${DATA_ROOT}" --seed "${SEED}" --split-index "${SPLIT}"
fi

JOB_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local_$$}}_${SLURM_ARRAY_TASK_ID:-0}"
POINTER="${CACHE}/ropc/jobs/${JOB_KEY}.suite"
SUITE=""
RESUMING=0
if [[ "${MODE}" != resume && "${SLURM_RESTART_COUNT:-0}" -gt 0 && -f "${POINTER}" ]]; then
  MODE=resume; SUITE="$(cat "${POINTER}")"
fi
if [[ "${MODE}" == resume ]]; then
  SUITE="${SUITE:-${3:?Provide the existing suite path}}"
  SUITE="$(cd -- "${SUITE}" && pwd)"
  saved_text="$("${PYTHON}" "${PROJECT}/scripts/cluster_manifest.py" resume --suite "${SUITE}")"
  mapfile -t saved <<< "${saved_text}"
  if [[ "${SLUG}" != "${saved[0]}" ]]; then
    echo '[ERROR] Dataset wrapper does not match the saved suite.' >&2; exit 2
  fi
  DATASET="${saved[0]}"; PRESET="${saved[1]}"; SEED="${saved[2]}"; CONFIG="${saved[3]}"
  DATA_ROOT="${saved[4]}"; DEVICE="${ROPC_DEVICE:-${saved[5]}}"; SPLIT="${saved[6]}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "${SEED}" ]]; then
    echo '[ERROR] Resume this seed as a single job or matching array task.' >&2; exit 2
  fi
  RESUMING=1
else
  PRESET="${MODE}"
  [[ -f "${CONFIG_SOURCE}" ]] || { echo "[ERROR] Missing dataset configuration: ${CONFIG_SOURCE}" >&2; exit 2; }
  exec 9>"${CACHE}/ropc/suite_number.lock"; flock -x 9
  counter="${CACHE}/ropc/next_suite_number"
  serial=1; [[ ! -s "${counter}" ]] || serial="$(cat "${counter}")"
  [[ "${serial}" =~ ^[0-9]+$ ]] || { echo '[ERROR] Invalid suite counter.' >&2; exit 2; }
  while :; do
    SUITE="${OUTPUTS}/ROPC_${SLUG//-/_}_SUITE_$(printf '%04d' "${serial}")_seed${SEED}"
    [[ -e "${SUITE}" ]] || break
    serial=$((serial+1))
  done
  mkdir -p "${SUITE}/source"
  printf '%s\n' "$((serial+1))" > "${counter}"
  flock -u 9
  for item in ropc scripts experiments configs protocols tests docs .github .gitignore README.md LICENSE NOTICE.md CITATION.bib pyproject.toml requirements.txt; do
    if [[ -d "${PROJECT}/${item}" ]]; then
      tar -C "${PROJECT}" --exclude=__pycache__ --exclude='*.pyc' -cf - "${item}" | tar -C "${SUITE}/source" -xf -
    elif [[ -f "${PROJECT}/${item}" ]]; then
      cp -a "${PROJECT}/${item}" "${SUITE}/source/"
    fi
  done
  CONFIG="selected_config.json"
  cp -- "${CONFIG_SOURCE}" "${SUITE}/source/${CONFIG}"
  "${PYTHON}" "${PROJECT}/scripts/cluster_manifest.py" create --suite "${SUITE}" --dataset "${SLUG}" \
    --preset "${PRESET}" --seed "${SEED}" --config "${CONFIG}" --data-root "${DATA_ROOT}" --device "${DEVICE}" --split-index "${SPLIT}"
fi
exec 8>"${SUITE}/.cluster.lock"
flock -n 8 || { echo '[ERROR] This suite is already running.' >&2; exit 2; }
printf '%s\n' "${SUITE}" > "${POINTER}"
exec > >(tee -a "${SUITE}/stdout.log") 2> >(tee -a "${SUITE}/stderr.log" >&2)
HELPER="${SUITE}/source/scripts/cluster_manifest.py"
on_error() {
  local rc=$?
  trap - ERR
  "${PYTHON}" "${HELPER}" failure --suite "${SUITE}" --code "${rc}" || true
  echo "[FAILED] Resume with: sbatch experiments/${SLUG}/run.sh resume ${SUITE}" >&2
  exit "${rc}"
}
trap on_error ERR
module list > "${SUITE}/modules.txt" 2>&1 || true
"${PYTHON}" -m pip freeze > "${SUITE}/pip_freeze.txt"
uname -a > "${SUITE}/uname.txt"
nvidia-smi > "${SUITE}/nvidia_smi.txt" 2>&1 || true
cmd=("${PYTHON}" -u -m ropc run --dataset "${DATASET}" --preset "${PRESET}" --config "${SUITE}/source/${CONFIG}"
     --data-root "${DATA_ROOT}" --output "${SUITE}/run" --seeds "${SEED}" --device "${DEVICE}"
     --split-index "${SPLIT}" --threads "${OMP_NUM_THREADS}")
[[ "${PRESET}" != smoke ]] || cmd+=(--synthetic)
[[ "${RESUMING}" != 1 ]] || cmd+=(--resume)
echo "[OUTPUT] ${SUITE}"
printf '[COMMAND]'; printf ' %q' "${cmd[@]}"; printf '\n'
cd -- "${SUITE}/source"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun --ntasks=1 "${cmd[@]}"
else
  "${cmd[@]}"
fi
"${PYTHON}" "${HELPER}" finish --suite "${SUITE}"
ln -sfn "${SUITE}" "${OUTPUTS}/ROPC_${SLUG//-/_}_LATEST_seed${SEED}"
echo "[COMPLETED] ${SUITE}"

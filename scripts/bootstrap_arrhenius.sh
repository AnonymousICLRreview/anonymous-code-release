#!/bin/bash
# Persistent ARM64 environment shared by Arrhenius jobs.
set -Eeuo pipefail
BASE="${ROPC_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls}"
VENV="${ROPC_VENV_DIR:-${BASE}/code/venv_arrhenius_gh200}"
CACHE="${ROPC_CACHE_DIR:-${BASE}/cache}"
PYTHON="${ROPC_PYTHON:-${VENV}/bin/python}"
if [[ "${ROPC_SKIP_MODULES:-0}" != 1 ]]; then
  module --force purge
  module load GPU/buildenv-nvhpc/25.9-cu13.0
  module load GPU/Python/3.13.5-bare-gcc-2025b-eb
fi
mkdir -p "${CACHE}/ropc" "${CACHE}/pip"
exec 9>"${CACHE}/ropc/venv.lock"
flock -x 9
if [[ -z "${ROPC_PYTHON:-}" && ! -x "${PYTHON}" ]]; then
  if [[ -e "${VENV}" ]]; then
    echo "[ERROR] Existing environment has no executable Python: ${VENV}" >&2
    exit 2
  fi
  python3 -m venv "${VENV}"
fi
[[ -x "${PYTHON}" ]] || { echo "[ERROR] Python is not executable: ${PYTHON}" >&2; exit 2; }
missing="$("${PYTHON}" - <<'PY'
import importlib.util
packages = {'torch':'torch', 'numpy':'numpy>=1.26,<3', 'scipy':'scipy>=1.11,<2', 'ogb':'ogb==1.3.6',
            'torch_geometric':'torch-geometric>=2.7,<3', 'sklearn':'scikit-learn>=1.4,<2',
            'pandas':'pandas>=2.1,<4', 'huggingface_hub':'huggingface-hub>=0.30,<2',
            'outdated':'outdated>=0.2.2', 'xxhash':'xxhash>=3'}
print(' '.join(package for module, package in packages.items()
               if importlib.util.find_spec(module) is None))
PY
)"
if [[ -n "${missing}" && "${ROPC_INSTALL_MISSING:-1}" != 1 ]]; then
  echo "[ERROR] Missing packages: ${missing}; ROPC_INSTALL_MISSING=0 disables installation." >&2
  exit 2
fi
if [[ " ${missing} " == *' torch '* ]]; then
  "${PYTHON}" -m pip install 'torch==2.12.1+cu130' --index-url https://download.pytorch.org/whl/cu130
  missing=" ${missing} "
  missing="${missing/ torch / }"
fi
if [[ -n "${missing// /}" ]]; then
  read -ra packages <<< "${missing}"
  "${PYTHON}" -m pip install "${packages[@]}"
fi
flock -u 9
"${PYTHON}" - <<'PY'
import importlib.metadata as metadata
import os, platform, re, torch, numpy, scipy
requirements = {'torch':'>=2.5,<3', 'numpy':'>=1.26,<3', 'scipy':'>=1.11,<2', 'ogb':'==1.3.6',
                'torch-geometric':'>=2.7,<3', 'scikit-learn':'>=1.4,<2', 'pandas':'>=2.1,<4',
                'huggingface-hub':'>=0.30,<2', 'outdated':'>=0.2.2', 'xxhash':'>=3'}
def release(text):
    match = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?', text)
    if match is None:
        raise SystemExit('Cannot determine installed package version: '+text)
    return tuple(int(value or 0) for value in match.groups())
incompatible = []
for package, bounds in requirements.items():
    actual = metadata.version(package)
    current = release(actual)
    for bound in bounds.split(','):
        operator, minimum = re.fullmatch(r'(>=|<|==)(.+)', bound).groups()
        target = release(minimum)
        valid = current >= target if operator == '>=' else current < target if operator == '<' else current == target
        if not valid:
            incompatible.append(f'{package}=={actual}; required {package}{bounds}')
            break
if incompatible:
    raise SystemExit('Existing environment has incompatible packages; install the compatible requirements before resubmission:\n  '+
                     '\n  '.join(incompatible)+'\nThe bootstrap does not replace existing packages automatically.')
if platform.machine() != 'aarch64' and os.environ.get('ROPC_SKIP_ARCH_CHECK') != '1':
    raise SystemExit('Expected Arrhenius ARM64/aarch64 Python')
print('[ENV]', platform.machine(), 'torch', torch.__version__, 'CUDA', torch.version.cuda,
      'numpy', numpy.__version__, 'scipy', scipy.__version__)
PY

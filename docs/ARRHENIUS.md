# Running ROPC on Arrhenius

Every dataset has its own configuration and Python/shell entry point under
`experiments/<dataset>/`. The scripts share one launcher and one Python
pipeline. A Slurm array runs independent seeds; each array task receives a
separate suite, source snapshot, and log pair.

## Installation and preserved server settings

Place the repository at:

```text
/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/code/ROPC
```

The launcher preserves these values from the supplied server script:

| Setting | Value |
|---|---|
| Allocation account | `naiss2025-5-643-gpu` |
| Partition | `gpu` |
| Storage base | `/nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls` |
| Python environment | `<base>/code/venv_arrhenius_gh200` |
| Build module | `GPU/buildenv-nvhpc/25.9-cu13.0` |
| Python module | `GPU/Python/3.13.5-bare-gcc-2025b-eb` |
| Architecture | ARM64 / `aarch64` |
| Data root | `<base>/datasets/ogb_cache` |
| Cache, logs, outputs | `<base>/cache`, `<base>/logs`, `<base>/outputs` |

The default allocation requests one GPU, 16 CPUs, 110 GB host memory, and 24
hours. These are adjustable resource requests, not measured dataset-specific
requirements. Use `sbatch --time`, `--mem`, and `--cpus-per-task` to match the
experiment and site limits.

The environment bootstrap uses a lock to protect shared installation. It
reuses an existing Python environment and installs missing packages only. A
missing PyTorch installation uses `torch==2.12.1+cu130` from the official CUDA
13.0 wheel index, matching the supplied server configuration. An existing
PyTorch build is retained and checked by the runtime preflight. Missing data
loader dependencies, including the Hugging Face client for LRIM, are installed
through pip using the bounds in `pyproject.toml`. Existing packages are checked
against those bounds; PyTorch earlier than 2.5 or PyG earlier than 2.7 raises a
clear environment error. Existing packages are not silently replaced.
Set `ROPC_INSTALL_MISSING=0`
to make missing dependencies an explicit error.

## First run

Run `init` before submitting the first job: Slurm opens its output files before
the shell script can create their directory.

```bash
cd /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/code/ROPC
bash experiments/ogbn-arxiv/run.sh init
bash experiments/ogbn-arxiv/run.sh dry-run full
sbatch --time=00:30:00 experiments/ogbn-arxiv/run.sh preflight
sbatch --time=02:00:00 experiments/ogbn-arxiv/run.sh smoke
sbatch experiments/ogbn-arxiv/run.sh pilot
sbatch experiments/ogbn-arxiv/run.sh full
```

Review each completed step before submitting a larger run. `preflight` checks
action projection and sparse forward/backward support without a download.
`smoke` explicitly uses a small synthetic fixture with the selected dataset's
task semantics. `pilot` uses the actual dataset and smaller training budgets.
`full` uses its Table IX settings and full budgets. The time limits above are
submission examples; completion time depends on the runtime and selected task.

The same commands apply to every dataset. For example:

```bash
bash experiments/peptides-func/run.sh dry-run full
sbatch experiments/peptides-func/run.sh pilot
sbatch experiments/lrim-32/run.sh full
```

The shared launcher is also callable directly:

```bash
sbatch scripts/run_arrhenius.sh ogbn-arxiv full
sbatch scripts/run_arrhenius.sh ogbn-arxiv prepare
```

Dataset preparation uses the selected loader, official split, and source
fingerprint. It does not substitute an arxiv cache for another dataset.

## Independent seeds and split selection

```bash
sbatch --array=0-9%2 experiments/ogbn-arxiv/run.sh full
```

Each task uses `SLURM_ARRAY_TASK_ID` as its seed. `ROPC_SEED` overrides that
value for an individual job. Split index is separate from training seed:

```bash
ROPC_SPLIT_INDEX=2 sbatch experiments/roman-empire/run.sh full
```

The launcher does not embed comma-separated seeds, banks, or variants in a
Slurm `--export` string. Environment variables inherit normally through
Slurm's default `ALL` export. If the calling environment changes that default,
pass `sbatch --export=ALL` explicitly.

For several seeds in one sequential Python invocation, use the package CLI:

```bash
python -m ropc run --dataset ogbn-arxiv --preset full \
  --config experiments/ogbn-arxiv/config.json \
  --data-root /path/to/data --output /path/to/run \
  --seeds 0,1,2 --device cuda
```

Its reduction includes every requested seed. Sample SD uses `ddof=1`; 95%
Student t intervals are descriptive across training seeds and remain null for
one observation. Metric direction follows the registered task: regression
errors are minimized. Seed intervals do not establish generalization across
independent graphs.

Combine completed independent array suites without another GPU job:

```bash
python scripts/summarize_suites.py \
  '/path/to/outputs/ROPC_ogbn_arxiv_SUITE_*_seed*' \
  --output /path/to/seed_summary
```

The summarizer accepts suite directories or their `run/` directories. It
requires one dataset, configuration, code identity, split index, preset, and
synthetic status; duplicate seeds are rejected. It writes strict JSON plus
observation, per-arm summary, and paired-difference CSV files. A single seed has
no estimated sample SD or t interval. Choose an explicit set of paths when the
output directory contains several experiments.

Family A2 uses the mean of the frozen executed action traces on value-training
graphs, then retrains the backbone under this fixed schedule. Full runs use
every value graph; pilot and smoke runs limit the sample to their branch
budget. The selected IDs and this schedule definition are saved in
`distill_A2.json`. Validation and test graph traces do not fit that schedule.
Per-input compile and latency diagnostics describe the bound input; they do
not stand in for a whole-family timing study.

Audit results remain separate from successful command completion. If an
initial audit and its predeclared reserve fallback both fail, evaluation still
reports that tested policy with `deployment_status: uncertified` and
`deployment_audit_pass: false`.

## Source snapshots and resumption

The launcher prints the suite directory, for example:

```text
<base>/outputs/ROPC_ogbn_arxiv_SUITE_0001_seed0
```

The suite contains `source/`, `run/`, `cluster_manifest.json`, `stdout.log`,
`stderr.log`, module/package information, and `ROPC_RESULTS.zip`. The snapshot
contains the package, configurations, launchers, tests, and documentation;
Python caches are omitted. The selected configuration is copied to
`source/selected_config.json`.

Resume with the exact suite path printed by the job:

```bash
sbatch experiments/ogbn-arxiv/run.sh resume \
  /nobackup/proj/disk/naiss2025-6-428/from-alvis/tnnls/outputs/ROPC_ogbn_arxiv_SUITE_0001_seed0
```

Resumption verifies the source snapshot, restores the saved dataset, preset,
seed, configuration, split index, and data root, and invokes the snapshot's
Python package with `--resume`. The Python run also verifies code/configuration
identity and dataset fingerprints. Completed task artifacts must match their
hashes before the task is skipped. Changed scientific inputs require a new
output directory. A lock prevents concurrent workers from modifying one suite.

Slurm requeue uses the same recorded suite pointer. The launcher does not
submit an unbounded retry chain. A failed command records its exit status and
packages available text/JSON/CSV output; a successful shell command is reported
separately from the scientific results.

## Overrides and local checks

| Variable | Purpose |
|---|---|
| `ROPC_PROJECT_DIR` | Repository location, useful when submitting outside its root |
| `ROPC_BASE` | Alternate storage base |
| `ROPC_DATA_ROOT` | Dataset storage location |
| `ROPC_CONFIG` | Alternate JSON configuration, copied into the snapshot |
| `ROPC_SEED` | Explicit seed for one allocation |
| `ROPC_SPLIT_INDEX` | Registered dataset split |
| `ROPC_DEVICE` | `cuda`, `cpu`, or `auto` |
| `ROPC_VENV_DIR`, `ROPC_PYTHON` | Existing environment or explicit interpreter |
| `ROPC_INSTALL_MISSING` | Set to `0` to disable package installation |

`#SBATCH` directives contain literal log paths. If changing `ROPC_BASE`, create
the new log directory and supply matching `sbatch --output` and `--error`
arguments; shell variables cannot alter directives that Slurm has already read.

Help and dry-run perform no writes or installations. For a deliberate local
launcher check, set `ROPC_ALLOW_LOCAL=1`, `ROPC_SKIP_MODULES=1`,
`ROPC_SKIP_ARCH_CHECK=1`, `ROPC_DEVICE=cpu`, an explicit Python interpreter,
and a writable temporary `ROPC_BASE`. Ordinary local experiments can simply
use `python -m ropc` or the per-dataset `run.py` entry point.

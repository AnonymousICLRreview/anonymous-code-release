# Verification and release scope

This repository was checked with Python 3.12.14, PyTorch 2.5.1+cpu,
NumPy 2.3.5, SciPy 1.17.0, PyG 2.7.0 and OGB 1.3.6 on Linux. The checks below
are execution and numerical validation. They do not reproduce the paper's
reported benchmark scores.

## Automated regression suite

All **75 tests** passed under `python -m unittest discover -s tests -v`.

| Test module | Tests | Main contracts |
|---|---:|---|
| `test_model_control.py` | 16 | Feasible and eroded actions, spectral bounds, sparse reductions, role likelihood, permutation behavior, depth-zero nulls, observer constraints, convex heads, planner majorants, causal memory and replay |
| `test_cli_launchers.py` | 13 | Every dataset entry/config pair, seed reduction, calibration hash round trips, failed-audit status, artifact verification, source snapshots, resume, shell dry runs and array-summary cohort isolation |
| `test_datasets_tasks.py` | 11 | All 24 task shapes and gradients, graph isolation, fixed holdouts, split columns, missing labels, official metric conventions, positional encodings and edge features |
| `test_dataset_edges.py` | 11 | Input identity, custom split fractions, numerical failures, missing-label gradients, graph-averaged link ranks, preprocessing stamps, empty-edge inputs and exact lattice RWSE |
| `test_training_evaluation.py` | 13 | Whole-family metrics, task training, observer gradients, continuation targets, raw branch observations, optimizer/RNG resume, calibration support and complete policy hashes |
| `test_ogb_compat.py` | 3 | Official cache loading and splits, legacy directory handling, stale release rejection and scoped PyTorch cache compatibility |
| `test_transfer.py` | 8 | P5 source requirements, verified artifacts, fixed target depths, test-split scope, unchanged source state, fallback identity and synthetic/official separation |

The transfer tests use small fixtures and mocked completed-source boundaries.
They do not represent an official LRIM-256 inference run. The cluster shell test
executes the real snapshot/resume machinery while replacing the expensive
scientific job with a deterministic test executable; it does not submit Slurm.

## Complete pipeline checks

Every one of the **24 Table IX dataset configurations** completed all stages
under `--preset smoke --synthetic` on CPU: warm-up, bank selection, open-loop
training, planner/amortizer rounds, final refit, calibration, audit, evaluation
and seed reduction. Each configuration used its own output directory.

These small fixtures cover the actual node, graph, regression, multilabel and
link-prediction routes. They retain dataset-specific output dimensions and
kernel banks while reducing graph sizes, training budgets and depth. A completed
audit computation can correctly report an unqualified policy; passing this
integration check does not imply passing a statistical coverage audit.

Two-seed execution and completed-run resume are checked separately, including
seed aggregation and unchanged artifact hashes after resume. The concise
machine-readable release results are in `validation_results.json`.

To repeat the integration matrix from the repository root:

```bash
for config in experiments/*/config.json; do
  dataset="$(basename "$(dirname "$config")")"
  python -m ropc run \
    --dataset "$dataset" --preset smoke --synthetic \
    --device cpu --threads 1 --output "runs/check-$dataset"
done
```

Use a fresh output directory when code or configuration changes. Reusing a
completed directory requires `--resume` and the original scientific identity.

## Actual official-data checks

Two real released datasets were downloaded and passed preprocessing, propagation,
finite task-loss checks and backward-gradient checks:

| Dataset | Inspected graph | Input features | Target shape |
|---|---|---:|---|
| Minesweeper | 10,000 nodes | 7 | `[10000, 1]` |
| LRIM-16, hard setting | 256 nodes, 1,024 directed arcs; official 10,000-graph release loaded | 17, including 16 RWSE channels | `[256, 1]` |

These checks used untrained small models. Their temporary losses and task
scores are not benchmark results and are not distributed as results tables.
The optimized periodic-lattice RWSE was also checked against general sparse
matrix powers on the actual LRIM-16 graph and on sides 3, 4 and 16, including
wraparound, edge-order changes and malformed-topology rejection.

OGB compatibility was verified with offline official-format cache/split fixtures.
The live OGB archive endpoint was unavailable during the check; a successful
fresh OGB network download is not claimed. Other dataset adapters were exercised
through their task/interface fixtures, not through a complete download and
training run of every public benchmark.

## Packaging and operational checks

- Dataset configurations match the registry and separate experiment entry points.
- Every shell script passes `bash -n`; every Python file compiles.
- All 24 shell dry runs resolve their dataset and selected configuration.
- A Python wheel builds from `pyproject.toml` without fetching build dependencies.
- Relative documentation links resolve, including the separate P5 protocol.
- The archive includes source, configurations, tests, documentation and GitHub CI.
  Generated bytecode, build metadata, raw data, checkpoints, logs and temporary
  validation runs are excluded.

The GitHub Actions workflow is provided for Python 3.10 and 3.12. Its local
commands were checked; an actual GitHub-hosted workflow run is not claimed.

## Scientific and deployment boundaries

No live Arrhenius job or GPU run was possible in this environment. The launcher
preserves the supplied account, partition, project paths and module names, and
checks the selected runtime before starting computation. Compatibility with
the server's current CUDA driver, module availability, GPU architecture and
memory budget remains a preflight check on that server.

The repository contains no trained checkpoints or copied paper-result values.
Full-budget training, ten-seed statistical reproduction, every published
comparison arm, measured GPU speedups, parameter-budget matching and the
complete P1–P11 experiment matrix are not established by these checks.

`METHOD.md` and `DATASETS.md` distinguish equations implemented from the supplied
paper from choices not uniquely specified by the supplied code/documents,
including input encoders, the large-graph partitioner and regression uncertainty.
They also record the difference between official binary LRIM inputs and the
supplement's description of continuous features. These boundaries remain visible
so subsequent experimental results can be assessed correctly.

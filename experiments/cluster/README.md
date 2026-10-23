# CLUSTER

This experiment uses the shared ROPC implementation and this directory's
`config.json`. The loader and task contract are registered in `ropc/registry.py`.

| Setting | Value |
|---|---|
| Task | `node_classification` |
| Metric | `balanced_accuracy` (max) |
| Deployed depth | 16 |
| Training ceiling | 64 |
| Operator bank | `P0,Pg,Pc,Px` |
| Transport floor | 2 |
| Training batch | 128 graphs per task batch |

## Python

```bash
python experiments/cluster/run.py --preset smoke --synthetic \
  --device cpu --output runs/cluster_smoke

python experiments/cluster/run.py --preset pilot \
  --data-root data --output runs/cluster_pilot
```

The synthetic smoke run validates this task's execution path. Benchmark
measurements require the actual dataset and its registered splits. `full` uses
the Table IX depths/bank/floor and full training budgets; `pilot` reduces the
training budget. Add `--resume` with the same output and settings to resume.

## Arrhenius

From the repository root:

```bash
bash experiments/cluster/run.sh init
bash experiments/cluster/run.sh dry-run full
sbatch experiments/cluster/run.sh preflight
sbatch experiments/cluster/run.sh smoke
sbatch experiments/cluster/run.sh pilot
sbatch --array=0-9%2 experiments/cluster/run.sh full
```

See [the Arrhenius guide](../../docs/ARRHENIUS.md) for paths, environment overrides,
source snapshots, and suite resumption. Each dataset and array seed has its own
suite and log files.

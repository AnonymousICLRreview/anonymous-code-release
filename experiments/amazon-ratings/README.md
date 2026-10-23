# Amazon-ratings

This experiment uses the shared ROPC implementation and this directory's
`config.json`. The loader and task contract are registered in `ropc/registry.py`.

| Setting | Value |
|---|---|
| Task | `node_classification` |
| Metric | `accuracy` (max) |
| Deployed depth | 16 |
| Training ceiling | 64 |
| Operator bank | `P0,Pg,Px,Pc` |
| Transport floor | 2 |
| Training batch | full graph |

## Python

```bash
python experiments/amazon-ratings/run.py --preset smoke --synthetic \
  --device cpu --output runs/amazon-ratings_smoke

python experiments/amazon-ratings/run.py --preset pilot \
  --data-root data --output runs/amazon-ratings_pilot
```

The synthetic smoke run validates this task's execution path. Benchmark
measurements require the actual dataset and its registered splits. `full` uses
the Table IX depths/bank/floor and full training budgets; `pilot` reduces the
training budget. Add `--resume` with the same output and settings to resume.

## Arrhenius

From the repository root:

```bash
bash experiments/amazon-ratings/run.sh init
bash experiments/amazon-ratings/run.sh dry-run full
sbatch experiments/amazon-ratings/run.sh preflight
sbatch experiments/amazon-ratings/run.sh smoke
sbatch experiments/amazon-ratings/run.sh pilot
sbatch --array=0-9%2 experiments/amazon-ratings/run.sh full
```

See [the Arrhenius guide](../../docs/ARRHENIUS.md) for paths, environment overrides,
source snapshots, and suite resumption. Each dataset and array seed has its own
suite and log files.

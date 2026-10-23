# LRIM-64

This experiment uses the shared ROPC implementation and this directory's
`config.json`. The loader and task contract are registered in `ropc/registry.py`.

| Setting | Value |
|---|---|
| Task | `node_regression` |
| Metric | `log10_mse` (min) |
| Deployed depth | 96 |
| Training ceiling | 128 |
| Operator bank | `P0,Pg,Px,Pc` |
| Transport floor | 8 |
| Training batch | 1 graphs per task batch |

## Python

```bash
python experiments/lrim-64/run.py --preset smoke --synthetic \
  --device cpu --output runs/lrim-64_smoke

python experiments/lrim-64/run.py --preset pilot \
  --data-root data --output runs/lrim-64_pilot
```

The synthetic smoke run validates this task's execution path. Benchmark
measurements require the actual dataset and its registered splits. `full` uses
the Table IX depths/bank/floor and full training budgets; `pilot` reduces the
training budget. Add `--resume` with the same output and settings to resume.

## Arrhenius

From the repository root:

```bash
bash experiments/lrim-64/run.sh init
bash experiments/lrim-64/run.sh dry-run full
sbatch experiments/lrim-64/run.sh preflight
sbatch experiments/lrim-64/run.sh smoke
sbatch experiments/lrim-64/run.sh pilot
sbatch --array=0-9%2 experiments/lrim-64/run.sh full
```

See [the Arrhenius guide](../../docs/ARRHENIUS.md) for paths, environment overrides,
source snapshots, and suite resumption. Each dataset and array seed has its own
suite and log files.

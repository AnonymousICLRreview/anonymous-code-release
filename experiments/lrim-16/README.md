# LRIM-16

This experiment uses the shared ROPC implementation and this directory's
`config.json`. The loader and task contract are registered in `ropc/registry.py`.

| Setting | Value |
|---|---|
| Task | `node_regression` |
| Metric | `log10_mse` (min) |
| Deployed depth | 32 |
| Training ceiling | 32 |
| Operator bank | `P0,Pg,Px,Pc` |
| Transport floor | 8 |
| Training batch | 32 graphs per task batch |

## Python

```bash
python experiments/lrim-16/run.py --preset smoke --synthetic \
  --device cpu --output runs/lrim-16_smoke

python experiments/lrim-16/run.py --preset pilot \
  --data-root data --output runs/lrim-16_pilot
```

The synthetic smoke run validates this task's execution path. Benchmark
measurements require the actual dataset and its registered splits. `full` uses
the Table IX depths/bank/floor and full training budgets; `pilot` reduces the
training budget. Add `--resume` with the same output and settings to resume.

## Arrhenius

From the repository root:

```bash
bash experiments/lrim-16/run.sh init
bash experiments/lrim-16/run.sh dry-run full
sbatch experiments/lrim-16/run.sh preflight
sbatch experiments/lrim-16/run.sh smoke
sbatch experiments/lrim-16/run.sh pilot
sbatch --array=0-9%2 experiments/lrim-16/run.sh full
```

See [the Arrhenius guide](../../docs/ARRHENIUS.md) for paths, environment overrides,
source snapshots, and suite resumption. Each dataset and array seed has its own
suite and log files.

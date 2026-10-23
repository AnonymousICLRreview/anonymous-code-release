# MalNet-Tiny

This experiment uses the shared ROPC implementation and this directory's
`config.json`. The loader and task contract are registered in `ropc/registry.py`.

| Setting | Value |
|---|---|
| Task | `graph_classification` |
| Metric | `accuracy` (max) |
| Deployed depth | 32 |
| Training ceiling | 64 |
| Operator bank | `P0,Pr0,Pg,Pc` |
| Transport floor | 4 |
| Training batch | 16 graphs per task batch |

## Python

```bash
python experiments/malnet-tiny/run.py --preset smoke --synthetic \
  --device cpu --output runs/malnet-tiny_smoke

python experiments/malnet-tiny/run.py --preset pilot \
  --data-root data --output runs/malnet-tiny_pilot
```

The synthetic smoke run validates this task's execution path. Benchmark
measurements require the actual dataset and its registered splits. `full` uses
the Table IX depths/bank/floor and full training budgets; `pilot` reduces the
training budget. Add `--resume` with the same output and settings to resume.

## Arrhenius

From the repository root:

```bash
bash experiments/malnet-tiny/run.sh init
bash experiments/malnet-tiny/run.sh dry-run full
sbatch experiments/malnet-tiny/run.sh preflight
sbatch experiments/malnet-tiny/run.sh smoke
sbatch experiments/malnet-tiny/run.sh pilot
sbatch --array=0-9%2 experiments/malnet-tiny/run.sh full
```

See [the Arrhenius guide](../../docs/ARRHENIUS.md) for paths, environment overrides,
source snapshots, and suite resumption. Each dataset and array seed has its own
suite and log files.

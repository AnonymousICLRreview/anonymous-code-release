# COCO-SP

This experiment uses the shared ROPC implementation and this directory's
`config.json`. The loader and task contract are registered in `ropc/registry.py`.

| Setting | Value |
|---|---|
| Task | `node_classification` |
| Metric | `macro_f1` (max) |
| Deployed depth | 64 |
| Training ceiling | 64 |
| Operator bank | `P0,Pg,Pf,Pc` |
| Transport floor | 4 |
| Training batch | 32 graphs per task batch |

## Python

```bash
python experiments/coco-sp/run.py --preset smoke --synthetic \
  --device cpu --output runs/coco-sp_smoke

python experiments/coco-sp/run.py --preset pilot \
  --data-root data --output runs/coco-sp_pilot
```

The synthetic smoke run validates this task's execution path. Benchmark
measurements require the actual dataset and its registered splits. `full` uses
the Table IX depths/bank/floor and full training budgets; `pilot` reduces the
training budget. Add `--resume` with the same output and settings to resume.

## Arrhenius

From the repository root:

```bash
bash experiments/coco-sp/run.sh init
bash experiments/coco-sp/run.sh dry-run full
sbatch experiments/coco-sp/run.sh preflight
sbatch experiments/coco-sp/run.sh smoke
sbatch experiments/coco-sp/run.sh pilot
sbatch --array=0-9%2 experiments/coco-sp/run.sh full
```

See [the Arrhenius guide](../../docs/ARRHENIUS.md) for paths, environment overrides,
source snapshots, and suite resumption. Each dataset and array seed has its own
suite and log files.

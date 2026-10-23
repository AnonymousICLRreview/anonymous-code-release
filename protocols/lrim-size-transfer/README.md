# P5: LRIM size transfer

P5 trains a source model on **LRIM-16 with random task depth `L ≤ 64`** and deploys the frozen policy on larger lattices at depth twice their side length. The supplementary protocol notes in Appendix F and Table LIV(a) specify this experiment. It is separate from the primary LRIM-16 experiment, whose Table IX training ceiling is 32.

The configuration in this directory changes that training ceiling to 64 while retaining `L_star=32`, the `P0,Pg,Px,Pc` kernel bank, and transport floor `r_min=8`. The existing official LRIM loader uses the hard setting, `σ=0.6`, and the released train/validation/test split.

## Train and freeze the source

From the repository root, with the project environment activated:

```bash
python -m ropc run \
  --dataset LRIM-16 \
  --preset full \
  --config protocols/lrim-size-transfer/train_config.json \
  --data-root ./data \
  --output ./runs/lrim-size-transfer-source \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --device cuda
```

Use `--resume` with the same arguments to continue an interrupted run. Transfer requires verified completed refit, calibration, and independent audit tasks for the selected seed. A fully completed source suite is preferable; optional source evaluation and reduction may still be pending, and their status is preserved in the transfer report.

The source must have the P5 configuration above. The primary LRIM-16 run and the `smoke` preset have different depth ceilings and are rejected. An audit failure remains visible as an uncertified source policy; it is never converted into an audit pass by transfer.

### Train the source on Arrhenius

Use the existing LRIM-16 wrapper with the P5 configuration override. From the
repository root:

```bash
bash experiments/lrim-16/run.sh init
ROPC_CONFIG="$PWD/protocols/lrim-size-transfer/train_config.json" \
  bash experiments/lrim-16/run.sh dry-run full
ROPC_CONFIG="$PWD/protocols/lrim-size-transfer/train_config.json" \
  sbatch --export=ALL --array=0-9%2 experiments/lrim-16/run.sh full
```

The wrapper retains the Arrhenius account, storage paths, modules, and resource
settings. Each array task saves its selected configuration and source snapshot
in a separate suite. For transfer, pass that suite's `run/` directory as
`--source-run` and use the same array-task seed. See the
[Arrhenius guide](../../docs/ARRHENIUS.md) for resumption and resource overrides.

## Evaluate larger lattices

```bash
python -m ropc transfer \
  --source-run ./runs/lrim-size-transfer-source \
  --seed 0 \
  --data-root ./data \
  --sizes 32,64,128,256 \
  --output ./runs/lrim-size-transfer/seed0 \
  --device cuda \
  --threads 4
```

Repeat for each trained seed to obtain independent repetition-level records. The command evaluates ROPC using the source run's selected calibrated envelope, including its pre-registered fallback when applicable.

| Target | Nodes per lattice | Evaluation depth |
|---|---:|---:|
| LRIM-32 | 1,024 | 64 |
| LRIM-64 | 4,096 | 128 |
| LRIM-128 | 16,384 | 256 |
| LRIM-256 | 65,536 | 512 |

Every target's complete official test split is evaluated. The metric is `log10(MSE)` pooled over finite test-node targets, with lower values better; the logarithm is not replaced by its absolute value. The reported inference batch size comes from the target dataset specification. Target inference recomputes graph-dependent roles, kernels, observations and controller memory using the frozen model.

The command preserves backbone weights, controller parameters, observation normalization, loss scale, support statistics, calibration thresholds and depth-support settings. It performs no fitting or depth selection on the target. A hash check after every target detects mutation of the source state. The source run's recorded scientific runtime files must match the active implementation.

## Output and interpretation

`<output>/transfer.json` contains:

- Source configuration, source run identity, source policy hash and artifact hashes.
- Source audit status and whether the original or fallback envelope was selected.
- Each target's input fingerprint, complete test-graph count, depth, score and inference batch size.
- Runtime settings, elapsed times and explicit adaptation/frozen-state fields.
- An explicit statement that the source audit guarantee does **not** transfer to a different lattice-size population.

`--synthetic` is accepted only when the source run is also synthetic. Synthetic fixtures validate interfaces and execution; their nominal target names do not make them physical lattices or benchmark measurements. Official and synthetic modes cannot be mixed. The package tests exercise transfer with small fixtures and mocked completed source artifacts; they do not claim reproduction of Table LIV.

This command supplies the ROPC size-transfer path. It does not label the other comparator rows in Table LIV as reproduced results.

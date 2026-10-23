"""Command-line entry points for ROPC datasets, training, and validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(prog="ropc", description="Role-conditioned Operator Predictive Control")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List the supported dataset configurations")
    listing.add_argument("--json", action="store_true", help="Emit machine-readable dataset specifications")
    prepare = commands.add_parser("prepare", help="Download/process a dataset or generate a validation fixture")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--data-root", default="./data")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--split-index", type=int, default=0)
    prepare.add_argument("--synthetic", action="store_true")
    prepare.add_argument("--output", help="Optional directory for dataset.json")
    run = commands.add_parser("run", help="Run resumable scientific stages for one dataset")
    run.add_argument("--dataset", required=True)
    run.add_argument("--preset", choices=("full", "pilot", "smoke"), default="full")
    run.add_argument("--data-root", default="./data")
    run.add_argument("--output", required=True)
    run.add_argument("--seeds", default="0", help="Distinct nonnegative seeds separated by commas")
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--split-index", type=int, default=0)
    run.add_argument("--mode", choices=("joint", "fixed"))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--config", help="JSON Config overrides for this dataset")
    run.add_argument("--stages", help="Comma-separated stages, or all (default)")
    run.add_argument("--synthetic", action="store_true", help="Explicit validation fixture; never benchmark measurements")
    transfer = commands.add_parser("transfer", help="Evaluate a frozen P5 LRIM-16 policy on larger lattices")
    transfer.add_argument("--source-run", required=True, help="Source suite trained with the P5 configuration")
    transfer.add_argument("--seed", type=int, default=0)
    transfer.add_argument("--data-root", default="./data")
    transfer.add_argument("--sizes", default="32,64,128,256", help="Distinct target lattice sides, separated by commas")
    transfer.add_argument("--output", required=True)
    transfer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    transfer.add_argument("--threads", type=int, default=4)
    transfer.add_argument("--synthetic", action="store_true", help="Require an explicitly synthetic source and targets")
    validate = commands.add_parser("validate", help="Check action geometry and sparse forward/backward support")
    validate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    validate.add_argument("--threads", type=int, default=2)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from .registry import DATASETS, dataset_spec
        if args.command == "list":
            from dataclasses import asdict
            specs = list(DATASETS.values())
            if args.json:
                print(json.dumps([asdict(spec) for spec in specs], indent=2))
            else:
                print("Dataset                 Task                 Metric             L*   Lmax  Bank")
                for spec in specs:
                    print(f"{spec.slug:23s} {spec.task_type:20s} {spec.metric:18s} {spec.L_star:3d}  {spec.L_max:4d}  {spec.bank}")
            return 0
        if getattr(args, "split_index", 0) < 0 or getattr(args, "seed", 0) < 0:
            raise ValueError("Seed and split index must be nonnegative")
        from .runner import dataset_identity, run_experiment, validate_runtime, write_json
        if args.command == "validate":
            print(json.dumps(validate_runtime(args.device, args.threads), indent=2))
        elif args.command == "prepare":
            from .datasets import load_dataset
            spec = dataset_spec(args.dataset)
            graph = load_dataset(spec.name, args.data_root, seed=args.seed,
                                 split_index=args.split_index, synthetic=args.synthetic)
            record = dict(dataset=spec.name, task_type=spec.task_type, metric=spec.metric,
                          synthetic=args.synthetic, seed=args.seed, split_index=args.split_index,
                          data_identity=dataset_identity(graph))
            if args.output:
                write_json(Path(args.output)/"dataset.json", record)
            print(json.dumps(record, indent=2))
        elif args.command == "transfer":
            from .transfer import run_transfer
            run_transfer(args)
            print(f"[OUTPUT] {Path(args.output).expanduser().resolve() / 'transfer.json'}")
        else:
            output = run_experiment(args)
            print(f"[OUTPUT] {output}")
        return 0
    except (ValueError, RuntimeError, FloatingPointError, FileNotFoundError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the LRIM-16 experiment using the shared ROPC pipeline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ropc.cli import main

if __name__ == "__main__":
    if any(arg == "--dataset" or arg.startswith("--dataset=") for arg in sys.argv[1:]):
        raise SystemExit("This entry point fixes --dataset=lrim-16; use python -m ropc run for another dataset")
    raise SystemExit(main(["run", "--dataset", "lrim-16", "--config",
                          str(Path(__file__).with_name("config.json")), *sys.argv[1:]]))

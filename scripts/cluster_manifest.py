#!/usr/bin/env python3
"""Source snapshot identity and cluster run settings, using the standard library."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import zipfile


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    tmp.replace(path)


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def event():
    return dict(utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
                architecture=platform.machine(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("create", "resume", "finish", "failure"))
    p.add_argument("--suite", required=True, type=Path)
    p.add_argument("--dataset")
    p.add_argument("--preset", choices=("full", "pilot", "smoke"))
    p.add_argument("--seed", type=int)
    p.add_argument("--config")
    p.add_argument("--data-root")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split-index", type=int, default=0)
    p.add_argument("--code", type=int, default=1)
    args = p.parse_args(argv)
    suite = args.suite.resolve()
    manifest = suite/"cluster_manifest.json"
    if args.action == "create":
        if manifest.exists():
            raise SystemExit("Cluster manifest already exists")
        for field in ("dataset", "preset", "seed", "config", "data_root"):
            if getattr(args, field) is None:
                raise SystemExit(f"Missing --{field.replace('_', '-')}")
        config = (suite/"source"/args.config).resolve()
        if not config.is_relative_to(suite/"source") or not config.is_file():
            raise SystemExit("Configuration must be inside the source snapshot")
        value = dict(project="ROPC", dataset=args.dataset, preset=args.preset, seed=args.seed,
                     config=args.config, data_root=str(Path(args.data_root).resolve()),
                     device=args.device, split_index=args.split_index, synthetic=args.preset == "smoke",
                     source_hashes=source_hashes(suite/"source"), state="created", created=event())
        atomic_json(manifest, value)
        return 0
    value = json.loads(manifest.read_text())
    if value.get("project") != "ROPC":
        raise SystemExit("Not a ROPC cluster suite")
    if args.action == "resume":
        actual = source_hashes(suite/"source")
        if actual != value["source_hashes"]:
            changed = sorted(k for k in actual.keys() | value["source_hashes"].keys()
                             if actual.get(k) != value["source_hashes"].get(k))
            raise SystemExit("Source snapshot changed: "+", ".join(changed))
        for key in ("dataset", "preset", "seed", "config", "data_root", "device", "split_index"):
            text = str(value[key])
            if "\n" in text or "\r" in text:
                raise SystemExit("Malformed multiline manifest field")
            print(text)
        return 0
    value["state"] = "command_completed" if args.action == "finish" else "command_failed"
    value["last_event"] = event()
    if args.action == "failure":
        value["exit_code"] = args.code
    atomic_json(manifest, value)
    archive = suite/"ROPC_RESULTS.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(suite.rglob("*")):
            if not path.is_file() or (suite/"source") in path.parents or path == archive:
                continue
            if path.suffix.lower() in (".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md") and path.stat().st_size <= 20*1024*1024:
                output.write(path, path.relative_to(suite))
    print(f"[REPORT] {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

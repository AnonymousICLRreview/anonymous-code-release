#!/usr/bin/env python3
"""Combine completed, compatible independent seed suites from Slurm arrays."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ropc.runner import object_hash, sample_summary, write_json


def locate_run(path):
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent.parent if path.name == "summary.json" and path.parent.name == "900_REDUCE" else path.parent
    if (path/"run"/"run_spec.json").is_file():
        path = path/"run"
    if not (path/"run_spec.json").is_file() or not (path/"900_REDUCE"/"summary.json").is_file():
        raise ValueError(f"No completed Python run and reduction at {path}")
    return path


def cohort_key(manifest):
    inputs = []
    for record in manifest["datasets"].values():
        data = dict(record.get("manifest", {}))
        if not manifest["synthetic"] and not (data.get("source_files") or data.get("input_sha256")):
            raise ValueError("Cannot verify seed-independent input identity; source_files or input_sha256 is required")
        for key in ("seed", "partition_sha256", "tensor_sha256"):
            data.pop(key, None)
        if manifest["synthetic"]:
            # Fixture generators intentionally use the training seed. Their
            # generator code and parameters are already part of this cohort.
            data.pop("input_sha256", None)
        if "source_files" in data:
            data["source_files"] = sorted(data["source_files"], key=lambda item: (item["name"], item["sha256"]))
        identity = dict(scope=record.get("scope"), data=data)
        if identity not in inputs:
            inputs.append(identity)
    if len(inputs) != 1:
        raise ValueError("Seeds within a suite disagree on their underlying dataset or preprocessing")
    return dict(dataset=manifest["dataset"], dataset_slug=manifest["dataset_slug"],
                preset=manifest["preset"], synthetic=manifest["synthetic"],
                split_index=manifest["split_index"],
                config={k: v for k, v in manifest["config"].items() if k != "seed"},
                python_sources=manifest["python_sources"], input_identity=inputs[0])


def summarize(paths, output):
    records, audits, provenance, seeds_seen = [], [], [], set()
    expected = None
    metric = mode = None
    for given in paths:
        run = locate_run(given)
        manifest = json.loads((run/"run_spec.json").read_text())
        result = json.loads((run/"900_REDUCE"/"summary.json").read_text())
        current = cohort_key(manifest)
        if expected is None:
            expected, metric, mode = current, result["metric"], result["metric_mode"]
        if current != expected or metric != result["metric"] or mode != result["metric_mode"]:
            raise ValueError("Suites differ in source, dataset, task metric, configuration, split, preset, or synthetic status")
        if result["status"] != "complete" or result.get("missing_evaluation_seeds"):
            raise ValueError(f"Reduction is incomplete: {run}")
        seeds = set(map(int, result["requested_seeds"]))
        if seeds != set(map(int, manifest["seeds"])) or len(seeds) != len(result["requested_seeds"]):
            raise ValueError(f"Seed manifest disagrees with its reduction: {run}")
        duplicate = seeds_seen & seeds
        if duplicate:
            raise ValueError(f"Duplicate seeds across suites: {sorted(duplicate)}")
        seeds_seen.update(seeds)
        if result.get("run_spec_sha256") != manifest.get("run_spec_sha256"):
            raise ValueError(f"Reduction does not match the run identity: {run}")
        for row in result["observations"]:
            if int(row["seed"]) not in seeds or not math.isfinite(float(row["value"])):
                raise ValueError(f"Invalid observation in {run}")
            if bool(row["synthetic"]) != bool(expected["synthetic"]):
                raise ValueError("Observation synthetic flag disagrees with the cohort")
            records.append(row)
        audits.extend(result.get("audits", []))
        provenance.append(dict(run=str(run), seeds=sorted(seeds),
                               run_spec_sha256=manifest["run_spec_sha256"], datasets=manifest["datasets"]))
    if expected is None or not records:
        raise ValueError("At least one completed suite with finite observations is required")
    groups = {}
    for row in records:
        depth = row["depth"] if row["protocol"] == "depth" else None
        key = (row["arm"], row["protocol"], depth)
        values = groups.setdefault(key, {})
        if row["seed"] in values:
            raise ValueError(f"Duplicate per-seed measurement: {key}, seed {row['seed']}")
        values[row["seed"]] = float(row["value"])
    summaries = [dict(arm=a, protocol=p, depth=d, seeds=sorted(values),
                      **sample_summary(list(values.values())))
                 for (a, p, d), values in sorted(groups.items(), key=lambda item: str(item[0]))]
    paired = []
    for (arm, protocol, depth), left in groups.items():
        if arm != "ROPC":
            continue
        for base in ("base", "A0", "ROPC-A"):
            right = groups.get((base, protocol, depth), {})
            seeds = sorted(left.keys() & right.keys())
            if not seeds:
                continue
            differences = [left[seed]-right[seed] for seed in seeds]
            direction = -1 if mode == "min" or protocol == "P2_PDD" else 1
            paired.append(dict(comparison=f"ROPC_minus_{base}", protocol=protocol, depth=depth,
                               seeds=seeds, **sample_summary(differences),
                               mean_improvement=direction*statistics.fmean(differences),
                               seed_differences=[dict(seed=s, value=v) for s, v in zip(seeds, differences)]))
    result = dict(dataset=expected["dataset"], synthetic=expected["synthetic"], metric=metric, metric_mode=mode,
                  seeds=sorted(seeds_seen), cohort_sha256=object_hash(expected), cohort=expected,
                  sources=provenance, observations=records, summaries=summaries,
                  paired_differences=paired, audits=audits,
                  statistics=dict(std_ddof=1, interval="two-sided 95% Student t; null for fewer than two seeds",
                                  scope="descriptive variation across distinct training seeds",
                                  independent_graph_generalization_claim=False))
    output = Path(output)
    write_json(output/"summary.json", result)
    for name, rows, columns in (
        ("observations.csv", records, ["seed", "arm", "protocol", "depth", "metric", "metric_mode", "value", "synthetic"]),
        ("summary.csv", summaries, ["arm", "protocol", "depth", "n", "mean", "std", "ci95_lower", "ci95_upper"]),
        ("paired_differences.csv", paired, ["comparison", "protocol", "depth", "n", "mean", "std", "ci95_lower", "ci95_upper", "mean_improvement"]),
    ):
        with (output/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Explicit run/suite paths or quoted glob patterns")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        paths = []
        for pattern in args.paths:
            expanded = sorted(glob.glob(pattern))
            if not expanded:
                raise ValueError(f"No suite matches {pattern!r}")
            paths.extend(expanded)
        result = summarize(paths, args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(dict(dataset=result["dataset"], seeds=result["seeds"], synthetic=result["synthetic"],
                          output=str(args.output.resolve())), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

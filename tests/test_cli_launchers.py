"""Dataset entry points and experiment provenance/seed contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ropc.cli import build_parser
from ropc.registry import DATASETS
from ropc.runner import (make_tasks, object_hash, parse_seeds, resolve_config,
                         run_experiment, sample_summary, stage_reduce, verified_complete, write_json)


ROOT = Path(__file__).resolve().parents[1]


class CliContractTests(unittest.TestCase):
    def test_all_dataset_entries_match_registry(self):
        self.assertEqual(len(DATASETS), 24)
        for slug, spec in DATASETS.items():
            with self.subTest(dataset=slug):
                entry = ROOT/"experiments"/slug
                self.assertEqual(json.loads((entry/"config.json").read_text()), spec.config_overrides())
                self.assertEqual(json.loads((ROOT/"configs"/f"{slug}.json").read_text()), spec.config_overrides())
                self.assertTrue((entry/"run.py").is_file())
                self.assertTrue((entry/"run.sh").is_file())
                self.assertTrue((entry/"README.md").is_file())

    def test_seed_list_rejects_duplicates_and_invalid_values(self):
        self.assertEqual(parse_seeds("0,2,7"), [0, 2, 7])
        for text in ("", "1,1", "-1", "0,", "a"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_seeds(text)

    def test_smoke_enforces_small_budget_and_dataset_bank(self):
        args = build_parser().parse_args(["run", "--dataset", "Peptides-func", "--preset", "smoke",
                                         "--config", str(ROOT/"experiments/peptides-func/config.json"),
                                         "--output", "unused", "--synthetic"])
        cfg, spec = resolve_config(args)
        self.assertEqual(cfg.dataset, spec.name)
        self.assertEqual(cfg.bank, spec.bank)
        self.assertEqual(cfg.L_star, 4)
        self.assertEqual(cfg.L_max, 8)
        self.assertEqual(cfg.d, 16)
        tasks = make_tasks(cfg, [0, 2])
        self.assertEqual(tasks["warmup"], ["warmup_s0", "warmup_s2"])
        self.assertEqual(tasks["reduce"], ["reduce_all"])
        self.assertTrue(all("_s" in name for name in tasks["evaluate"]))

    def test_unknown_config_fields_and_dataset_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"config.json"
            args = build_parser().parse_args(["run", "--dataset", "ogbn-arxiv", "--output", "unused", "--config", str(path)])
            path.write_text(json.dumps(dict(dataset="ZINC")))
            with self.assertRaisesRegex(ValueError, "dataset differs"):
                resolve_config(args)
            path.write_text(json.dumps(dict(not_a_parameter=123)))
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                resolve_config(args)
        args = build_parser().parse_args(["run", "--dataset", "lrim-256", "--output", "unused"])
        with self.assertRaisesRegex(ValueError, "transfer evaluation target"):
            resolve_config(args)

    def test_calibration_hash_matches_the_state_actually_saved(self):
        import torch
        from ropc.control import Envelope
        from ropc.evaluation import PolicyBundle, policy_hash
        from ropc.model import ActionGeometry
        from ropc.runner import stage_calibrate
        from ropc.training import RoundState
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(["run", "--dataset", "ogbn-arxiv", "--output", temporary,
                                             "--preset", "smoke", "--synthetic", "--device", "cpu"])
            cfg, _ = resolve_config(args)
            geom = ActionGeometry(cfg.M, cfg.Kc, cfg.alpha_min, cfg.alpha_max, cfg.r_min)
            rs = RoundState(cfg, geom, torch.device("cpu"))
            bundle = PolicyBundle(variant="ropc", rs=rs, H=cfg.H)
            graph = SimpleNamespace(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]),
                                    task_type="node_classification", metric="accuracy")
            plant = SimpleNamespace(bb=torch.nn.Linear(2, 2), normalizer=torch.nn.Linear(1, 1), loss_bar=1.,
                                    sensor=SimpleNamespace(S=torch.ones(1, 1)), sens_seed=0, graph=graph,
                                    source=None, reference_graph=graph)
            envelope = Envelope((1., 2., float("inf")))
            partition = SimpleNamespace(tune_seeds=[1, 2])
            with patch("ropc.runner.load_bundle", return_value=(plant, graph, None, partition, bundle)), \
                    patch("ropc.evaluation.calibrate_policy", return_value={"envelope": envelope, "info": {}}):
                result = stage_calibrate(cfg, args, root, root, "calibrate_ropc_s0", torch.device("cpu"))
            path = root/"ARTIFACTS"/"seed0"/"policies"/"ropc"
            saved = torch.load(path/"state.pt", map_location="cpu", weights_only=False)
            restored = RoundState(cfg, geom, torch.device("cpu"))
            restored.load_state_dict(saved, torch.zeros(geom.Kc, geom.p))
            env = Envelope.from_dict(json.loads((path/"envelope.json").read_text()))
            reloaded = PolicyBundle(variant="ropc", rs=restored, envelope=env, H=cfg.H)
            self.assertEqual(result["hash"], policy_hash(cfg, reloaded, env, plant.bb, plant=plant))
            self.assertIsNone(json.loads((path/"envelope.json").read_text())["slopes"][-1])

    def test_failed_reserve_audit_stays_explicitly_unqualified(self):
        from ropc.runner import deployment_qualification
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(["run", "--dataset", "ogbn-arxiv", "--output", temporary])
            cfg, _ = resolve_config(args)
            path = root/"ARTIFACTS"/"seed0"/"audit"/"ropc"/"audit_summary.json"
            write_json(path, {"audit": {"policy_pass": False},
                              "fallback": {"audit": {"policy_pass": False}}})
            result = deployment_qualification(cfg, root, "ropc")
            self.assertFalse(result["deployment_audit_pass"])
            self.assertEqual(result["deployment_status"], "uncertified")
            self.assertEqual(result["evaluated_policy"], "fallback")

    def test_completion_marker_rejects_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root/"weights.txt"
            artifact.write_text("measured fixture")
            from ropc.runner import file_hash
            marker = root/"task"
            write_json(marker/"complete.json", dict(status="complete", run_spec_sha256="identity",
                       artifacts={"weights.txt": file_hash(artifact)}))
            self.assertTrue(verified_complete(marker, root, "identity"))
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed or disappeared"):
                verified_complete(marker, root, "identity")

    def test_seed_statistics_are_descriptive_and_handle_one_seed(self):
        one = sample_summary([.5])
        self.assertIsNone(one["std"])
        self.assertIsNone(one["ci95_lower"])
        two = sample_summary([.04, .08])
        self.assertAlmostEqual(two["mean"], .06)
        self.assertAlmostEqual(two["ci95_lower"], .06-12.706204736432095*.02, places=10)

    def test_reduction_reads_every_requested_seed_and_pairs_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = build_parser().parse_args(["run", "--dataset", "ogbn-arxiv", "--output", str(root)])
            cfg, _ = resolve_config(args)
            args._seeds = [0, 2]
            args._run_spec_hash = "fixture"
            for seed, score in [(0, .60), (2, .70)]:
                arms = {}
                for arm, value in [("base", score), ("ROPC", score+.04)]:
                    arms[arm] = {"P1": {"L_selected": 32, "test_at_selected": value},
                                 "P2": {"DA": value, "PDD": .01, "curve": {32: value}}}
                write_json(root/"ARTIFACTS"/f"seed{seed}"/"eval"/"p1p2.json", {"arms": arms})
            result = stage_reduce(cfg, args, root, root/"unused", "reduce_all", "cpu")
            self.assertEqual(result["requested_seeds"], [0, 2])
            self.assertEqual(result["missing_evaluation_seeds"], [])
            summary = next(r for r in result["summaries"] if r["arm"] == "ROPC" and r["protocol"] == "P1")
            self.assertEqual(summary["n"], 2)
            self.assertAlmostEqual(summary["mean"], .69)
            pair = next(r for r in result["paired_differences"] if r["comparison"] == "ROPC_minus_base" and r["protocol"] == "P1")
            self.assertAlmostEqual(pair["mean"], .04)
            self.assertEqual(pair["seeds"], [0, 2])

    def test_array_summary_requires_one_cohort_and_unique_seeds(self):
        from scripts.summarize_suites import summarize
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for seed, score, base in ((2, .8, .7), (5, .9, .75)):
                suite = root/f"suite{seed}"
                run = suite/"run"
                manifest = dict(dataset="ogbn-arxiv", dataset_slug="ogbn-arxiv", preset="smoke",
                                synthetic=True, split_index=0, config={"d": 16, "seed": seed},
                                python_sources={"runner.py": "same source"}, seeds=[seed],
                                datasets={str(seed): {"fingerprint": f"fixture{seed}"}})
                manifest["run_spec_sha256"] = object_hash(manifest)
                write_json(run/"run_spec.json", manifest)
                rows = [dict(seed=seed, arm=arm, protocol="P1", depth=4, metric="accuracy",
                             metric_mode="max", value=value, synthetic=True)
                        for arm, value in (("ROPC", score), ("base", base))]
                write_json(run/"900_REDUCE"/"summary.json", dict(metric="accuracy", metric_mode="max",
                           status="complete", requested_seeds=[seed], missing_evaluation_seeds=[],
                           run_spec_sha256=manifest["run_spec_sha256"], observations=rows, audits=[]))
                paths.append(suite)
            result = summarize(paths, root/"combined")
            self.assertEqual(result["seeds"], [2, 5])
            self.assertTrue(result["synthetic"])
            self.assertAlmostEqual(next(r for r in result["summaries"] if r["arm"] == "ROPC")["mean"], .85)
            self.assertAlmostEqual(result["paired_differences"][0]["mean"], .125)
            self.assertTrue((root/"combined"/"paired_differences.csv").is_file())
            with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
                summarize(paths+[paths[0]], root/"invalid")
            second = paths[1]/"run"/"run_spec.json"
            manifest = json.loads(second.read_text())
            manifest["config"]["d"] = 32
            write_json(second, manifest)
            with self.assertRaisesRegex(ValueError, "Suites differ"):
                summarize(paths, root/"different")

    def test_resume_verifies_source_and_skips_completed_task(self):
        import numpy as np
        calls = []
        graph = SimpleNamespace(x=np.ones((3, 2)), y=np.array([0, 1, 0]),
                                edge_index=np.array([[0, 1], [1, 2]]), source=None)
        def phase(cfg, args, suite, result_dir, task, device):
            calls.append(task)
            return {"fixture": True}
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(["run", "--dataset", "ogbn-arxiv", "--output", temporary,
                                             "--device", "cpu", "--synthetic", "--preset", "smoke"])
            with patch("ropc.runner.STAGES", {"warmup": phase}), patch("ropc.runner.load_graph_for", return_value=graph), \
                    patch("ropc.runner.source_identity", return_value={"runner.py": "unchanged"}):
                run_experiment(args)
                args.resume = True
                run_experiment(args)
                self.assertEqual(calls, ["warmup_s0"])
            with patch("ropc.runner.STAGES", {"warmup": phase}), patch("ropc.runner.load_graph_for", return_value=graph), \
                    patch("ropc.runner.source_identity", return_value={"runner.py": "changed"}):
                with self.assertRaisesRegex(ValueError, "different code"):
                    run_experiment(args)

    def test_help_list_and_wrapper_dry_run_do_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)/"uncreated"
            env = dict(os.environ, ROPC_BASE=str(base), ROPC_PROJECT_DIR=str(ROOT))
            for slug in DATASETS:
                command = ["bash", str(ROOT/"experiments"/slug/"run.sh"), "dry-run", "smoke"]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--dataset {slug}", result.stdout)
                self.assertIn("--synthetic", result.stdout)
            self.assertFalse(base.exists())
            result = subprocess.run([sys.executable, "-m", "ropc", "list", "--json"], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)), 24)

    def test_cluster_snapshot_resume_restores_settings_and_rejects_tampering(self):
        """Run the real shell while replacing only the expensive Python job."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root/"project"
            (project/"ropc").mkdir(parents=True)
            (project/"ropc"/"cli.py").write_text("# deterministic fixture\n")
            shutil.copytree(ROOT/"scripts", project/"scripts", ignore=shutil.ignore_patterns("__pycache__"))
            entry = project/"experiments"/"ogbn-arxiv"
            entry.mkdir(parents=True)
            (entry/"config.json").write_text(json.dumps(DATASETS["ogbn-arxiv"].config_overrides()))
            interpreter = root/"fixture_python"
            interpreter.write_text("#!"+sys.executable+"\n"+r'''
import json, os, pathlib, sys
args=sys.argv[1:]
if args == ['-']:
    sys.stdin.read()
    raise SystemExit(0)
if args[:3] == ['-m','pip','freeze']:
    print('fixture-only==1')
    raise SystemExit(0)
if args[:4] == ['-u','-m','ropc','run']:
    path=pathlib.Path(os.environ['FIXTURE_CALLS'])
    with path.open('a') as stream:
        stream.write(json.dumps({'args':args,'cwd':os.getcwd()})+'\n')
    output=pathlib.Path(args[args.index('--output')+1]); output.mkdir(parents=True,exist_ok=True)
    (output/'fixture.json').write_text(json.dumps({'explicit_test_fixture':True}))
    raise SystemExit(0)
os.execv(sys.executable,[sys.executable,*args])
''')
            interpreter.chmod(0o755)
            call_log = root/"calls.jsonl"
            base = root/"cluster"
            data = root/"original data"
            env = dict(os.environ, ROPC_BASE=str(base), ROPC_PROJECT_DIR=str(project),
                       ROPC_PYTHON=str(interpreter), ROPC_ALLOW_LOCAL="1", ROPC_SKIP_MODULES="1",
                       ROPC_SKIP_ARCH_CHECK="1", ROPC_INSTALL_MISSING="0", ROPC_DEVICE="cpu",
                       ROPC_SEED="7", ROPC_SPLIT_INDEX="2", ROPC_DATA_ROOT=str(data),
                       FIXTURE_CALLS=str(call_log))
            launcher = project/"scripts"/"run_arrhenius.sh"
            result = subprocess.run(["bash", str(launcher), "ogbn-arxiv", "smoke"], env=env,
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            suite = next(p for p in (base/"outputs").iterdir() if p.is_dir() and not p.is_symlink())
            self.assertTrue((suite/"ROPC_RESULTS.zip").is_file())
            self.assertEqual(json.loads((suite/"cluster_manifest.json").read_text())["seed"], 7)
            changed_env = dict(env, ROPC_SEED="99", ROPC_SPLIT_INDEX="0", ROPC_DATA_ROOT=str(root/"different"))
            result = subprocess.run(["bash", str(launcher), "ogbn-arxiv", "resume", str(suite)],
                                    env=changed_env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            calls = [json.loads(line) for line in call_log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            resumed = calls[-1]["args"]
            self.assertIn("--resume", resumed)
            self.assertIn("--synthetic", resumed)
            self.assertEqual(resumed[resumed.index("--seeds")+1], "7")
            self.assertEqual(resumed[resumed.index("--split-index")+1], "2")
            self.assertEqual(resumed[resumed.index("--data-root")+1], str(data))
            self.assertEqual(calls[-1]["cwd"], str(suite/"source"))
            (suite/"source"/"ropc"/"cli.py").write_text("# modified source\n")
            result = subprocess.run(["bash", str(launcher), "ogbn-arxiv", "resume", str(suite)],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Source snapshot changed", result.stderr)
            self.assertEqual(len(call_log.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()

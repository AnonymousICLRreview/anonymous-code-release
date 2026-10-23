"""P5: evaluate a frozen LRIM-16 policy on larger lattices.

The source is trained separately with random task depth at most 64. Target
inference uses depth twice the lattice side and never fits target statistics.
"""
from __future__ import annotations

import copy
import json
import math
import time
from argparse import Namespace
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path


TARGET_SIDES = (32, 64, 128, 256)
CORE_SOURCE_FILES = ("model.py", "control.py", "training.py", "evaluation.py",
                     "datasets.py", "tasks.py", "registry.py", "runner.py")


def _read_json(path):
    with Path(path).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sizes(text):
    values = str(text).split(",")
    try:
        sides = [int(value.strip()) for value in values]
    except ValueError as exc:
        raise ValueError("Transfer sizes must be comma-separated lattice sides") from exc
    if not sides or len(set(sides)) != len(sides) or any(side not in TARGET_SIDES for side in sides):
        raise ValueError("Choose distinct transfer sides from 32,64,128,256")
    return sides


def _source_configuration(suite, seed, synthetic):
    from . import runner
    from .training import Config

    manifest = _read_json(suite/"run_spec.json")
    scientific = {key: value for key, value in manifest.items()
                  if key not in ("created_utc", "run_spec_sha256")}
    if runner.object_hash(scientific) != manifest.get("run_spec_sha256"):
        raise ValueError("The source run manifest has changed or has an invalid identity")
    if manifest.get("dataset") != "LRIM-16":
        raise ValueError("P5 requires a policy trained on LRIM-16")
    if seed < 0 or seed not in manifest.get("seeds", []):
        raise ValueError("The requested seed is absent from the source run")
    if type(manifest.get("synthetic")) is not bool or bool(synthetic) != manifest["synthetic"]:
        raise ValueError("Synthetic and official source/target data cannot be mixed; --synthetic must match the source run")
    raw_config = manifest.get("config", {})
    if not isinstance(raw_config, dict):
        raise ValueError("The source run has no configuration object")
    unknown = set(raw_config)-{field.name for field in fields(Config)}
    if unknown:
        raise ValueError("Unsupported source configuration fields: "+", ".join(sorted(unknown)))
    cfg = Config(**raw_config)
    cfg.seed = int(seed)
    if (cfg.dataset != "LRIM-16" or cfg.L_max != 64 or cfg.L_star != 32 or
            cfg.bank_list != ["P0", "Pg", "Px", "Pc"] or cfg.r_min != 8):
        raise ValueError("P5 requires LRIM-16 with L_max=64, L_star=32, bank=P0,Pg,Px,Pc and r_min=8; "
                         "train with protocols/lrim-size-transfer/train_config.json and --preset full")
    # A parameter hash alone does not identify the algorithm that interprets it.
    # Require the scientific runtime files recorded by the source run. Adding
    # this transfer command or changing CLI documentation does not invalidate it.
    runtime = runner.source_identity()
    source_files = manifest.get("python_sources", {})
    for name in CORE_SOURCE_FILES:
        if source_files.get(name) != runtime.get(name):
            raise ValueError(f"Source/runtime scientific code identity differs: {name}")
    tasks = runner.make_tasks(cfg, manifest["seeds"])
    completed = {}
    for stage, task in (("refit", f"refit_s{seed}"),
                        ("calibrate", f"calibrate_ropc_s{seed}"),
                        ("audit", f"audit_ropc_s{seed}")):
        try:
            ordinal = tasks[stage].index(task)
        except ValueError as exc:
            raise ValueError(f"The source run does not include {task}") from exc
        directory = runner.task_directory(suite, stage, ordinal, task)
        if not runner.verified_complete(directory, suite, manifest["run_spec_sha256"]):
            raise ValueError(f"Transfer requires a verified completed source task: {task}")
        completed[stage] = str(directory.relative_to(suite))
    status = _read_json(suite/"status.json") if (suite/"status.json").exists() else {"status": "partial"}
    if status.get("run_spec_sha256", manifest["run_spec_sha256"]) != manifest["run_spec_sha256"]:
        raise ValueError("Source completion status refers to another run")
    return cfg, manifest, completed, status.get("status", "partial"), runtime


def _freeze_modules(plant, bundle):
    modules = [plant.bb, bundle.rs.model, bundle.rs.value, bundle.rs.stage,
               bundle.rs.base, bundle.rs.student_tw, bundle.amortized]
    modules.extend(bundle.heads_ref or ())
    for module in modules:
        if module is not None:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def run_transfer(args):
    """Run P5 and atomically write ``<args.output>/transfer.json``.

    Required argument fields: source_run, seed, data_root, sizes, output, device,
    threads and synthetic. Returns the complete machine-readable result dict.
    Verified refit/calibration/audit tasks suffice; optional source evaluation
    and reduction may still be incomplete, and their status is recorded.
    """
    import torch
    from . import runner
    from .datasets import load_dataset
    from .evaluation import build_policy_ctrl, policy_hash
    from .model import UnionSupport
    from .registry import dataset_spec
    from .training import Plant, evaluate_depths, make_partition

    suite = Path(args.source_run).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    seed = int(args.seed)
    if int(args.threads) < 1:
        raise ValueError("Transfer threads must be a positive integer")
    sides = _sizes(getattr(args, "sizes", "32,64,128,256"))
    synthetic = bool(getattr(args, "synthetic", False))
    cfg, manifest, completed, source_status, runtime_sources = _source_configuration(suite, seed, synthetic)
    device = runner.torch_setup(args)
    runner.seed_phase(seed, int(args.threads))
    # Reconstruct the same loader Namespace as the source run. In particular,
    # the source split, value fraction and synthetic flag stay source-fixed.
    load_args = Namespace(data_root=str(Path(args.data_root).expanduser().resolve()),
                          dataset=cfg.dataset, synthetic=synthetic, split_index=cfg.split_index,
                          device=str(device), threads=int(args.threads))
    source_plant, source_graph, _, _, bundle = runner.load_bundle(
        cfg, load_args, suite, device, "ropc", calibrated=True)
    envelope, envelope_file = runner.deployed_envelope(cfg, suite, "ropc")
    bundle.envelope = envelope
    runner.verify_deployed_policy(cfg, suite, "ropc", bundle, source_plant, envelope)
    qualification = runner.deployment_qualification(cfg, suite, "ropc")
    source_identity = runner.dataset_identity(source_graph)
    recorded_identity = manifest.get("datasets", {}).get(str(seed))
    if recorded_identity is None or recorded_identity.get("fingerprint") != source_identity.get("fingerprint"):
        raise ValueError("The loaded LRIM-16 input does not match the source run fingerprint")
    pdir = runner.artifacts_dir(suite, cfg)/"policies"/"ropc"
    audit_path = runner.artifacts_dir(suite,cfg)/"audit"/"ropc"/"audit_summary.json"
    audit = _read_json(audit_path)
    initial_hash = (pdir/"hash.txt").read_text().strip()
    if audit.get("hash") != initial_hash:
        raise ValueError("The source audit does not identify the calibrated policy")
    fallback = envelope_file == "envelope_fallback.json"
    if fallback != ("fallback" in audit):
        raise ValueError("The selected source envelope and audit fallback record disagree")
    selected = audit["fallback"] if fallback else audit
    marker = pdir/("fallback_hash.txt" if fallback else "hash.txt")
    source_hash = policy_hash(cfg,bundle,envelope,source_plant.bb,source_plant)
    if source_hash != marker.read_text().strip() or selected.get("hash") != source_hash:
        raise ValueError("The selected frozen source policy does not match its audit hash")
    _freeze_modules(source_plant,bundle)
    frozen_hash = policy_hash(cfg,bundle,envelope,source_plant.bb,source_plant)
    source_config = asdict(cfg)
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for side in sides:
            target_name = f"LRIM-{side}"
            spec = dataset_spec(target_name)
            graph = load_dataset(target_name,load_args.data_root,seed=seed,
                                 split_index=cfg.split_index,synthetic=synthetic,value_slice=cfg.value_slice)
            identity = runner.dataset_identity(graph)
            if graph.x.shape[1] != source_plant.bb.F_in or graph.num_classes != source_plant.bb.n_classes:
                raise ValueError(f"{target_name} feature/readout dimensions differ from the frozen LRIM-16 model")
            if graph.task_type != "node_regression" or graph.metric != "log10_mse":
                raise ValueError(f"{target_name} does not expose the required LRIM regression task")
            source = getattr(graph,"source",None)
            if source is None or not source.is_family or not source.partitions.get("test"):
                raise ValueError(f"{target_name} has no complete graph-family test partition")
            if bool(getattr(source,"synthetic",False)) != synthetic:
                raise ValueError(f"{target_name} data mode differs from the source run")
            # Batching is an execution setting; all learned parameters and
            # controller depth/support settings remain the LRIM-16 source's.
            target_cfg = copy.deepcopy(cfg)
            target_cfg.batch_size = spec.batch_size
            graph = graph.to(device)
            plant = Plant(target_cfg,source_plant.bb,graph,
                          UnionSupport(graph.edge_index,graph.num_nodes),
                          make_partition(graph,target_cfg),device,source_plant.sens_seed)
            plant.sensor = source_plant.sensor
            plant.normalizer = source_plant.normalizer
            plant.loss_bar = source_plant.loss_bar
            depth = 2*side
            before = time.perf_counter()
            metrics = evaluate_depths(plant,lambda:build_policy_ctrl(cfg,plant,bundle,envelope),
                                      [depth],split="test",sens_seed=source_plant.sens_seed)
            score = float(metrics[depth])
            if not math.isfinite(score):
                raise ValueError(f"{target_name} test metric is non-finite")
            if policy_hash(cfg,bundle,envelope,source_plant.bb,source_plant) != frozen_hash:
                raise RuntimeError("Frozen source state changed during transfer evaluation")
            rows.append({"dataset":target_name,"lattice_side":side,"nominal_nodes_per_graph":side*side,
                         "depth":depth,"metric":"log10_mse","metric_mode":"min","value":score,
                         "metric_definition":"log10 of MSE pooled over all finite test-node targets",
                         "split":"test","n_test_graphs":len(source.partitions["test"]),
                         "inference_batch_size":spec.batch_size,"input_identity":identity,
                         "target_fingerprint":identity["fingerprint"],"seconds":time.perf_counter()-before,
                         "validation_scope":"synthetic interface fixture" if synthetic else "all official held-out test graphs",
                         "synthetic":synthetic})
            del plant, graph
    # Include exact source artifact bytes as well as the semantic policy hash.
    paths = [suite/"run_spec.json",runner.artifacts_dir(suite,cfg)/"refit"/"backbone_frozen.pt",
             runner.artifacts_dir(suite,cfg)/"openloop"/"normalizer.pt",pdir/"state.pt",
             pdir/"envelope.json",pdir/"hash.txt",audit_path]
    if fallback:
        paths.extend([pdir/"envelope_fallback.json",pdir/"fallback_hash.txt"])
    artifacts = {}
    completion_hashes = {}
    for stage, relative in completed.items():
        marker_path = suite/relative/"complete.json"
        artifacts.update(_read_json(marker_path)["artifacts"])
        completion_hashes[stage] = runner.file_hash(marker_path)
    for path in paths:
        relative = str(path.relative_to(suite))
        digest = runner.file_hash(path)
        if relative in artifacts and artifacts[relative] != digest:
            raise RuntimeError(f"Source artifact changed during transfer: {relative}")
        artifacts[relative] = digest
    result = {"protocol":"P5_LRIM_size_transfer","status":"complete","seed":seed,
              "created_utc":datetime.now(timezone.utc).isoformat(),"synthetic":synthetic,
              "validation_scope":"synthetic execution check; no benchmark reproduction" if synthetic else
                                 "frozen LRIM-16 policy evaluated on complete official target test splits",
              "source_run":str(suite),"source_run_status":source_status,
              "source_run_spec_sha256":manifest["run_spec_sha256"],"source_policy_sha256":source_hash,
              "source_config":source_config,"source_input_identity":source_identity,
              "source_artifact_sha256":artifacts,"verified_source_tasks":completed,
              "source_completion_marker_sha256":completion_hashes,
              "source_qualification":qualification,"selected_source_envelope":envelope_file,
              "source_audit_guarantee_transfers":False,"target_policy_audited":False,
              "qualification_note":"The LRIM-16 audit qualification concerns its source population; it provides no transferred target-size guarantee.",
              "adaptation":{"parameter_updates":False,"normalizer_refit":False,"loss_scale_refit":False,
                            "controller_refit":False,"recalibration":False,"test_depth_selection":False},
              "frozen_state_unchanged":True,"depth_rule":"2 * lattice_side",
              "runtime":{"device":str(device),"threads":int(args.threads),"torch":torch.__version__,
                         "python_source_sha256":runtime_sources},
              "results":rows,"seconds":time.perf_counter()-started}
    runner.write_json(output/"transfer.json",result)
    return result

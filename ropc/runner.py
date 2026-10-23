"""Scientific stages and provenance-checked experiment orchestration for ROPC."""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import sys
import time
import traceback
from typing import Dict, List, Optional

import numpy as np

STAGE_DIRECTORIES = {
    "warmup": "010_WARMUP", "bank": "020_BANK", "openloop": "030_OPENLOOP",
    "rounds": "040_ROUNDS", "refit": "050_REFIT", "calibrate": "060_CALIBRATE",
    "audit": "070_AUDIT", "evaluate": "080_EVALUATE", "reduce": "900_REDUCE",
}


def json_safe(value):
    """Strict JSON records use null for unavailable/nonfinite diagnostics."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def object_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parent
    return {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}


def seed_phase(seed, threads):
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(threads)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def torch_setup(args):
    import torch
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose --device cpu for a CPU run")
    torch.set_num_threads(max(1, int(args.threads)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(requested)


def load_graph_for(args, cfg):
    from .datasets import load_dataset
    key = (cfg.dataset, cfg.seed, cfg.split_index, cfg.value_slice, bool(args.synthetic))
    if getattr(args, "_graph_cache_key", None) != key:
        args._graph_cache = load_dataset(cfg.dataset, args.data_root, seed=cfg.seed,
                                        split_index=cfg.split_index, synthetic=args.synthetic,
                                        value_slice=cfg.value_slice)
        args._graph_cache_key = key
    return args._graph_cache


def build_setup(args, cfg, device, bank=None):
    from .model import UnionSupport
    from .training import make_partition
    graph = load_graph_for(args, cfg).to(device)
    return graph, UnionSupport(graph.edge_index, graph.num_nodes), make_partition(graph, cfg)


def artifacts_dir(suite: Path, cfg) -> Path:
    return suite / "ARTIFACTS" / f"seed{cfg.seed}"


def task_seed(task: str, default: int) -> int:
    if "_s" in task:
        try:
            return int(task.rsplit("_s", 1)[1])
        except ValueError:
            return default
    return default


# ============================================================================================
# stages
# ============================================================================================


def stage_warmup(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Algorithm 3 line 1: warm up the task route with the reference schedule on the candidate bank;
    checkpoints for the bank selection."""
    import torch
    from .training import build_backbone, Plant, make_controller, train_task_route, ensure_dir, save_backbone, set_seed
    set_seed(cfg.seed)
    graph, sup, part = build_setup(args, cfg, device)
    cand = [k for k in cfg.candidate_bank.split(",") if k] if cfg.bank == "auto" else cfg.bank_list
    bb = build_backbone(cfg, graph, cand).to(device)
    bb.geom.to(device)
    plant = Plant(cfg, bb, graph, sup, part, device)
    out_dir = ensure_dir(artifacts_dir(suite, cfg) / "warmup")
    ckpts = [int(v) for v in cfg.warmup_ckpts.split(",") if v]
    res = train_task_route(cfg, plant, lambda: make_controller(cfg, plant, "schedule"), cfg.epochs_warmup, out_dir, "warmup", ckpt_epochs=ckpts)
    summary = {"candidate_bank": cand, "train": res, "n_nodes": graph.num_nodes, "n_arcs": int(graph.edge_index.shape[1]), "nnz_union": sup.nnz}
    write_json(out_dir / "warmup_summary.json", summary)
    return summary


def stage_bank(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Algorithm 3 line 2: E σ̄(C_H) per unit cost over layers ℓ ≥ 1 of the warm-up checkpoints (identifiable
    subspace at ℓ = 0), over 4-kernel subsets of the candidate bank."""
    import itertools
    import torch
    from .model import ActionGeometry, UnionSupport
    from .training import load_backbone, Plant, make_controller, run_pass, response_jacobian, sigma_bar, bank_cost, ensure_dir, depth_t
    graph, sup, part = build_setup(args, cfg, device)
    wdir = artifacts_dir(suite, cfg) / "warmup"
    ckpts = sorted(wdir.glob("warmup_ep*.pt")) or [wdir / "warmup_final.pt"]
    scores = {}
    costs = {}
    torch.set_grad_enabled(False)
    for ck in ckpts:
        bb, _ = load_backbone(cfg, graph, ck, device)
        bb.eval()
        plant = Plant(cfg, bb, graph, sup, part, device)
        # fit a provisional normaliser on this checkpoint's reference trajectory
        plant.prepare(grad=False)
        ctrl = make_controller(cfg, plant, "schedule")
        ctrl.reset(1, plant.ctx)
        H = plant.H0
        raws, Hs = [], []
        L = min(cfg.L_max, 16)
        for ell in range(L + 1):
            raws.append(plant.observe(H)[0])
            if ell % max(1, L // 4) == 0:
                Hs.append((ell, H))
            A, rec = ctrl.step(ell, None, plant.ctx)
            H = plant.layer(H, A)
            ctrl.commit(ell, A, rec, plant.ctx)
        plant.normalizer.fit(torch.stack(raws))
        full_bank = bb.bank
        sparse_candidates = [k for k in full_bank]
        for subset in itertools.combinations(full_bank, min(4, len(full_bank))):
            subset = list(subset)
            key = ",".join(subset)
            # restrict the tangent chart to the subset's transport coordinates: columns of C_H for the subset
            geom_sub = ActionGeometry(len(subset), cfg.Kc, cfg.alpha_min, cfg.alpha_max, cfg.r_min).to(device)
            vals = []
            for (ell, Hl) in Hs:
                Aref = bb.ref_schedule(depth_t(ell, cfg.ell0).to(device))[0]               # (Kc, M_full+2)
                # map the reference action onto the subset (renormalise transport mass over the subset)
                idx = [2 + full_bank.index(k) for k in subset]
                At = torch.cat([Aref[:, :2], Aref[:, idx]], 1)
                At[:, 2:] = At[:, 2:] / At[:, 2:].sum(1, keepdim=True).clamp_min(1e-9) * (1 - At[:, 0:1] - At[:, 1:2])
                # finite differences in the subset chart require executing subset actions on the full block: embed
                def embed(a_sub):
                    a_full = torch.zeros(a_sub.shape[0], 2 + len(full_bank), device=device)
                    a_full[:, :2] = a_sub[:, :2]
                    for j, k in enumerate(subset):
                        a_full[:, 2 + full_bank.index(k)] = a_sub[:, 2 + j]
                    return a_full
                u0 = geom_sub.to_tangent(At.unsqueeze(0))
                cols = []
                for j in range(geom_sub.du):
                    e = torch.zeros_like(u0); e[:, j] = 1e-4
                    zp = plant.normalizer(plant.observe(plant.layer(Hl, embed(geom_sub.from_tangent(u0 + e)[0]).unsqueeze(0))))
                    zm = plant.normalizer(plant.observe(plant.layer(Hl, embed(geom_sub.from_tangent(u0 - e)[0]).unsqueeze(0))))
                    cols.append((zp - zm)[0] / 2e-4)
                C = torch.stack(cols, 1)
                if ell == 0:
                    keep = [j for j in range(geom_sub.du) if j % (len(subset) + 1) != 0]
                    vals.append(sigma_bar(C, torch.tensor(keep)))
                else:
                    vals.append(sigma_bar(C))
            scores.setdefault(key, []).append(float(sum(vals) / len(vals)))
            costs[key] = bank_cost(subset, graph.num_nodes, sup.nnz, cfg.Kf, cfg.d)
    torch.set_grad_enabled(True)
    table = {k: {"sigma_bar_mean": float(sum(v) / len(v)), "per_checkpoint": v, "cost": costs[k], "score_per_cost": float(sum(v) / len(v)) / costs[k]} for k, v in scores.items()}
    best = max(table, key=lambda k: table[k]["score_per_cost"])
    out = {"selected": best.split(","), "declared": cfg.bank_list, "used": best.split(",") if cfg.bank == "auto" else cfg.bank_list, "table": table,
           "checkpoints": [str(c) for c in ckpts]}
    odir = ensure_dir(artifacts_dir(suite, cfg) / "bank")
    write_json(odir / "bank_selection.json", out)
    return out


def resolve_bank(cfg, suite: Path) -> List[str]:
    if cfg.bank != "auto":
        return cfg.bank_list
    sel = json.loads((artifacts_dir(suite, cfg) / "bank" / "bank_selection.json").read_text())
    return sel["used"]


def stage_openloop(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Open-loop backbone A0 (reference schedule) or the free-spline arm A1t on the selected bank;
    the A0 task also freezes the observation normaliser and ℓ̄ (0.99-quantile of the reference task loss)."""
    import torch
    from .model import SplineSchedule
    from .training import build_backbone, Plant, make_controller, train_task_route, ensure_dir, set_seed, perturb_backbone_
    set_seed(cfg.seed)
    graph, sup, part = build_setup(args, cfg, device)
    bank = resolve_bank(cfg, suite)
    cfg.bank = ",".join(bank)
    bb = build_backbone(cfg, graph, bank).to(device)
    bb.geom.to(device)
    plant = Plant(cfg, bb, graph, sup, part, device)
    out_dir = ensure_dir(artifacts_dir(suite, cfg) / "openloop")
    if task.startswith("schedule_A1t"):
        free = SplineSchedule(bb.geom, cfg.free_knots).to(device)
        params = [p for p in bb.parameters() if p.requires_grad] + list(free.parameters())
        res = train_task_route(cfg, plant, lambda: make_controller(cfg, plant, "schedule", schedule=free), cfg.epochs_openloop, out_dir, "A1t", params=params,
                               extra_state_fn=lambda: {k: v.detach().cpu().clone() for k, v in free.state_dict().items()})
        torch.save({"state": {k: v.detach().cpu() for k, v in free.state_dict().items()}, "n_knots": cfg.free_knots}, out_dir / "A1t_schedule.pt")
        return {"arm": "A1t", "train": res, "bank": bank}
    res = train_task_route(cfg, plant, lambda: make_controller(cfg, plant, "schedule"), cfg.epochs_openloop, out_dir, "A0")
    # Fit sensing and value-loss scales on the reserved value partition.
    from .training import load_backbone, fit_observation_normalizer
    bb, _ = load_backbone(cfg, graph, out_dir / "A0_best.pt", device)
    bb.eval()
    plant = Plant(cfg, bb, graph, sup, part, device)
    normalizer_info = fit_observation_normalizer(plant)
    torch.save({"normalizer": plant.normalizer.state_dict(), "loss_bar": plant.loss_bar, "bank": bank},
               out_dir / "normalizer.pt")
    summary = {"arm": "A0", "train": res, "bank": bank, "loss_bar": plant.loss_bar,
               "normalizer": normalizer_info}
    write_json(out_dir / "openloop_summary.json", summary)
    return summary



def load_plant(cfg, args, suite: Path, device, backbone_path: Path):
    import torch
    from .training import load_backbone, Plant
    graph, sup, part = build_setup(args, cfg, device)
    bank = resolve_bank(cfg, suite)
    cfg.bank = ",".join(bank)
    bb, extra = load_backbone(cfg, graph, backbone_path, device)
    bb.eval()
    plant = Plant(cfg, bb, graph, sup, part, device)
    nz = torch.load(artifacts_dir(suite, cfg) / "openloop" / "normalizer.pt", map_location="cpu", weights_only=False)
    plant.normalizer.load_state_dict(nz["normalizer"])
    plant.loss_bar = nz["loss_bar"]
    return plant, graph, sup, part


def stage_rounds(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Planner–amortizer rounds (Algorithm 3 lines 3–9) starting from the open-loop backbone."""
    import torch
    from .training import (RoundState, collect_branches, fit_round_components, make_controller, provisional_envelope, dagger_train,
                            anytime_of_policy, bootstrap_ci, train_task_route, ensure_dir, save_backbone, set_seed, planner_config, RotatingDesign)
    set_seed(cfg.seed)
    plant, graph, sup, part = load_plant(cfg, args, suite, device, artifacts_dir(suite, cfg) / "openloop" / "A0_best.pt")
    bb = plant.bb
    geom = bb.geom
    rdir = ensure_dir(artifacts_dir(suite, cfg) / "rounds")
    rs = RoundState(cfg, geom, device)
    ref_raw = bb.ref_schedule.raw[0].detach().clone()
    design = RotatingDesign(geom.du, cfg.Kb)
    summary = {"rounds": [], "mode": cfg.mode}
    J_prev_planner = None
    J_prev_samples = None
    stop_k = None
    for k in range(1, cfg.K_rounds + 1):
        kdir = ensure_dir(rdir / f"round{k}")
        if (kdir / "state.pt").exists() and (kdir / "summary.json").exists():       # resume
            st = torch.load(kdir / "state.pt", map_location="cpu", weights_only=False)
            rs.load_state_dict(st, ref_raw)
            if (kdir / "backbone.pt").exists():
                bb.load_state_dict(torch.load(kdir / "backbone.pt", map_location="cpu", weights_only=False)["state"])
            summary["rounds"].append(json.loads((kdir / "summary.json").read_text()))
            previous = summary["rounds"][-1]
            J_prev_planner = previous.get("J_planner_mean")
            J_prev_samples = previous.get("J_planner")
            if previous.get("stop_after_round", False):
                stop_k = k
                break
            if k < cfg.K_rounds:
                next_state = torch.load(kdir / "next_base.pt", map_location="cpu", weights_only=False)
                rs.base = rs.new_policy(ref_raw)
                rs.base.load_state_dict(next_state)
            continue
        t0 = time.time()
        base_k = rs.base                                                              # None in round 1 (β₁ = reference)
        make_base = lambda: make_controller(cfg, plant, "base", rs.model, rs.value, rs.stage, rs.health, rs.support, rs.envelope, rs.scorer, base=base_k, x_max=rs.x_max)
        plant.bb.eval()
        bd = collect_branches(cfg, plant, make_base, cfg.branch_traj, cfg.states_per_traj, cfg.T_cap, seed=cfg.seed * 100 + k)
        torch.save({"branches": bd.__dict__}, kdir / "branches.pt")
        comp = fit_round_components(cfg, plant, rs, bd, device, design)
        rs.envelope = provisional_envelope(cfg, rs, bd, plant.normalizer, device, planner_config(cfg))
        make_planner = lambda: make_controller(cfg, plant, "ropc", rs.model, rs.value, rs.stage, rs.health, rs.support, rs.envelope, rs.scorer,
                                               base=base_k, planner_over_reference=(base_k is None), x_max=rs.x_max)
        # joint mode: retrain the task route with π_k in the loop
        joint_res = None
        if cfg.mode == "joint" and cfg.epochs_round > 0:
            joint_res = train_task_route(cfg, plant, make_planner, cfg.epochs_round, kdir, f"joint_r{k}", train_mix=True)
            bb.load_state_dict(torch.load(kdir / f"joint_r{k}_best.pt", map_location="cpu", weights_only=False)["state"])
            bb.eval()
        # amortize: β_{k+1} ← DAgger(π_k)
        student = rs.new_policy(ref_raw)
        dag = dagger_train(cfg, plant, make_planner, student, cfg.N_DA, cfg.dagger_traj, seed=cfg.seed * 7 + k, label="executed", epochs=cfg.epochs_dagger)
        # round accounting on tuning samples: J(β_k), J(π_k), J(β_{k+1})
        tune = part.tune_seeds[: max(2, min(len(part.tune_seeds), 8))]
        J_beta = anytime_of_policy(cfg, plant, make_base, tune, cfg.T_cap, bb)
        J_pi = anytime_of_policy(cfg, plant, make_planner, tune, cfg.T_cap, bb)
        make_next = lambda: make_controller(cfg, plant, "base", rs.model, rs.value, rs.stage, rs.health, rs.support, rs.envelope, rs.scorer, base=student, x_max=rs.x_max)
        J_next = anytime_of_policy(cfg, plant, make_next, tune, cfg.T_cap, bb)
        G_k = [a - b for a, b in zip(J_beta, J_pi)]
        eps_k = [a - b for a, b in zip(J_next, J_pi)]
        rel = None
        if J_prev_samples is not None:
            if len(J_prev_samples) != len(J_pi):
                raise ValueError("Round stopping requires the same paired tuning samples")
            rel_samples = np.array([(before - after) / max(abs(before), 1e-9)
                                    for before, after in zip(J_prev_samples, J_pi)])
            rng = np.random.default_rng(k)
            boots = np.array([rng.choice(rel_samples, size=rel_samples.size, replace=True).mean() for _ in range(2000)])
            rel = {"mean": float(rel_samples.mean()), "lower95": float(np.quantile(boots, 0.025)), "upper95_two_sided": float(np.quantile(boots, 0.975)),
                   "upper95_one_sided": float(np.quantile(boots, 0.95))}
        # Preserve beta_k with the response model fitted for its continuations.
        # beta_(k+1) is promoted only when another round starts.
        torch.save(rs.state_dict(), kdir / "state.pt")
        torch.save(student.state_dict(), kdir / "next_base.pt")
        if cfg.mode == "joint":
            save_backbone(bb, kdir / "backbone.pt", {"round": k})
        rsum = {"round": k, "components": comp, "dagger": dag, "joint": joint_res, "J_beta": J_beta, "J_planner": J_pi, "J_next_base": J_next,
                "J_planner_mean": float(np.mean(J_pi)), "G_k": bootstrap_ci(G_k, seed=k), "eps_k": bootstrap_ci(eps_k, seed=k + 1),
                "relative_improvement": rel,
                "stop_after_round": bool(rel is not None and k >= 2 and rel["upper95_one_sided"] < cfg.kappa_stop),
                "base_identity": "beta_k", "next_base_file": "next_base.pt",
                "provisional_slope": rs.envelope.slopes, "seconds": time.time() - t0}
        write_json(kdir / "summary.json", rsum)
        summary["rounds"].append(rsum)
        J_prev_planner = float(np.mean(J_pi))
        J_prev_samples = list(J_pi)
        print(f"[ROUND {k}] G_k={rsum['G_k'][0]:.5f} eps_k={rsum['eps_k'][0]:.5f} rel={rel} ({time.time()-t0:.0f}s)", flush=True)
        if rsum["stop_after_round"]:
            stop_k = k
            break
        if k < cfg.K_rounds:
            rs.base = student
    summary["K_selected"] = stop_k or min(cfg.K_rounds, len(summary["rounds"]))
    write_json(rdir / "rounds_summary.json", summary)
    return summary


def stage_refit(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Algorithm 3 line 10: freeze the backbone, refit model/value/amortizer on fresh branches, fit V^{β₁}
    on the frozen backbone, train the gate-matched student ROPC-A_tw, record the backbone hash."""
    import torch
    from .training import (RoundState, collect_branches, fit_round_components, make_controller, provisional_envelope, dagger_train, fit_heads,
                            ensure_dir, save_backbone, set_seed, planner_config, RotatingDesign, ConvexHead)
    set_seed(cfg.seed + 1)
    rdir = artifacts_dir(suite, cfg) / "rounds"
    rsum = json.loads((rdir / "rounds_summary.json").read_text())
    K = int(rsum["K_selected"])
    kdir = rdir / f"round{K}"
    backbone_path = kdir / "backbone.pt" if (kdir / "backbone.pt").exists() else artifacts_dir(suite, cfg) / "openloop" / "A0_best.pt"
    plant, graph, sup, part = load_plant(cfg, args, suite, device, backbone_path)
    bb = plant.bb
    geom = bb.geom
    for p in bb.parameters():
        p.requires_grad_(False)
    odir = ensure_dir(artifacts_dir(suite, cfg) / "refit")
    save_backbone(bb, odir / "backbone_frozen.pt", {"hash": bb.parameter_hash(), "source": str(backbone_path), "K": K})
    rs = RoundState(cfg, geom, device)
    ref_raw = bb.ref_schedule.raw[0].detach().clone()
    rs.load_state_dict(torch.load(kdir / "state.pt", map_location="cpu", weights_only=False), ref_raw)
    design = RotatingDesign(geom.du, cfg.Kb)
    make_base = lambda: make_controller(cfg, plant, "base", rs.model, rs.value, rs.stage, rs.health, rs.support, rs.envelope, rs.scorer, base=rs.base, x_max=rs.x_max)
    t0 = time.time()
    bd = collect_branches(cfg, plant, make_base, cfg.branch_traj, cfg.states_per_traj, cfg.T_cap, seed=cfg.seed * 100 + 99)
    torch.save({"branches": bd.__dict__}, odir / "branches.pt")
    comp = fit_round_components(cfg, plant, rs, bd, device, design)
    rs.envelope = provisional_envelope(cfg, rs, bd, plant.normalizer, device, planner_config(cfg))
    # V^{β₁}: value of the reference base on the frozen backbone (for ROPC-A and Cr1)
    make_ref = lambda: make_controller(cfg, plant, "base", rs.model, rs.value, rs.stage, rs.health, rs.support, rs.envelope, rs.scorer, base=None, x_max=rs.x_max)
    bd_ref = collect_branches(cfg, plant, make_ref, max(1, cfg.branch_traj // 2), cfg.states_per_traj, cfg.T_cap, seed=cfg.seed * 100 + 98, explore=True)
    ctx_dim = 1 + cfg.dx + cfg.wA + cfg.Kc * geom.p
    value_ref = ConvexHead(cfg.dz, cfg.head_width, ctx_dim).to(device)
    stage_ref = ConvexHead(cfg.dz, cfg.head_width, 0).to(device)
    heads_ref_info = fit_heads(cfg, value_ref, stage_ref, bd_ref, plant.normalizer, device, cfg.epochs_heads)
    torch.save({"value": value_ref.state_dict(), "stage": stage_ref.state_dict()}, odir / "heads_ref.pt")
    make_planner = lambda: make_controller(cfg, plant, "ropc", rs.model, rs.value, rs.stage, rs.health, rs.support, rs.envelope, rs.scorer, base=rs.base, x_max=rs.x_max)
    amortized = rs.new_policy(ref_raw)
    dag = dagger_train(cfg, plant, make_planner, amortized, cfg.N_DA, cfg.dagger_traj, seed=cfg.seed * 7 + 50, label="executed", epochs=cfg.epochs_dagger)
    torch.save(amortized.state_dict(), odir / "amortized.pt")
    student_info = None
    if "tw" in cfg.students.split(","):
        rs.student_tw = rs.new_policy(ref_raw)
        student_info = dagger_train(cfg, plant, make_planner, rs.student_tw, cfg.N_DA, cfg.dagger_traj, seed=cfg.seed * 7 + 51, label="proposal", epochs=cfg.epochs_dagger)
    torch.save(rs.state_dict(), odir / "state.pt")
    summary = {"K": K, "backbone_hash": bb.parameter_hash(), "components": comp, "heads_ref": heads_ref_info, "amortized": dag, "student_tw": student_info,
               "provisional_slope": rs.envelope.slopes, "seconds": time.time() - t0}
    write_json(odir / "refit_summary.json", summary)
    return summary


def load_bundle(cfg, args, suite: Path, device, variant: str, calibrated: bool = True):
    """Deployment bundle for a variant: the refit round state (or the variant's calibrated copy), V^{β₁}, β_{K+1}."""
    import torch
    from .training import RoundState, ConvexHead
    from .evaluation import PolicyBundle
    from .control import Envelope
    plant, graph, sup, part = load_plant(cfg, args, suite, device, artifacts_dir(suite, cfg) / "refit" / "backbone_frozen.pt")
    bb = plant.bb
    for p in bb.parameters():
        p.requires_grad_(False)
    geom = bb.geom
    ref_raw = bb.ref_schedule.raw[0].detach().clone()
    rdir = artifacts_dir(suite, cfg) / "refit"
    pdir = artifacts_dir(suite, cfg) / "policies" / variant
    rs = RoundState(cfg, geom, device)
    src = pdir / "state.pt" if (calibrated and (pdir / "state.pt").exists()) else rdir / "state.pt"
    rs.load_state_dict(torch.load(src, map_location="cpu", weights_only=False), ref_raw)
    ctx_dim = 1 + cfg.dx + cfg.wA + cfg.Kc * geom.p
    heads_ref = None
    if (rdir / "heads_ref.pt").exists():
        hr = torch.load(rdir / "heads_ref.pt", map_location="cpu", weights_only=False)
        v = ConvexHead(cfg.dz, cfg.head_width, ctx_dim).to(device); v.load_state_dict(hr["value"])
        s = ConvexHead(cfg.dz, cfg.head_width, 0).to(device); s.load_state_dict(hr["stage"])
        heads_ref = (v, s)
    amortized = None
    if (rdir / "amortized.pt").exists():
        amortized = rs.new_policy(ref_raw)
        amortized.load_state_dict(torch.load(rdir / "amortized.pt", map_location="cpu", weights_only=False))
    env = None
    if calibrated and (pdir / "envelope.json").exists():
        e = json.loads((pdir / "envelope.json").read_text())
        env = Envelope.from_dict(e)
    pb = PolicyBundle(variant=variant, rs=rs, heads_ref=heads_ref, amortized=amortized, envelope=env, H=(1 if variant == "c1" else cfg.H))
    if calibrated:
        from .evaluation import policy_hash
        expected = (pdir / "hash.txt").read_text().strip()
        actual = policy_hash(cfg, pb, env, plant.bb, plant=plant)
        if actual != expected:
            raise ValueError(f"Frozen policy or preprocessing changed for {variant}")
    return plant, graph, sup, part, pb


def stage_calibrate(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Tuning partition: fix every quantity the deployed policy uses, then freeze and hash it."""
    import torch
    from .evaluation import calibrate_policy, policy_hash
    from .training import ensure_dir, set_seed
    variant = task.rsplit("_s", 1)[0].replace("calibrate_", "")
    set_seed(cfg.seed + 2)
    plant, graph, sup, part, pb = load_bundle(cfg, args, suite, device, variant, calibrated=False)
    pb.rs = copy.deepcopy(pb.rs)
    pdir = ensure_dir(artifacts_dir(suite, cfg) / "policies" / variant)
    res = calibrate_policy(cfg, plant, pb, part.tune_seeds, pdir)
    env = res["envelope"]
    # The hashed in-memory state must be identical to the serialized deployment.
    pb.rs.envelope = env
    pb.envelope = env
    torch.save(pb.rs.state_dict(), pdir / "state.pt")
    write_json(pdir / "envelope.json", env.to_dict())
    h = policy_hash(cfg, pb, env, plant.bb, plant=plant)
    (pdir / "hash.txt").write_text(h + "\n")
    write_json(pdir / "calibration.json", res["info"])
    return {"variant": variant, "envelope": env.to_dict(), "hash": h, "info": res["info"]}


def stage_audit(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Audit partition, run once with the frozen policy; the pre-registered fallback (failing bin's slope
    doubled) is audited on the reserve partition."""
    import torch
    from .evaluation import audit_policy, policy_hash
    from .training import ensure_dir, set_seed
    from .control import Envelope
    variant = task.rsplit("_s", 1)[0].replace("audit_", "")
    set_seed(cfg.seed + 3)
    plant, graph, sup, part, pb = load_bundle(cfg, args, suite, device, variant, calibrated=True)
    adir = ensure_dir(artifacts_dir(suite, cfg) / "audit" / variant)
    env = pb.envelope
    main = audit_policy(cfg, plant, pb, env, part.audit_seeds, adir, "audit")
    out = {"variant": variant, "envelope": env.to_dict(), "hash": (artifacts_dir(suite, cfg) / "policies" / variant / "hash.txt").read_text().strip(), "audit": main}
    if not main["policy_pass"]:
        failing = [b for b, v in main["bins"].items() if not v["pass"]]
        env_fb = env
        for b in failing:
            env_fb = env_fb.with_doubled(int(b))
        pb.envelope = env_fb
        fb = audit_policy(cfg, plant, pb, env_fb, part.reserve_seeds, adir, "reserve")
        out["fallback"] = {"failing_bins": failing, "envelope": env_fb.to_dict(), "hash": policy_hash(cfg, pb, env_fb, plant.bb, plant=plant), "audit": fb}
        pdir = artifacts_dir(suite, cfg) / "policies" / variant
        write_json(pdir / "envelope_fallback.json", env_fb.to_dict())
        (pdir / "fallback_hash.txt").write_text(out["fallback"]["hash"] + "\n")
    selected = out.get("fallback", {}).get("audit", main)
    out["deployment_audit_pass"] = bool(selected["policy_pass"])
    out["deployment_status"] = "audit_passed" if selected["policy_pass"] else "uncertified"
    out["evaluated_policy"] = "fallback" if "fallback" in out else "initial"
    out["qualification_scope"] = {
        "sample_unit": selected.get("sample_unit", "unspecified"),
        "decision_scope": selected.get("decision_scope", "unspecified"),
        "family_all_decision_guarantee_eligible": bool(selected.get("family_all_decision_guarantee_eligible", False)),
        "note": selected.get("note", "Empirical diagnostics; inspect the recorded sampling population."),
    }
    write_json(adir / "audit_summary.json", out)
    return out


def deployed_envelope(cfg, suite: Path, variant: str):
    from .control import Envelope
    pdir = artifacts_dir(suite, cfg) / "policies" / variant
    f = pdir / "envelope_fallback.json" if (pdir / "envelope_fallback.json").exists() else pdir / "envelope.json"
    e = json.loads(f.read_text())
    return Envelope.from_dict(e), f.name


def deployment_qualification(cfg, suite: Path, variant: str):
    """Report the tested policy even when the independent reserve audit failed."""
    path = artifacts_dir(suite, cfg) / "audit" / variant / "audit_summary.json"
    record = json.loads(path.read_text())
    selected = record.get("fallback", {}).get("audit", record["audit"])
    passed = bool(selected["policy_pass"])
    return {"variant": variant, "deployment_audit_pass": passed,
            "deployment_status": "audit_passed" if passed else "uncertified",
            "evaluated_policy": "fallback" if "fallback" in record else "initial",
            "qualification_scope": record.get("qualification_scope", selected.get("scope", "empirical audit"))}


def verify_deployed_policy(cfg, suite, variant, bundle, plant, envelope):
    from .evaluation import policy_hash
    pdir = artifacts_dir(suite, cfg) / "policies" / variant
    marker = "fallback_hash.txt" if (pdir / "envelope_fallback.json").exists() else "hash.txt"
    expected = (pdir / marker).read_text().strip()
    if policy_hash(cfg, bundle, envelope, plant.bb, plant=plant) != expected:
        raise ValueError(f"Deployed {variant} policy differs from its audited identity")


def arm_controllers(cfg, args, suite: Path, device, own_backbones: bool = True):
    """All evaluation arms.  Controller arms run on the frozen (refit) backbone; the schedule arms A0 and
    A1t run on their own trained backbones (the manuscript's "each arm's own" protocol), and A0_on_R is
    the reference schedule executed open loop on the frozen ROPC backbone.  Returns
    (plant, graph, sup, part, arms: name → (plant, make_ctrl), bundles)."""
    import torch
    from .model import SplineSchedule
    from .evaluation import build_policy_ctrl, TraceSchedule
    from .training import make_controller
    plant, graph, sup, part, pb_ropc = load_bundle(cfg, args, suite, device, "ropc", calibrated=True)
    arms = {}
    bundles = {"ropc": pb_ropc}
    env_ropc, _ = deployed_envelope(cfg, suite, "ropc")
    pb_ropc.envelope = env_ropc
    verify_deployed_policy(cfg, suite, "ropc", pb_ropc, plant, env_ropc)
    odir = artifacts_dir(suite, cfg) / "openloop"
    if own_backbones and (odir / "A0_best.pt").exists():
        plant_a0, _, _, _ = load_plant(cfg, args, suite, device, odir / "A0_best.pt")
        arms["A0"] = (plant_a0, lambda pl=plant_a0: make_controller(cfg, pl, "schedule"))
    arms["A0_on_R"] = (plant, lambda pl=plant: make_controller(cfg, pl, "schedule"))
    a1 = odir / "A1t_schedule.pt"
    if own_backbones and a1.exists() and (odir / "A1t_best.pt").exists():
        plant_a1, _, _, _ = load_plant(cfg, args, suite, device, odir / "A1t_best.pt")
        st = torch.load(a1, map_location="cpu", weights_only=False)
        ck = torch.load(odir / "A1t_best.pt", map_location="cpu", weights_only=False)
        free = SplineSchedule(plant_a1.geom, st["n_knots"]).to(device)
        free.load_state_dict(ck.get("extra", {}).get("aux_state") or st["state"])     # the schedule of the best epoch
        arms["A1t"] = (plant_a1, lambda pl=plant_a1, f=free: make_controller(cfg, pl, "schedule", schedule=f))
    for v in ("base", "amortized_raw", "student_tw"):
        if v == "student_tw" and pb_ropc.rs.student_tw is None:
            continue
        if v == "amortized_raw" and pb_ropc.amortized is None:
            continue
        pbv = copy.copy(pb_ropc); pbv.variant = v
        bundles[v] = pbv
        arms[v] = (plant, lambda pl=plant, b=pbv: build_policy_ctrl(cfg, pl, b, env_ropc))
    for v in [x for x in cfg.variants.split(",") if x]:
        pdir = artifacts_dir(suite, cfg) / "policies" / v
        if not (pdir / "state.pt").exists():
            continue
        variant_plant, _, _, _, pbv = load_bundle(cfg, args, suite, device, v, calibrated=True)
        env_v, _ = deployed_envelope(cfg, suite, v)
        pbv.envelope = env_v
        verify_deployed_policy(cfg, suite, v, pbv, variant_plant, env_v)
        bundles[v] = pbv
        name = {"ropc": "ROPC", "ropc_a": "ROPC-A", "cr1": "Cr1", "c1": "C1"}.get(v, v)
        arms[name] = (plant, lambda pl=plant, b=pbv, e=env_v: build_policy_ctrl(cfg, pl, b, e))
    return plant, graph, sup, part, arms, bundles


def stage_evaluate(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    import torch
    import numpy as np
    from .evaluation import (evaluate_arm, compile_and_replay, drift_test, library_regret, identification_diagnostics, latency, component_profile,
                           mac_ledger, conservatism, realized_gains, build_policy_ctrl, build_base_ctrl, TraceSchedule)
    from .training import make_controller, ensure_dir, set_seed, train_task_route, build_backbone, Plant
    set_seed(cfg.seed + 4)
    kind = task.rsplit("_s", 1)[0].replace("eval_", "")
    edir = ensure_dir(artifacts_dir(suite, cfg) / "eval")
    plant, graph, sup, part, arms, bundles = arm_controllers(cfg, args, suite, device)
    L_eval = max(128, cfg.L_max) if cfg.scale != "preflight" else max(2 * cfg.L_max, 8)
    if kind == "p1p2":
        qualifications = {v: deployment_qualification(cfg, suite, v)
                          for v in cfg.variants.split(",") if v}
        out = {"L_eval": L_eval, "arms": {}, "deployment_qualification": qualifications}
        for name, (pl, mk) in arms.items():
            res = evaluate_arm(cfg, pl, mk(), L_eval)
            res["backbone"] = "own" if pl is not plant else "frozen_ropc"
            variant = {"ROPC": "ropc", "ROPC-A": "ropc_a", "Cr1": "cr1", "C1": "c1"}.get(name)
            if variant in qualifications:
                res.update(qualifications[variant])
            tr = res.pop("trace")
            if name == "ROPC" and tr is not None:
                torch.save(tr.cpu(), edir / "ROPC_trace_online.pt")
            out["arms"][name] = res
            print(f"[P1/P2] {name}: P1 test@L*={res['P1']['test_at_selected']:.4f} (L={res['P1']['L_selected']}) DA={res['P2']['DA']:.4f} PDD={res['P2']['PDD']:.4f}", flush=True)
        write_json(edir / "p1p2.json", out)
        return out
    if kind == "distill":
        # A2: the deployed ROPC trace frozen, backbone retrained from scratch under it
        from .evaluation import distill_mean_trace
        trace_data = distill_mean_trace(plant, arms["ROPC"][1], L_eval,
                                        sample_count=None if cfg.scale == "full" else cfg.branch_traj)
        trace = trace_data.pop("trace").to(device)
        torch.save(trace.cpu(), edir / "A2_trace.pt")
        bb2 = build_backbone(cfg, graph, plant.bb.bank).to(device)
        bb2.geom.to(device)
        plant2 = Plant(cfg, bb2, graph, sup, part, device)
        plant2.normalizer = plant.normalizer
        plant2.loss_bar = plant.loss_bar
        sched = TraceSchedule(trace).to(device)
        res = train_task_route(cfg, plant2, lambda: make_controller(cfg, plant2, "schedule", schedule=sched), cfg.epochs_distill, edir, "A2")
        from .training import load_backbone
        bb2, _ = load_backbone(cfg, graph, edir / "A2_best.pt", device)
        plant2 = Plant(cfg, bb2, graph, sup, part, device)
        plant2.normalizer = plant.normalizer
        plant2.loss_bar = plant.loss_bar
        ev = evaluate_arm(cfg, plant2, make_controller(cfg, plant2, "schedule", schedule=sched), L_eval)
        ev.pop("trace")
        out = {"train": res, "A2": ev, "trace_definition": trace_data}
        write_json(edir / "distill_A2.json", out)
        return out
    if kind == "regret":
        pb = bundles["ropc"]
        base_ctrl = build_base_ctrl(cfg, plant, pb)
        ref_ctrl = make_controller(cfg, plant, "base", pb.rs.model, pb.rs.value, pb.rs.stage, pb.rs.health, pb.rs.support, pb.rs.envelope, pb.rs.scorer, base=None, x_max=pb.rs.x_max)
        shadow = {k: mk() for k, (pl, mk) in arms.items() if pl is plant and k != "A0_on_R"}
        out = library_regret(cfg, plant, shadow, base_ctrl, ref_ctrl, cfg.regret_states, seed=cfg.seed)
        write_json(edir / "regret.json", out)
        return out
    if kind == "identdiag":
        pb = bundles["ropc"]
        diag = identification_diagnostics(cfg, plant, pb.rs, arms["ROPC"][1](), cfg.diag_states, seed=cfg.seed)
        refit = json.loads((artifacts_dir(suite, cfg) / "refit" / "refit_summary.json").read_text())
        out = {"by_depth": diag, "contrast_fidelity": {"rank_corr": refit["components"]["model"]["rank_corr"], "order_accuracy": refit["components"]["model"]["order_accuracy"],
                                                      "nmse1": refit["components"]["model"]["nmse1"], "inactive_coords": refit["components"]["model"]["inactive_coords"]}}
        write_json(edir / "identdiag.json", out)
        return out
    if kind == "ledger":
        out = {"mac_ledger": mac_ledger(cfg, plant), "latency": {}, "components": {}}
        for name in ("A0_on_R", "ROPC-A", "ROPC"):
            if name in arms:
                out["latency"]["A0" if name == "A0_on_R" else name] = latency(plant, arms[name][1], cfg.L_star, cfg.latency_reps, cfg.latency_warmups)
        cr = compile_and_replay(cfg, plant, arms["ROPC"][1](), cfg.L_star)
        trace = cr.pop("trace")
        out["latency"]["compiled"] = latency(plant, lambda: make_controller(cfg, plant, "replay", trace=trace), cfg.L_star, cfg.latency_reps, cfg.latency_warmups)
        out["components"]["ROPC"] = component_profile(cfg, plant, arms["ROPC"][1](), cfg.L_star)
        if "ROPC-A" in arms:
            out["components"]["ROPC-A"] = component_profile(cfg, plant, arms["ROPC-A"][1](), cfg.L_star)
        out["device"] = str(device)
        write_json(edir / "ledger.json", out)
        return out
    if kind == "compile":
        cr = compile_and_replay(cfg, plant, arms["ROPC"][1](), cfg.L_star)
        trace = cr.pop("trace")
        rows = drift_test(cfg, plant, lambda pl: build_policy_ctrl(cfg, pl, bundles["ropc"], bundles["ropc"].envelope), trace, cfg.L_star)
        out = {"replay": cr, "drift": rows}
        write_json(edir / "compile.json", out)
        return out
    if kind == "conservatism":
        pb = bundles["ropc"]
        env = pb.envelope
        out = {"conditions": {}, "gains": None}
        for cond in ("in_support", "sensor_shift", "beyond_Lmax"):
            out["conditions"][cond] = conservatism(cfg, plant, pb, env, part.tune_seeds[: max(1, min(len(part.tune_seeds), 8))], cfg.harm_decisions, cond)
        out["gains"] = realized_gains(cfg, plant, pb, env, cfg.gain_pairs, seed=cfg.seed)
        write_json(edir / "conservatism.json", out)
        return out
    raise ValueError(f"unknown evaluation task {task}")


def sample_summary(values):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Seed summaries require nonempty finite observations")
    n = len(values)
    mean = statistics.fmean(values)
    sd = lower = upper = None
    if n > 1:
        from scipy.stats import t
        sd = statistics.stdev(values)
        half_width = float(t.ppf(.975, n-1)) * sd / math.sqrt(n)
        lower, upper = mean-half_width, mean+half_width
    return dict(n=n, mean=mean, std=sd, ci95_lower=lower, ci95_upper=upper)


def stage_reduce(cfg, args, suite: Path, run_dir: Path, task: str, device) -> Dict:
    """Summarize measured values from every requested seed of one configuration."""
    from .registry import dataset_spec
    spec = dataset_spec(cfg.dataset)
    records, missing, audit_records = [], [], []
    for seed in args._seeds:
        base = suite / "ARTIFACTS" / f"seed{seed}"
        path = base / "eval" / "p1p2.json"
        if not path.is_file():
            missing.append(seed)
            continue
        measured = json.loads(path.read_text())
        arms = dict(measured.get("arms", {}))
        distilled = base / "eval" / "distill_A2.json"
        if distilled.is_file():
            arms["A2"] = json.loads(distilled.read_text())["A2"]
        for arm, result in arms.items():
            points = [("P1", int(result["P1"]["L_selected"]), result["P1"]["test_at_selected"]),
                      ("P2_DA", None, result["P2"]["DA"]),
                      ("P2_PDD", None, result["P2"]["PDD"])]
            points += [("depth", int(depth), value) for depth, value in result["P2"].get("curve", {}).items()]
            for protocol, depth, value in points:
                if value is None or not math.isfinite(float(value)):
                    continue
                records.append(dict(seed=seed, arm=arm, protocol=protocol, depth=depth,
                                    metric=spec.metric, metric_mode=spec.metric_mode,
                                    value=float(value), synthetic=bool(args.synthetic)))
        for variant in cfg.variants.split(","):
            audit_path = base / "audit" / variant / "audit_summary.json"
            if audit_path.is_file():
                audit_records.append(dict(seed=seed, variant=variant,
                                          result=json.loads(audit_path.read_text())))
    groups = {}
    for row in records:
        # P1 compares each seed's validation-selected model; depth sweeps retain
        # a common explicit depth. Selection depths remain in the raw rows.
        depth = row["depth"] if row["protocol"] == "depth" else None
        key = (row["arm"], row["protocol"], depth)
        groups.setdefault(key, {})[row["seed"]] = row["value"]
    summaries = [dict(arm=a, protocol=p, depth=d, **sample_summary(list(values.values())), seeds=sorted(values))
                 for (a, p, d), values in sorted(groups.items(), key=lambda x: str(x[0]))]
    paired = []
    for (arm, protocol, depth), left in groups.items():
        if arm != "ROPC":
            continue
        for baseline in ("base", "A0", "ROPC-A"):
            right = groups.get((baseline, protocol, depth), {})
            seeds = sorted(left.keys() & right.keys())
            if not seeds:
                continue
            differences = [left[seed]-right[seed] for seed in seeds]
            direction = -1 if spec.metric_mode == "min" or protocol == "P2_PDD" else 1
            paired.append(dict(comparison=f"ROPC_minus_{baseline}", protocol=protocol, depth=depth,
                               **sample_summary(differences), seeds=seeds,
                               mean_improvement=direction*statistics.fmean(differences),
                               seed_differences=[dict(seed=s, value=v) for s, v in zip(seeds, differences)]))
    out = dict(dataset=spec.name, metric=spec.metric, metric_mode=spec.metric_mode,
               synthetic=bool(args.synthetic), requested_seeds=args._seeds, missing_evaluation_seeds=missing,
               configuration_sha256=object_hash({k: v for k, v in asdict(cfg).items() if k != "seed"}),
               run_spec_sha256=args._run_spec_hash, observations=records, summaries=summaries,
               paired_differences=paired, audits=audit_records,
               statistics=dict(std_ddof=1, interval="two-sided 95% Student t; null for fewer than two seeds",
                               scope="descriptive variation across distinct training seeds",
                               independent_graph_generalization_claim=False),
               status="complete" if records and not missing else "partial",
               note="Synthetic fixture results validate execution only." if args.synthetic else
                    "Reported values are measurements produced by these runs.")
    directory = suite / "900_REDUCE"
    write_json(directory / "summary.json", out)
    for name, rows, columns in (
        ("observations.csv", records, ["seed", "arm", "protocol", "depth", "metric", "metric_mode", "value", "synthetic"]),
        ("summary.csv", summaries, ["arm", "protocol", "depth", "n", "mean", "std", "ci95_lower", "ci95_upper"]),
        ("paired_differences.csv", paired, ["comparison", "protocol", "depth", "n", "mean", "std", "ci95_lower", "ci95_upper", "mean_improvement"]),
    ):
        with (directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    lines = [f"# ROPC results: {spec.name}", "", out["note"], "",
             f"Metric: {spec.metric} ({spec.metric_mode}); requested seeds: {args._seeds}.",
             "Intervals describe seed variation, not generalization across independent graphs.", "",
             "| Arm | Protocol | Depth | Seeds | Mean | Sample SD | 95% t interval |",
             "|---|---|---|---|---|---|---|"]
    for row in summaries:
        interval = "unavailable" if row["ci95_lower"] is None else f"[{row['ci95_lower']:.6g}, {row['ci95_upper']:.6g}]"
        sd = "unavailable" if row["std"] is None else f"{row['std']:.6g}"
        lines.append(f"| {row['arm']} | {row['protocol']} | {row['depth'] or '—'} | {row['n']} | {row['mean']:.6g} | {sd} | {interval} |")
    if missing:
        lines += ["", f"Missing evaluation seeds: {missing}."]
    (directory / "REPORT.md").write_text("\n".join(lines)+"\n")
    return out


STAGES = {"warmup": stage_warmup, "bank": stage_bank, "openloop": stage_openloop,
          "rounds": stage_rounds, "refit": stage_refit, "calibrate": stage_calibrate,
          "audit": stage_audit, "evaluate": stage_evaluate, "reduce": stage_reduce}


def parse_seeds(text):
    values = str(text).split(",")
    if not values or any(not v.strip().isdigit() for v in values):
        raise ValueError("--seeds must be a comma-separated list of nonnegative integers")
    seeds = [int(v.strip()) for v in values]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds would count the same repetition more than once")
    return seeds


def resolve_config(args):
    from .registry import dataset_spec
    from .training import Config, SCALE_PRESETS
    spec = dataset_spec(args.dataset)
    if spec.group == "transfer":
        raise ValueError(f"{spec.name} is a transfer evaluation target; train a registered source dataset instead")
    supplied = {}
    if args.config:
        supplied = json.loads(Path(args.config).read_text())
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be a JSON object of Config fields")
        if "dataset" in supplied and dataset_spec(supplied["dataset"]).slug != spec.slug:
            raise ValueError("Configuration dataset differs from --dataset")
    allowed = {field.name for field in fields(Config)}
    unknown = set(supplied)-allowed
    if unknown:
        raise ValueError("Unknown configuration fields: "+", ".join(sorted(unknown)))
    cfg = Config()
    for key, value in {**spec.config_overrides(), **supplied}.items():
        setattr(cfg, key, value)
    cfg.dataset = spec.name
    cfg.scale = "preflight" if args.preset == "smoke" else args.preset
    if args.preset != "full":
        for key, value in SCALE_PRESETS[cfg.scale].items():
            setattr(cfg, key, value)
    if args.preset == "smoke":
        small = dict(d=16, dff=16, Kf=8, role_hidden=16, sketch_b=4, n_sens=48,
                     head_width=8, wA=8, epochs_warmup=1, warmup_ckpts="1", epochs_openloop=1,
                     epochs_round=1, epochs_distill=1, epochs_heads=2, epochs_ident=2,
                     epochs_dagger=1, branch_traj=1, states_per_traj=4, dagger_traj=1,
                     T_cap=8, tune_samples=2, audit_samples=2, reserve_samples=2,
                     audit_decisions_per_sample=2, regret_states=1, diag_states=1,
                     gain_pairs=2, harm_decisions=2, latency_reps=1, latency_warmups=0,
                     L_a=8, N_DA=1, L_max=8, L_star=4, K_rounds=1, val_every=1,
                     No=1, Npd=2, variants="ropc,ropc_a", students="")
        for key, value in small.items():
            setattr(cfg, key, value)
    cfg.split_index = args.split_index
    if args.mode:
        cfg.mode = args.mode
    return cfg, spec


def dataset_identity(graph):
    source = getattr(graph, "source", None)
    if source is not None and getattr(source, "fingerprint", None):
        return dict(fingerprint=source.fingerprint, scope=source.fingerprint_scope,
                    manifest=getattr(source, "manifest", {}))
    metadata = getattr(graph, "metadata", {}) or {}
    fingerprint = metadata.get("fingerprint") or metadata.get("data_fingerprint")
    if fingerprint:
        return dict(fingerprint=fingerprint, scope="full_graph_tensors_and_splits", metadata=metadata)
    # Preserve a meaningful fallback for explicitly supplied GraphData fixtures.
    digest = hashlib.sha256()
    for name in ("x", "y", "edge_index", "edge_attr", "train_idx", "valid_idx", "test_idx",
                 "batch", "edge_label_index", "edge_label", "label_mask"):
        value = getattr(graph, name, None)
        if value is None:
            continue
        array = value.detach().cpu().contiguous().numpy() if hasattr(value, "detach") else np.ascontiguousarray(value)
        digest.update(name.encode()); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return dict(fingerprint=digest.hexdigest(), scope="full_graph_tensors_and_splits")


def make_tasks(cfg, seeds):
    variants = [v for v in cfg.variants.split(",") if v]
    if not variants or len(set(variants)) != len(variants) or "ropc" not in variants:
        raise ValueError("Variants must be unique and include ropc")
    if set(variants)-{"ropc", "ropc_a", "cr1", "c1"}:
        raise ValueError("Unknown policy variant")
    return {
        "warmup": [f"warmup_s{s}" for s in seeds], "bank": [f"bank_s{s}" for s in seeds],
        "openloop": [t for s in seeds for t in (f"openloop_A0_s{s}", f"schedule_A1t_s{s}")],
        "rounds": [f"rounds_s{s}" for s in seeds], "refit": [f"refit_s{s}" for s in seeds],
        "calibrate": [f"calibrate_{v}_s{s}" for s in seeds for v in variants],
        "audit": [f"audit_{v}_s{s}" for s in seeds for v in variants],
        "evaluate": [f"eval_{kind}_s{s}" for s in seeds for kind in
                     ("p1p2", "distill", "regret", "identdiag", "ledger", "compile", "conservatism")],
        "reduce": ["reduce_all"],
    }


def task_directory(suite, stage, ordinal, task):
    return Path(suite)/STAGE_DIRECTORIES[stage]/"runs"/f"{ordinal:04d}_{task}"/"RESULT"


def task_artifacts(suite, cfg, stage, task, result_dir):
    base = artifacts_dir(suite, cfg)
    if stage in ("warmup", "bank", "openloop", "rounds", "refit"):
        directories = [base/stage]
    elif stage == "calibrate":
        variant = task.rsplit("_s", 1)[0].replace("calibrate_", "")
        directories = [base/"policies"/variant]
    elif stage == "audit":
        variant = task.rsplit("_s", 1)[0].replace("audit_", "")
        directories = [base/"audit"/variant, base/"policies"/variant]
    elif stage == "evaluate":
        directories = [base/"eval"]
    else:
        directories = [Path(suite)/"900_REDUCE"]
    paths = {Path(result_dir)/"summary.json"}
    for directory in directories:
        paths.update(p for p in directory.rglob("*") if p.is_file() and "runs" not in p.relative_to(directory).parts)
    return {str(p.relative_to(suite)): file_hash(p) for p in sorted(paths)}


def verified_complete(path, suite, spec_hash):
    marker = Path(path)/"complete.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    if record.get("run_spec_sha256") != spec_hash or record.get("status") != "complete":
        raise ValueError(f"Task completion identity mismatch: {marker}")
    for relative, digest in record["artifacts"].items():
        candidate = (Path(suite)/relative).resolve()
        if not candidate.is_relative_to(Path(suite).resolve()) or not candidate.is_file() or file_hash(candidate) != digest:
            raise ValueError(f"Completed task artifact changed or disappeared: {relative}")
    return True


def run_experiment(args):
    """Execute ordered stages; completed tasks are reused only after verification."""
    import torch
    cfg, spec = resolve_config(args)
    args._seeds = parse_seeds(args.seeds)
    suite = Path(args.output).expanduser().resolve()
    suite.mkdir(parents=True, exist_ok=True)
    args.data_root = str(Path(args.data_root).expanduser().resolve())
    device = torch_setup(args)
    stages = list(STAGES) if args.stages in (None, "all") else args.stages.split(",")
    if not stages or any(s not in STAGES for s in stages) or len(set(stages)) != len(stages):
        raise ValueError("--stages must contain unique names from: "+", ".join(STAGES))
    stages = [s for s in STAGES if s in stages]
    dataset_records = {}
    for seed in args._seeds:
        seeded_cfg = copy.deepcopy(cfg); seeded_cfg.seed = seed
        graph = load_graph_for(args, seeded_cfg)
        dataset_records[str(seed)] = dataset_identity(graph)
    scientific = dict(dataset=spec.name, dataset_slug=spec.slug, preset=args.preset,
                      synthetic=bool(args.synthetic), seeds=args._seeds, split_index=args.split_index,
                      config=asdict(cfg), python_sources=source_identity(), datasets=dataset_records)
    args._run_spec_hash = object_hash(scientific)
    manifest = suite/"run_spec.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get("run_spec_sha256") != args._run_spec_hash:
            raise ValueError("Output belongs to different code, configuration, seeds, input contents, or split; choose a new --output")
        if not args.resume:
            raise ValueError("Output already contains an experiment; pass --resume or use a new --output")
    else:
        if args.resume:
            raise ValueError("Cannot resume: run_spec.json is missing")
        write_json(manifest, dict(scientific, run_spec_sha256=args._run_spec_hash,
                                 created_utc=datetime.now(timezone.utc).isoformat()))
    write_json(suite/"environment.json", dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
               platform=platform.platform(), machine=platform.machine(), device=str(device),
               cuda=torch.version.cuda, hostname=socket.gethostname(), data_root=args.data_root,
               slurm_job_id=os.environ.get("SLURM_JOB_ID")))
    man = suite/"000_MANIFEST"
    write_json(man/"config.json", asdict(cfg))
    tasks = make_tasks(cfg, args._seeds)
    for stage, names in tasks.items():
        text = "ordinal\ttask\tstage\n"+"".join(f"{i:04d}\t{task}\t{stage}\n" for i, task in enumerate(names))
        (man/f"tasks_{stage}.tsv").write_text(text)
    # Lock across stages as well as writes: concurrent resumers must not mutate
    # the same optimizer/checkpoint state.
    import fcntl
    with (suite/".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This experiment is already running") from exc
        for stage in stages:
            if stage != "reduce":
                for prior in list(STAGES)[:list(STAGES).index(stage)]:
                    for ordinal, task in enumerate(tasks[prior]):
                        path = task_directory(suite, prior, ordinal, task)
                        if not verified_complete(path, suite, args._run_spec_hash):
                            raise RuntimeError(f"Stage {stage} requires completed {prior}; run its prerequisites first")
            for ordinal, task in enumerate(tasks[stage]):
                result_dir = task_directory(suite, stage, ordinal, task)
                if verified_complete(result_dir, suite, args._run_spec_hash):
                    print(f"[RESUME] verified {stage}/{task}", flush=True)
                    continue
                result_dir.mkdir(parents=True, exist_ok=True)
                local_cfg = copy.deepcopy(cfg)
                local_cfg.seed = task_seed(task, args._seeds[0])
                seed_phase(local_cfg.seed+100003*(list(STAGES).index(stage)+1)+ordinal, args.threads)
                started = time.perf_counter()
                print(f"[START] {stage}/{task} dataset={spec.name} device={device} synthetic={args.synthetic}", flush=True)
                try:
                    result = STAGES[stage](local_cfg, args, suite, result_dir, task, device)
                    record = dict(status="complete", dataset=spec.name, synthetic=bool(args.synthetic),
                                  seed=local_cfg.seed, stage=stage, task=task,
                                  seconds=time.perf_counter()-started, result=result)
                    write_json(result_dir/"summary.json", record)
                    write_json(result_dir/"complete.json", dict(status="complete", run_spec_sha256=args._run_spec_hash,
                               artifacts=task_artifacts(suite, local_cfg, stage, task, result_dir)))
                    (result_dir/"failure.json").unlink(missing_ok=True)
                except Exception as exc:
                    write_json(result_dir/"failure.json", dict(status="failed", stage=stage, task=task,
                               error=repr(exc), traceback=traceback.format_exc(), seconds=time.perf_counter()-started))
                    raise
                print(f"[DONE] {stage}/{task} seconds={time.perf_counter()-started:.3f}", flush=True)
    complete = all(verified_complete(task_directory(suite, stage, i, task), suite, args._run_spec_hash)
                   for stage, names in tasks.items() for i, task in enumerate(names))
    write_json(suite/"status.json", dict(status="complete" if complete else "partial", dataset=spec.name,
               synthetic=bool(args.synthetic), seeds=args._seeds, requested_stages=stages,
               run_spec_sha256=args._run_spec_hash, report="900_REDUCE/REPORT.md" if "reduce" in stages else None))
    return suite


def validate_runtime(device="cpu", threads=2):
    """Small numerical/runtime checks without downloading a benchmark dataset."""
    import torch
    from types import SimpleNamespace
    from .model import ActionGeometry, UnionSupport, fused_spmm
    selected = torch_setup(SimpleNamespace(device=device, threads=threads))
    geometry = ActionGeometry(4, 4, .05, .4, 2.).to(selected)
    actions, _ = geometry.project(torch.randn(2, 4, 6, device=selected))
    violation = float(geometry.feasibility_violation(actions).max())
    edges = torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]], device=selected)
    support = UnionSupport(edges, 3)
    values = torch.ones(support.nnz, device=selected, requires_grad=True)
    features = torch.randn(3, 4, device=selected, requires_grad=True)
    result = fused_spmm(values, features, support)
    result.square().sum().backward()
    finite = bool(torch.isfinite(result).all() and torch.isfinite(values.grad).all() and torch.isfinite(features.grad).all())
    if violation > 2e-6 or not finite:
        raise RuntimeError("Action geometry or sparse forward/backward validation failed")
    return dict(status="passed", device=str(selected), torch=torch.__version__,
                projection_max_violation=violation, sparse_forward_backward_finite=finite,
                scope="runtime and numerical primitives")



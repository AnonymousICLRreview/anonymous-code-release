"""Frozen-policy calibration, independent audits and benchmark evaluation.

Includes exact split metrics, fixed-input trace replay, response diagnostics,
measured software latency and sampled closed-loop stability experiments.
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ActionGeometry, Backbone, SplineSchedule, exact_spectral_norm
from .control import (BasePolicyGRU, ConvexHead, Controller, Envelope, HealthRegion, Planner, RotatingDesign, SupportDistance,
                          TransitionModel, depth_t, excitation_profile, pooled_response, slack_radii, eroded_bounds)
from .training import (Config, Plant, RoundState, BranchData, ControllerSnapshot, anytime_objective, base_continuation,
                        bootstrap_ci, continuation_value, dump_json, ensure_dir, log, make_controller, perturb_backbone_, run_pass,
                        response_jacobian, spearman, pairwise_order_accuracy, train_task_route, preserve_plant,
                        evaluate_depths, family_batch_size)


# ============================================================================================
# deployment bundles
# ============================================================================================


class TraceSchedule(nn.Module):
    """Fixed per-layer table of executed actions (distilled schedule A2 / compiled trace), hold-last."""

    def __init__(self, trace: torch.Tensor):
        super().__init__()
        self.register_buffer("trace", trace.clone())        # (L, Kc, p)
        self.ell0 = 8.0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        ell = int(round(float(t) * self.ell0 / max(1e-9, 1.0 - float(t))))
        return self.trace[min(ell, self.trace.shape[0] - 1)].unsqueeze(0)


@dataclass
class PolicyBundle:
    """What a deployed policy consists of."""
    variant: str
    rs: RoundState                      # final (refit) round state: model, heads, base βK, student βK+1 / tw
    heads_ref: Optional[Tuple[ConvexHead, ConvexHead]] = None   # V^{β1}, ĝ refit on the frozen backbone
    amortized: Optional[BasePolicyGRU] = None                    # β_{K+1}
    envelope: Optional[Envelope] = None
    H: Optional[int] = None

    def mode(self) -> str:
        return {"ropc": "ropc", "cr1": "ropc", "c1": "ropc", "ropc_a": "amortized_gated", "student_tw": "student_tw",
                "base": "base", "amortized_raw": "amortized_raw"}[self.variant]

    def base_policy(self) -> Optional[BasePolicyGRU]:
        return None if self.variant in ("cr1", "ropc_a") else self.rs.base


def build_policy_ctrl(cfg: Config, plant: Plant, pb: PolicyBundle, envelope: Optional[Envelope] = None) -> Controller:
    rs = pb.rs
    env = envelope or pb.envelope or rs.envelope
    v = pb.variant
    if v in ("ropc", "c1"):
        return make_controller(cfg, plant, "ropc", rs.model, rs.value, rs.stage, rs.health, rs.support, env, rs.scorer, base=rs.base,
                               H=(1 if v == "c1" else cfg.H), x_max=rs.x_max)
    if v == "cr1":
        val = pb.heads_ref[0] if pb.heads_ref else rs.value
        return make_controller(cfg, plant, "ropc", rs.model, val, rs.stage, rs.health, rs.support, env, rs.scorer, base=None,
                               planner_over_reference=True, x_max=rs.x_max)
    if v == "ropc_a":
        val = pb.heads_ref[0] if pb.heads_ref else rs.value
        return make_controller(cfg, plant, "amortized_gated", rs.model, val, rs.stage, rs.health, rs.support, env, rs.scorer,
                               student=pb.amortized, gate_value=val, x_max=rs.x_max)
    if v == "student_tw":
        return make_controller(cfg, plant, "student_tw", rs.model, rs.value, rs.stage, rs.health, rs.support, env, rs.scorer, base=rs.base,
                               student=rs.student_tw, x_max=rs.x_max)
    if v == "base":
        return make_controller(cfg, plant, "base", rs.model, rs.value, rs.stage, rs.health, rs.support, env, rs.scorer, base=rs.base, x_max=rs.x_max)
    if v == "amortized_raw":
        return make_controller(cfg, plant, "amortized_raw", rs.model, rs.value, rs.stage, rs.health, rs.support, env, rs.scorer,
                               student=pb.amortized, x_max=rs.x_max)
    raise ValueError(v)


def build_base_ctrl(cfg: Config, plant: Plant, pb: PolicyBundle) -> Controller:
    """The base β of the policy (for continuations): βK (GRU) or the reference."""
    rs = pb.rs
    return make_controller(cfg, plant, "base", rs.model, rs.value, rs.stage, rs.health, rs.support, rs.envelope, rs.scorer,
                           base=pb.base_policy(), x_max=rs.x_max)


def policy_hash(cfg: Config, pb: PolicyBundle, env: Envelope, bb: Backbone, plant: Optional[Plant] = None) -> str:
    """Hash all deployed parameters, algorithm settings and optional input state.

    Callers that freeze/audit a deployment must pass the plant, so observation
    normalization and data/preprocessing identity are covered too.
    """
    h = hashlib.sha256()
    def update(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            update({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
            raw = tensor.numpy().tobytes()
            h.update(len(raw).to_bytes(8, "little")); h.update(raw)
        elif isinstance(value, dict):
            h.update(b"dict")
            for key in sorted(value):
                update(str(key)); update(value[key])
        elif isinstance(value, (tuple, list)):
            h.update(b"sequence"); update(len(value))
            for item in value:
                update(item)
        else:
            encoded = json.dumps(value, sort_keys=True, default=str, allow_nan=True).encode()
            h.update(len(encoded).to_bytes(8, "little")); h.update(encoded)
    update({"schema": 2, "config": asdict(cfg), "backbone": bb.state_dict(), "round": pb.rs.state_dict(),
            "reference_heads": None if pb.heads_ref is None else [m.state_dict() for m in pb.heads_ref],
            "amortized": None if pb.amortized is None else pb.amortized.state_dict(),
            "envelope": env.to_dict() if hasattr(env, "to_dict") else asdict(env), "variant": pb.variant, "H": pb.H})
    if plant is not None:
        update({"normalizer": plant.normalizer.state_dict(), "loss_bar": plant.loss_bar,
                "sensor_sketch": plant.sensor.S, "sensor_seed": plant.sens_seed,
                "task_type": getattr(plant.graph, "task_type", "node_classification"),
                "metric": getattr(plant.graph, "metric", "accuracy")})
        if plant.source is not None:
            update({"data_fingerprint": plant.source.fingerprint,
                    "data_scope": getattr(plant.source, "fingerprint_scope", "")})
        else:
            update({"features": plant.reference_graph.x, "edges": plant.reference_graph.edge_index})
    return h.hexdigest()


@contextlib.contextmanager
def evaluation_sample(plant: Plant, sample_id: int, split: str, base_state=None):
    """Families use distinct graphs; single-graph diagnostics perturb checkpoints."""
    with preserve_plant(plant):
        if plant.is_family:
            plant.bind_sample(sample_id, split)
        else:
            plant.bind_graph(plant.reference_graph)
            if base_state is not None:
                plant.bb.load_state_dict(base_state)
                perturb_backbone_(plant.bb, plant.cfg.ckpt_perturb, sample_id)
        plant.prepare(grad=False)
        try:
            yield
        finally:
            if base_state is not None and not plant.is_family:
                plant.bb.load_state_dict(base_state)
                plant.bb.update_sigma()


def observation_noise(raw, plant, noise_seed, ell, scale):
    """A stable per-decision key makes snapshot replays use identical sensor noise."""
    if scale <= 0:
        return raw
    gen = torch.Generator().manual_seed(int(noise_seed)*7919+int(ell))
    return raw + scale*plant.normalizer.scale.to(raw.device)*torch.randn(raw.shape,generator=gen).to(raw.device)


# ============================================================================================
# decisions with snapshots, contrast errors
# ============================================================================================


@torch.no_grad()
def run_with_snapshots(plant: Plant, ctrl: Controller, L: int, snap_layers: List[int], sens_seed: Optional[int],
                       obs_noise: float = 0.0, noise_seed: int = 0, readout_layers: Optional[List[int]] = None,
                       stage_losses: bool = False):
    """Run the policy for L layers; snapshot (H_ℓ, memory) before the decision at the listed layers;
    optional Gaussian observation noise (sensor shift)."""
    if plant.H0 is None:
        plant.prepare(grad=False)
    ctrl.role_mask = plant.role_mask
    ctrl.reset(plant.B, plant.ctx)
    H = plant.H0
    snaps, recs, gs, readouts = {}, [], [], {}
    rl = set(readout_layers or [])
    for ell in range(L):
        if ell in snap_layers:
            snaps[ell] = (H, ControllerSnapshot(ctrl))
        z_raw = plant.observe(H, sens_seed) if ctrl.needs_sensing else None
        if z_raw is not None:
            z_raw = observation_noise(z_raw, plant, noise_seed, ell, obs_noise)
        if stage_losses:
            gs.append(float(plant.stage_loss(H).mean()))
        A, rec = ctrl.step(ell, z_raw, plant.ctx)
        H = plant.layer(H, A)
        ctrl.commit(ell, A, rec, plant.ctx)
        recs.append({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in rec.items()} | {"A": A.cpu()})
        if (ell + 1) in rl:
            readouts[ell + 1] = {"valid": plant.accuracy(H, plant.graph.valid_idx), "test": plant.accuracy(H, plant.graph.test_idx)}
    return {"H": H, "snaps": snaps, "recs": recs, "g": gs, "readouts": readouts}


@torch.no_grad()
def contrast_error(cfg: Config, plant: Plant, ctrl: Controller, base_ctrl: Controller, snap, ell: int, sens_seed: Optional[int],
                   obs_noise: float = 0.0, noise_seed: int = 0, T_cap: Optional[int] = None, extra_full: bool = False) -> Dict:
    """Re-issue the decision at a snapshot and compare the identified contrast Q̂(b) − Q̂(A) with the true
    contrast Q^β(b) − Q^β(A) from paired base continuations truncated at T_cap.  With extra_full the
    full step b + d is evaluated as well (conservatism table)."""
    T_cap = T_cap or cfg.T_cap
    H, memory = snap
    memory.restore(ctrl)
    if plant.B != 1:
        raise ValueError("Contrast evaluation requires one graph per statistical sample")
    ctrl.role_mask = base_ctrl.role_mask = plant.role_mask
    z_raw = observation_noise(plant.observe(H, sens_seed), plant, noise_seed, ell, obs_noise)
    A, rec = ctrl.step(ell, z_raw, plant.ctx)
    b = rec["b"]
    out = {"ell": ell, "rho": float(rec["rho"][0]) if "rho" in rec else 1.0, "r": float(rec["r"][0]) if "r" in rec else 0.0,
           "bin": int(rec["bin"][0]) if "bin" in rec else 0, "dnorm": float(plant.geom.norm_1sigma(A, b)[0]),
           "delta_hat": float(rec["delta_hat"][0]) if "delta_hat" in rec else 0.0, "ramp": bool(rec["ramp"][0]) if "ramp" in rec else False}
    actions = {"b": b, "A": A}
    if extra_full and "d" in rec:
        actions["full"] = b + rec["d"]
        out["full_norm"] = float(plant.geom.norm_1sigma(actions["full"], b)[0])
    # memory after executing each action, under the base
    t = depth_t(ell, cfg.ell0).to(H.device).expand(1)
    dz0 = torch.tensor([ell == 0], device=H.device)
    G, Qhat = {}, {}
    for name, act in actions.items():
        Hn = plant.layer(H, act)
        base_ctrl.reset(plant.B, plant.ctx)
        base_ctrl.x_prior = ctrl.model.predict(rec["x"], t, plant.geom.to_tangent(act), plant.ctx, dz0)
        base_ctrl.eta = rec["eta_next"].clone() if "eta_next" in rec else torch.zeros_like(base_ctrl.eta)
        base_ctrl.A_prev = act.clone()
        base_ctrl.V_prev = None
        Gk, _, _ = base_continuation(plant, base_ctrl, Hn, ell + 1, T_cap, sens_seed, n_obs=0, hazard=cfg.hazard)
        G[name] = Gk
        Qhat[name] = float(ctrl.lookahead_from_record(ell, rec, act, plant.ctx)[0])
    hz = cfg.hazard
    true_contrast = (1 - hz) * (G["b"] - G["A"])
    pred_contrast = Qhat["b"] - Qhat["A"]
    allowance = 2.0 * (1 - hz) ** T_cap
    out.update({"true_contrast": true_contrast, "pred_contrast": pred_contrast, "e_abs": abs(pred_contrast - true_contrast) + allowance,
                "G": G, "Qhat": Qhat})
    return out


def select_decisions(recs: List[Dict], n_max: int, all_decisions: bool, gen: torch.Generator) -> List[int]:
    acting = [i for i, r in enumerate(recs) if ("rho" in r and float(r["rho"][0]) > 0.0)]
    if all_decisions or len(acting) <= n_max:
        return acting
    perm = torch.randperm(len(acting), generator=gen)[:n_max]
    return sorted(acting[int(i)] for i in perm)


# ============================================================================================
# calibration of the envelope (Appendix B: tuning fixes every quantity the policy uses)
# ============================================================================================


def quantile_index_slope(scores: List[float], q: float) -> float:
    """The ⌈(n+1)q⌉ order statistic, with an infinite (n+1)-st element.

    Empty or undersized bins provide no finite coverage envelope. Replacing
    this value by the largest observed score would overstate coverage.
    """
    if not 0 < q < 1:
        raise ValueError("The calibration quantile must lie strictly between zero and one")
    if not scores:
        return float("inf")
    if any(math.isnan(float(v)) or float(v) < 0 for v in scores):
        raise ValueError("Calibration scores must be nonnegative and not NaN")
    s = sorted(scores)
    k = int(math.ceil((len(s) + 1) * q))
    return float(s[k - 1]) if k <= len(s) else float("inf")


@torch.no_grad()
def calibrate_policy(cfg: Config, plant: Plant, pb: PolicyBundle, tune_seeds: List[int], out_dir: Path, n_passes: int = 2) -> Dict:
    """Fit tuning-only reliability statistics, then regenerate policy bin scores.

    Each family score is a maximum over acting decisions in one held-out graph.
    Single-graph samples are checkpoint/sensing perturbations and support only
    an empirical statement. Empty and undersized bins retain infinite slopes.
    """
    if n_passes < 1 or not tune_seeds:
        raise ValueError("Calibration needs tuning samples and at least one regeneration pass")
    rs = pb.rs
    base_state = copy.deepcopy(plant.bb.state_dict())
    env = pb.envelope or rs.envelope
    n_bins = 3
    gen = torch.Generator().manual_seed(cfg.seed + 101)
    info = {"passes": [], "sample_unit": "graph" if plant.is_family else "checkpoint_and_sensing_seed",
            "decision_scope": "all_acting" if cfg.audit_all_decisions else "sampled_acting",
            "sample_ids": list(map(int, tune_seeds))}
    t0 = time.time()
    # Freeze the CDF and support/health thresholds before recording any binned
    # errors: fitting them after binning would change the meaning of each bin.
    innovations, latents, health_distances, actions = [], [], [], []
    for sample in tune_seeds:
        with evaluation_sample(plant, sample, "tune", base_state):
            ctrl = build_policy_ctrl(cfg, plant, pb, env)
            first = run_with_snapshots(plant, ctrl, cfg.L_a, [], sample)
            for rec in first["recs"]:
                innovations.append(rec["e"][0]); latents.append(rec["x"][0])
                actions.append(plant.geom.to_tangent(rec["A"])[0])
                if "health_dist" in rec:
                    health_distances.append(float(rec["health_dist"][0]))
    if innovations:
        rs.scorer.fit_tuning(torch.stack(innovations), torch.stack(latents))
        if health_distances:
            rs.health.rho_c = float(torch.quantile(torch.tensor(health_distances), 0.95))
        if actions:
            distances = rs.support.m(torch.stack(actions).to(plant.device))
            rs.support.chi_q = float(torch.quantile(distances, cfg.sup_quantile))
    info["reliability_fit_rows"] = len(innovations)
    for iteration in range(n_passes):
        bin_scores = {b: [] for b in range(n_bins)}
        n_decisions = 0
        for sample in tune_seeds:
            with evaluation_sample(plant, sample, "tune", base_state):
                ctrl = build_policy_ctrl(cfg, plant, pb, env)
                base_ctrl = build_base_ctrl(cfg, plant, pb)
                first = run_with_snapshots(plant, ctrl, cfg.L_a, [], sample)
                chosen = select_decisions(first["recs"], cfg.audit_decisions_per_sample,
                                          bool(cfg.audit_all_decisions), gen)
                if not chosen:
                    continue
                second = run_with_snapshots(plant, ctrl, max(chosen)+1, chosen, sample)
                per_bin = {}
                for ell in chosen:
                    ce = contrast_error(cfg, plant, ctrl, base_ctrl, second["snaps"][ell], ell, sample)
                    if ce["dnorm"] <= 1e-9:
                        continue
                    # Use the final three-bin reliability rule even when the
                    # provisional envelope has a single constant slope.
                    bin_id = min(2, int(ce["r"] >= 0.80) + int(ce["r"] >= 0.95))
                    per_bin[bin_id] = max(per_bin.get(bin_id, 0.), ce["e_abs"]/ce["dnorm"])
                    n_decisions += 1
                for bin_id, score in per_bin.items():
                    bin_scores[bin_id].append(score)
        slopes = [quantile_index_slope(bin_scores[b], cfg.q_t) for b in range(n_bins)]
        for b in range(1, n_bins):
            slopes[b] = max(slopes[b], slopes[b-1])
        env = Envelope(tuple(slopes))
        info["passes"].append({"pass": iteration, "envelope": env.to_dict(),
            "slopes": [v if math.isfinite(v) else None for v in slopes],
            "n_scores_per_bin": {b: len(v) for b,v in bin_scores.items()},
            "n_decisions": n_decisions, "rho_c": rs.health.rho_c,
            "chi_q": rs.support.chi_q, "q_e": rs.scorer.q_e, "q_d": rs.scorer.q_d})
        log(f"[CALIBRATE:{pb.variant}] pass {iteration}: slopes={slopes} decisions={n_decisions} ({time.time()-t0:.0f}s)")
    info["envelope"] = env.to_dict()
    info["seconds"] = time.time()-t0
    return {"envelope": env, "info": info}


# ============================================================================================
# audit (Proposition 3) — single graph: samples are sensing seeds + checkpoint perturbations
# ============================================================================================


def perturbed_plant(cfg: Config, plant: Plant, rel_scale: float, seed: int) -> Plant:
    """A plant on a copy of the graph whose features are perturbed by rel_scale × per-feature std
    (sensed set, roles, kernels and H₀ recomputed); shares backbone, normalizer and ℓ̄."""
    g = torch.Generator().manual_seed(seed)
    x = plant.graph.x
    graph_p = copy.copy(plant.graph)
    graph_p.x = x + rel_scale * x.std(0, keepdim=True, unbiased=False) * torch.randn(x.shape, generator=g).to(x.device)
    plant_p = Plant(cfg, plant.bb, graph_p, plant.sup, plant.part, plant.device, plant.sens_seed)
    plant_p.normalizer = plant.normalizer
    plant_p.loss_bar = plant.loss_bar
    plant_p.prepare(grad=False)
    return plant_p


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    from scipy.stats import beta as _beta
    if n <= 0 or k <= 0:
        return 0.0
    return float(_beta.ppf(alpha, k, n - k + 1))


@torch.no_grad()
def block_distance(plant: Plant, cfg: Config, H1, H2, c1: Controller, c2: Controller, x_max: float) -> float:
    """Max over blocks of the diameter-normalised distances of the augmented state (H, x⁻, A_{ℓ−1}, V_prev, η)."""
    geom = plant.geom
    dH = float((H1 - H2).norm(dim=1).max()) / (2 * cfg.R0)
    ds = [dH]
    if c1.x_prior is not None and c2.x_prior is not None:
        ds.append(float((c1.x_prior - c2.x_prior).norm()) / (2 * x_max))
    ds.append(float(geom.norm_1sigma(c1.A_prev, c2.A_prev)[0]) / (2 * geom.Kc))
    if c1.V_prev is not None and c2.V_prev is not None:
        ds.append(float((c1.V_prev - c2.V_prev).abs().sum()) / (2 * geom.Kc * c1.V_prev.shape[1]))
    ds.append(float((c1.eta - c2.eta).abs().max()) / 2.0)
    return max(ds)


@torch.no_grad()
def audit_policy(cfg: Config, plant: Plant, pb: PolicyBundle, env: Envelope, seeds: List[int], out_dir: Path, partition_name: str) -> Dict:
    """Audit a frozen policy once; no fitting or fallback selection occurs here."""
    if partition_name not in ("audit", "reserve"):
        raise ValueError("Audit observations must come from the audit or reserve partition")
    rs = pb.rs
    base_state = copy.deepcopy(plant.bb.state_dict())
    n_bins = env.n_bins
    gen = torch.Generator().manual_seed(cfg.seed+202)
    rows = []
    t0 = time.time()
    hz = cfg.hazard
    for sample in seeds:
        with evaluation_sample(plant, sample, partition_name, base_state):
            ctrl = build_policy_ctrl(cfg, plant, pb, env)
            base_ctrl = build_base_ctrl(cfg, plant, pb)
            row = {"sample_id": int(sample)}
            first = run_with_snapshots(plant, ctrl, cfg.L_a, [], sample)
            chosen = select_decisions(first["recs"], cfg.audit_decisions_per_sample,
                                      bool(cfg.audit_all_decisions), gen)
            covered = {b: True for b in range(n_bins)}
            visited = {b: False for b in range(n_bins)}
            gains = []
            if chosen:
                second = run_with_snapshots(plant, ctrl, max(chosen)+1, chosen, sample)
                for ell in chosen:
                    ce = contrast_error(cfg, plant, ctrl, base_ctrl, second["snaps"][ell], ell, sample)
                    if ce["dnorm"] <= 1e-9:
                        continue
                    b = min(ce["bin"], n_bins-1)
                    visited[b] = True
                    slope = float(env.L_bar(torch.tensor([ce["r"]]))[0])
                    covered[b] = covered[b] and ce["e_abs"] <= slope*ce["dnorm"]
                    gains.append((1-hz)*(ce["G"]["b"]-ce["G"]["A"]))
            for b in range(n_bins):
                row[f"visit_{b}"] = visited[b]
                row[f"cov_{b}"] = covered[b] if visited[b] else None
            row["simultaneous"] = all(covered[b] for b in range(n_bins) if visited[b])
            row["n_checked_decisions"] = len(gains)
            row["mean_true_gain"] = float(np.mean(gains)) if gains else float("nan")
            Jpi = anytime_objective(run_pass(plant, build_policy_ctrl(cfg, plant, pb, env), cfg.T_cap,
                prepared=True, sens_seed=sample, stage_losses=True)["g"], hz)
            Jbe = anytime_objective(run_pass(plant, build_base_ctrl(cfg, plant, pb), cfg.T_cap,
                prepared=True, sens_seed=sample, stage_losses=True)["g"], hz)
            row["J_pi"], row["J_beta"] = Jpi, Jbe
            row["trajectory"] = bool(Jpi-Jbe+2*(1-hz)**cfg.T_cap <= (1-hz)**cfg.L_a)
            plant_p = perturbed_plant(cfg, plant, 1e-2, sample+55)
            plant.prepare(grad=False)
            ctrl_c = build_policy_ctrl(cfg, plant, pb, env)
            ctrl_p = build_policy_ctrl(cfg, plant_p, pb, env)
            ctrl_c.role_mask, ctrl_p.role_mask = plant.role_mask, plant_p.role_mask
            ctrl_c.reset(plant.B, plant.ctx); ctrl_p.reset(plant_p.B, plant_p.ctx)
            Hc, Hp = plant.H0, plant_p.H0
            d0 = block_distance(plant, cfg, Hc, Hp, ctrl_c, ctrl_p, rs.x_max)
            ok, dmax = True, d0
            for ell in range(cfg.L_a):
                Ac, rc = ctrl_c.step(ell, plant.observe(Hc, sample), plant.ctx)
                Ap, rp = ctrl_p.step(ell, plant_p.observe(Hp, sample), plant_p.ctx)
                Hc, Hp = plant.layer(Hc, Ac), plant_p.layer(Hp, Ap)
                ctrl_c.commit(ell, Ac, rc, plant.ctx); ctrl_p.commit(ell, Ap, rp, plant_p.ctx)
                distance = block_distance(plant, cfg, Hc, Hp, ctrl_c, ctrl_p, rs.x_max)
                dmax = max(dmax, distance)
                ok = ok and distance <= d0*(1+1e-6)
            row["non_expansion"] = bool(ok)
            row["entry_distance"], row["max_distance"] = d0, dmax
            rows.append(row)
            log(f"[AUDIT:{pb.variant}:{partition_name}] sample {sample}: bins={[row[f'cov_{b}'] for b in range(n_bins)]} "
                f"simult={row['simultaneous']} traj={row['trajectory']} nonexp={ok} ({time.time()-t0:.0f}s)")
    n = len(rows)
    alpha = 0.05/(n_bins+2)
    summary = {"partition": partition_name, "n_samples": n, "alpha_per_event": alpha, "bins": {},
        "sample_unit": "graph" if plant.is_family else "checkpoint_and_sensing_seed",
        "decision_scope": "all_acting" if cfg.audit_all_decisions else "sampled_acting",
        "note": ("Distinct held-out graphs; bounds refer to the recorded sampling population." if plant.is_family else
                 "Single graph: checkpoint and sensing perturbations support empirical statements only.")}
    all_pass = n > 0
    for b in range(n_bins):
        nb = sum(bool(r[f"visit_{b}"]) for r in rows)
        kb = sum(bool(r[f"visit_{b}"] and r[f"cov_{b}"]) for r in rows)
        lower = clopper_pearson_lower(kb, nb, alpha)
        passed = nb > 0 and lower >= cfg.q_c
        summary["bins"][b] = {"n": nb, "k": kb, "lower_bound": lower, "pass": passed}
        all_pass = all_pass and passed
    ks = sum(r["simultaneous"] for r in rows)
    kt = sum(r["trajectory"] for r in rows)
    kn = sum(r["non_expansion"] for r in rows)
    finite_gains = [r["mean_true_gain"] for r in rows if math.isfinite(r["mean_true_gain"])]
    summary.update({"simultaneous": {"k": ks, "n": n, "lower_bound": clopper_pearson_lower(ks,n,alpha)},
        "trajectory": {"k": kt, "n": n, "lower_bound": clopper_pearson_lower(kt,n,alpha)},
        "non_expansion": {"k": kn, "n": n, "lower_bound": clopper_pearson_lower(kn,n,0.05)},
        "recorded_bin_pass": bool(all_pass), "policy_pass": bool(all_pass and cfg.audit_all_decisions),
        "family_all_decision_guarantee_eligible": bool(plant.is_family and cfg.audit_all_decisions and n),
        "mean_true_gain": float(np.mean(finite_gains)) if finite_gains else None,
        "mean_J_pi": float(np.mean([r["J_pi"] for r in rows])) if rows else None,
        "mean_J_beta": float(np.mean([r["J_beta"] for r in rows])) if rows else None,
        "seconds": time.time()-t0})
    import csv
    out_dir = ensure_dir(out_dir)
    with open(out_dir/f"indicator_table_{partition_name}.csv", "w", newline="") as handle:
        fields = list(rows[0]) if rows else ["sample_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return summary


# ============================================================================================
# protocols P1 / P2 on a single graph; compile mode
# ============================================================================================


@torch.no_grad()
def evaluate_arm(cfg: Config, plant: Plant, ctrl: Controller, L_eval: int = 128, sens_seed: int = 0, record: bool = True) -> Dict:
    """P1 validation-selected depth and P2 whole-split depth curves.

    Family metrics concatenate all predictions (MRR averages graph scores).
    PDD is oriented so positive means deterioration for both max/min metrics.
    """
    if L_eval < 2:
        raise ValueError("P2 requires at least two evaluated layers")
    depths = list(range(1, L_eval+1))
    if plant.is_family:
        valid = evaluate_depths(plant, lambda: ctrl, depths, "valid", sens_seed)
        test = evaluate_depths(plant, lambda: ctrl, depths, "test", sens_seed)
        rd = {d: {"valid": valid[d], "test": test[d]} for d in depths}
        out = {"recs": []}
    else:
        plant.prepare(grad=False)
        out = run_with_snapshots(plant, ctrl, L_eval, [], sens_seed, readout_layers=depths)
        rd = out["readouts"]
    grid = sorted({d for d in (4,8,12,16,24,32,48,64,cfg.L_star) if 1 <= d <= L_eval})
    if not grid:
        grid = [L_eval]
    vals = {d: rd[d]["valid"] for d in grid if math.isfinite(rd[d]["valid"])}
    if not vals:
        raise ValueError("No finite validation metric is available for depth selection")
    minimize = getattr(plant.graph, "metric_mode", "max") == "min"
    L_sel = (min if minimize else max)(vals, key=vals.get)
    tests = [rd[d]["test"] for d in range(2,L_eval+1)]
    if not all(math.isfinite(v) for v in tests):
        raise ValueError("A held-out test depth has no finite benchmark metric")
    peak_depth = int(np.argmin(tests) if minimize else np.argmax(tests))+2
    pdd = (rd[L_eval]["test"]-min(tests)) if minimize else (max(tests)-rd[L_eval]["test"])
    res = {"metric": getattr(plant.graph,"metric","accuracy"),
           "metric_mode": "min" if minimize else "max", "metric_scale": "benchmark_raw",
           "evaluation_scope": "all_official_split_graphs" if plant.is_family else "official_node_split",
           "P1": {"L_selected": L_sel, "valid_at_selected": vals[L_sel],
                  "test_at_selected": rd[L_sel]["test"],
                  "test_at_L_star": rd[cfg.L_star]["test"] if cfg.L_star in rd else None,
                  "valid_sweep": vals},
           "P2": {"DA": float(np.mean(tests)), "PDD": float(pdd), "peak_depth": peak_depth,
                  "test_at_deepest": rd[L_eval]["test"],
                  "curve": {d: rd[d]["test"] for d in depths}}}
    recs = out["recs"]
    if recs and "rho" in recs[0]:
        rho = torch.tensor([float(r["rho"][0]) for r in recs])
        auth = torch.tensor([float(ctrl.authority(r["ell"], r["A"].to(plant.device))[0]) for r in recs])
        dn = torch.tensor([float(r["dnorm"][0]) for r in recs])
        dmax = plant.geom.Kc * (plant.geom.M + 2) * cfg.delta_tr
        A_all = torch.stack([r["A"][0] for r in recs])
        floor = ((1 + cfg.r_min) * A_all[..., 0] + A_all[..., 1] >= 1 - 1e-6).float().mean()
        in_sup = slice(0, min(cfg.L_max, len(recs)))
        res["telemetry"] = {"rho_full_share": float((rho == 1).float().mean()), "rho_zero_share": float((rho == 0).float().mean()),
                            "rho_partial_share": float(((rho > 0) & (rho < 1)).float().mean()),
                            "rho_full_share_in_support": float((rho[in_sup] == 1).float().mean()),
                            "rho_full_share_beyond_Lmax": float((rho[cfg.L_max:] == 1).float().mean()) if len(recs) > cfg.L_max else None,
                            "authority_p50": float(auth.median()), "authority_p90": float(torch.quantile(auth, 0.9)),
                            "floor_share": float(floor), "ramp_share": float(torch.tensor([bool(r["ramp"][0]) for r in recs]).float().mean()),
                            "dnorm_over_dmax_median": float((dn / dmax).median()) if dmax > 0 else None,
                            "health_active_share": float(torch.tensor([float(r.get("health_dist", torch.zeros(1))[0]) > ctrl.health.rho_c for r in recs]).float().mean()) if ctrl.health is not None else None,
                            "cert_full_share": float((torch.stack([r["omega"][0] for r in recs if "omega" in r]) >= 1).float().mean()) if "omega" in recs[0] else None,
                            "cert_partial_share": float(((torch.stack([r["omega"][0] for r in recs if "omega" in r]) > 0) & (torch.stack([r["omega"][0] for r in recs if "omega" in r]) < 1)).float().mean()) if "omega" in recs[0] else None,
                            "pg_residual_median": float(torch.stack([r["pg_residual"][0] for r in recs if "pg_residual" in r]).median()) if "pg_residual" in recs[0] else None,
                            "objective_reduction_mean": float(torch.stack([(r["J_warm"][0] - r["J_plan"][0]) for r in recs if "J_warm" in r]).mean()) if "J_warm" in recs[0] else None}
    res["trace"] = (torch.stack([r["A"][0] if plant.B == 1 else r["A"] for r in recs]) if recs and record else None)
    return res


@torch.no_grad()
def compile_and_replay(cfg: Config, plant: Plant, ctrl_online: Controller, L: int, sens_seed: int = 0) -> Dict:
    """Corollary 1: cache the online trace, replay it open loop, compare bit for bit."""
    plant.prepare(grad=False)
    online = run_with_snapshots(plant, ctrl_online, L, [], sens_seed)
    trace = torch.stack([r["A"][0] if plant.B == 1 else r["A"] for r in online["recs"]]).to(plant.device)
    ctrl_replay = make_controller(cfg, plant, "replay", trace=trace)
    replay = run_with_snapshots(plant, ctrl_replay, L, [], sens_seed)
    from .tasks import predictions
    lo = predictions(plant.graph, online["H"], plant.bb)
    lr = predictions(plant.graph, replay["H"], plant.bb)
    return {"scope": "exact_bound_input", "sample_id": getattr(plant.graph,"sample_id",None),
            "n_graphs": plant.B, "metric": getattr(plant.graph,"metric","accuracy"),
            "bitwise_equal": bool(torch.equal(lo, lr)), "max_abs_diff": float((lo - lr).abs().max()),
            "acc_online": plant.accuracy(online["H"], plant.graph.test_idx), "acc_replay": plant.accuracy(replay["H"], plant.graph.test_idx), "trace": trace}


@torch.no_grad()
def drift_test(cfg: Config, plant: Plant, make_online, trace: torch.Tensor, L: int, sigmas=(0.0, 0.01, 0.05, 0.10), sens_seed: int = 0) -> List[Dict]:
    """Table LVIII(b): online ROPC against the trace compiled on the undrifted input, under Gaussian
    feature noise of relative scale σ (preprocessing recomputed on the drifted input)."""
    rows = []
    for sg in sigmas:
        pl = plant if sg == 0 else perturbed_plant(cfg, plant, sg, int(1000 * sg) + 7)
        pl.prepare(grad=False)
        on = run_with_snapshots(pl, make_online(pl), L, [], sens_seed)
        rp = run_with_snapshots(pl, make_controller(cfg, pl, "replay", trace=trace), L, [], sens_seed)
        rows.append({"sigma": sg, "online": pl.accuracy(on["H"], pl.graph.test_idx), "compiled": pl.accuracy(rp["H"], pl.graph.test_idx)})
    plant.prepare(grad=False)
    return rows


@torch.no_grad()
def distill_mean_trace(plant: Plant, make_ctrl, L: int, sample_count: Optional[int] = None,
                       sens_seed: int = 0) -> Dict:
    """Freeze one feasible mean schedule using value-training graphs only.

    A family has no input-independent exact online trace. Its A2 comparison
    uses the arithmetic mean of per-graph executed traces, which stays in the
    convex action polytope. This implementation choice is recorded explicitly.
    """
    if L < 1:
        raise ValueError("A distilled schedule needs positive depth")
    total, count, sample_ids = None, 0, []
    with preserve_plant(plant):
        if plant.is_family:
            ids = list(map(int, plant.source.partitions["value"]))
            if sample_count is not None:
                ids = ids[:max(0, int(sample_count))]
            graphs = (plant.source.sample(i, "value") for i in ids)
        else:
            ids, graphs = [-1], [plant.reference_graph]
        for sample_id, graph in zip(ids, graphs):
            plant.bind_graph(graph); plant.prepare(grad=False)
            output = run_with_snapshots(plant, make_ctrl(), L, [], sens_seed)
            trace = torch.stack([r["A"] for r in output["recs"]]).double().sum(1)
            total = trace if total is None else total+trace
            count += plant.B; sample_ids.append(sample_id)
    if count == 0:
        raise ValueError("No value-training graphs were selected for A2")
    return {"trace": (total/count).float(), "sample_ids": sample_ids, "n_graphs": count,
            "partition": "value" if plant.is_family else "single_training_graph",
            "scope": "mean_training_graph_trace" if plant.is_family else "exact_bound_input_trace",
            "definition": "Arithmetic mean over value-training graph traces; fixed during backbone retraining."}


# ============================================================================================
# library task regret (Theorem 2; Table XXVII TR/RC on reference states)
# ============================================================================================


@torch.no_grad()
def library_regret(cfg: Config, plant: Plant, arms: Dict[str, Controller], base_ctrl: Controller, ref_ctrl: Controller, n_states: int,
                   seed: int, T_cap: Optional[int] = None) -> Dict:
    if n_states < 1:
        raise ValueError("Regret diagnostics need a positive state budget")
    totals = {name: {"TR": 0., "RC": 0., "n_states": 0} for name in arms}
    ids = list(plant.source.partitions["test"]) if plant.is_family else [-1]
    used = []
    with preserve_plant(plant):
        remaining, trajectory = n_states, 0
        while remaining:
            sample = int(ids[trajectory % len(ids)])
            if plant.is_family:
                plant.bind_sample(sample, "test")
            else:
                plant.bind_graph(plant.reference_graph)
            budget = min(remaining, cfg.L_max)
            result = _library_regret_one(cfg, plant, arms, base_ctrl, ref_ctrl, budget, seed+trajectory*1009, T_cap)
            for name, result_arm in result.items():
                count = result_arm["n_states"]
                totals[name]["n_states"] += count
                for key in ("TR", "RC"):
                    totals[name][key] += result_arm[key]*count
            used.append(sample); remaining -= budget; trajectory += 1
    for item in totals.values():
        for key in ("TR", "RC"):
            item[key] /= item["n_states"]
        item.update({"sample_ids": used, "n_distinct_graphs": len(set(used)),
                     "scope": "held_out_test_graph_states" if plant.is_family else "single_graph_sensing_trajectories"})
    return totals


@torch.no_grad()
def _library_regret_one(cfg: Config, plant: Plant, arms: Dict[str, Controller], base_ctrl: Controller, ref_ctrl: Controller, n_states: int,
                   seed: int, T_cap: Optional[int] = None) -> Dict:
    """States along the reference trajectory; every arm runs in shadow (observes the reference states,
    commits the executed reference action) and proposes its action; the library adds the base proposal
    and 2K_lib antithetic branches; each library action is executed once and continued under the base."""
    T_cap = T_cap or cfg.T_cap
    geom, dev = plant.geom, plant.device
    gen = torch.Generator().manual_seed(seed)
    design = RotatingDesign(geom.du, cfg.K_lib)
    plant.prepare(grad=False)
    L = cfg.L_max
    layers = sorted(torch.randperm(L, generator=gen)[:min(n_states,L)].tolist())
    for c in list(arms.values()) + [base_ctrl, ref_ctrl]:
        c.role_mask = plant.role_mask
        c.reset(plant.B, plant.ctx)
    H = plant.H0
    sens_seed = 4242 + seed
    per_arm = {k: [] for k in arms}
    raw = {k: [] for k in arms}
    for ell in range(max(layers) + 1):
        z_raw = plant.observe(H, sens_seed)
        A_ref, rec_ref = ref_ctrl.step(ell, z_raw, plant.ctx)
        props, recs = {}, {}
        for k, c in arms.items():
            props[k], recs[k] = c.step(ell, z_raw, plant.ctx)
        A_b, rec_b = base_ctrl.step(ell, z_raw, plant.ctx)
        if ell in layers:
            lib = {"ref": A_ref, "base": A_b}
            lib.update({k: v for k, v in props.items()})
            rows, _ = design.next_rows()
            u_b = geom.to_tangent(A_b)
            for i in range(rows.shape[0]):
                for sgn, nm in ((1.0, "p"), (-1.0, "m")):
                    lib[f"br{i}{nm}"] = geom.project(geom.from_tangent(u_b + sgn * 0.05 * rows[i:i + 1].to(dev)))[0]
            J = {}
            t = depth_t(ell, cfg.ell0).to(dev).expand(1)
            dz0 = torch.tensor([ell == 0], device=dev)
            g_now = float(plant.stage_loss(H))
            for name, act in lib.items():
                Hn = plant.layer(H, act)
                snap = ControllerSnapshot(base_ctrl)
                base_ctrl.x_prior = base_ctrl.model.predict(rec_b["x"], t, geom.to_tangent(act), plant.ctx, dz0)
                base_ctrl.A_prev = act.clone()
                base_ctrl.eta = rec_b["eta_next"].clone()
                base_ctrl.V_prev = None
                G, _, _ = base_continuation(plant, base_ctrl, Hn, ell + 1, T_cap, sens_seed, 0, cfg.hazard)
                snap.restore(base_ctrl)
                J[name] = cfg.hazard * g_now + (1 - cfg.hazard) * G
            Jmin, Jmax = min(J.values()), max(J.values())
            for k in arms:
                raw[k].append(J[k] - Jmin)
                per_arm[k].append((J[k] - Jmin) / (Jmax - Jmin) if Jmax > Jmin else 0.0)
        H = plant.layer(H, A_ref)
        ref_ctrl.commit(ell, A_ref, rec_ref, plant.ctx)
        base_ctrl.commit(ell, A_ref, rec_b, plant.ctx)
        for k, c in arms.items():
            c.commit(ell, A_ref, recs[k], plant.ctx)
    return {k: {"TR": float(np.mean(per_arm[k])) if per_arm[k] else float("nan"), "RC": float(np.mean(raw[k])) if raw[k] else float("nan"),
                "n_states": len(per_arm[k])} for k in arms}


# ============================================================================================
# identification diagnostics (Tables XIX, XX, L; Fig. 2)
# ============================================================================================


@torch.no_grad()
def identification_diagnostics(cfg: Config, plant: Plant, rs: RoundState, ctrl: Controller, n_states: int, seed: int) -> Dict:
    if n_states < 1:
        raise ValueError("Identification diagnostics need a positive state budget")
    groups, used = {}, []
    ids = list(plant.source.partitions["test"]) if plant.is_family else [-1]
    per_trajectory = min(128,max(cfg.L_max*2,8))
    with preserve_plant(plant):
        remaining, trajectory = n_states, 0
        while remaining:
            sample = int(ids[trajectory % len(ids)])
            plant.bind_sample(sample,"test") if plant.is_family else plant.bind_graph(plant.reference_graph)
            budget = min(remaining,per_trajectory)
            result = _identification_diagnostics_one(cfg,plant,rs,ctrl,budget,seed+trajectory*1009)
            for key, values in result.items():
                target = groups.setdefault(key,{"n":0})
                target["n"] += values["n"]
                for field,value in values.items():
                    if field != "n":
                        target[field] = target.get(field,0.)+value*values["n"]
            used.append(sample); remaining -= budget; trajectory += 1
    for values in groups.values():
        for key in list(values):
            if key != "n":
                values[key] /= values["n"]
        values["sample_ids"] = used
        values["scope"] = "held_out_test_graph_states" if plant.is_family else "single_graph_sensing_trajectories"
    return groups


@torch.no_grad()
def _identification_diagnostics_one(cfg: Config, plant: Plant, rs: RoundState, ctrl: Controller, n_states: int, seed: int) -> Dict:
    """True block Jacobian (finite differences), raw pooled Hadamard estimate (same-state block) and the
    corrected (model) response at sampled states of the deployed trajectory, grouped by depth."""
    geom, dev = plant.geom, plant.device
    gen = torch.Generator().manual_seed(seed)
    design = RotatingDesign(geom.du, cfg.Kb)
    plant.prepare(grad=False)
    L = min(128, max(cfg.L_max * 2, 8))
    layers = sorted([0]+(torch.randperm(L-1,generator=gen)[:min(n_states,L)-1]+1).tolist())
    ctrl.role_mask = plant.role_mask
    ctrl.reset(plant.B, plant.ctx)
    H = plant.H0
    sens_seed = 1234 + seed
    groups = {"0": [], "1-3": [], "4-15": [], "16-63": [], ">=64": []}
    sr = [c * (geom.M + 1) for c in range(geom.Kc)]
    for ell in range(max(layers) + 1):
        z_raw = plant.observe(H, sens_seed)
        A, rec = ctrl.step(ell, z_raw, plant.ctx)
        if ell in layers:
            b = rec["b"] if "b" in rec else A
            radii = slack_radii(geom, geom.to_tangent(b), cfg.delta, cfg.eta_s)
            lower, upper, beta = eroded_bounds(geom, radii)
            center = geom.project(b, lower, upper, beta)[0]
            C_true = response_jacobian(plant, H, ell, center, sens_seed=sens_seed)
            u_c = geom.to_tangent(center)
            zp, zm = [], []
            # Use the complete ordered orthogonal block so rows and responses
            # remain aligned even when Kb does not divide the number of rows.
            for row in design.H.to(dev):
                up, um = u_c+radii*row, u_c-radii*row
                zp.append(plant.normalizer(plant.observe(plant.layer(H, geom.from_tangent(up)),sens_seed))[0])
                zm.append(plant.normalizer(plant.observe(plant.layer(H, geom.from_tangent(um)),sens_seed))[0])
            C_raw = pooled_response(torch.stack(zp),torch.stack(zm),design.H.to(dev),radii)
            t = depth_t(ell, cfg.ell0).to(dev).expand(1)
            C_model = rs.model.response(rec["x"], t, torch.tensor([ell == 0], device=dev))[0]
            def spec(C):
                s = torch.linalg.svdvals(C.double())
                nulls = int((s < 1e-9 * max(1.0, float(s.max()))).sum())
                tail = float((s[-4:] ** 2).sum() / (s ** 2).sum().clamp_min(1e-30)) if s.numel() >= 4 else float("nan")
                eps = excitation_profile(C)
                return {"nulls": nulls, "tail": tail, "inactive": int((eps < cfg.tau_exc).sum()), "smin_over_smax": float(s.min() / s.max().clamp_min(1e-30))}
            rel_raw = float((C_raw - C_true).norm() / C_true.norm().clamp_min(1e-12))
            rel_mod = float((C_model - C_true).norm() / C_true.norm().clamp_min(1e-12))
            key = "0" if ell == 0 else "1-3" if ell <= 3 else "4-15" if ell <= 15 else "16-63" if ell <= 63 else ">=64"
            groups[key].append({"ell": ell, "true": spec(C_true), "raw": spec(C_raw) | {"rel_err": rel_raw}, "corrected": spec(C_model) | {"rel_err": rel_mod},
                                "sr_column_energy_true": float((C_true[:, sr] ** 2).sum() / (C_true ** 2).sum().clamp_min(1e-30)),
                                "sr_column_energy_corrected": float((C_model[:, sr] ** 2).sum() / (C_model ** 2).sum().clamp_min(1e-30))})
        H = plant.layer(H, A)
        ctrl.commit(ell, A, rec, plant.ctx)
    summary = {}
    for k, rows in groups.items():
        if not rows:
            continue
        agg = {}
        for obj in ("true", "raw", "corrected"):
            for f in ("nulls", "tail", "inactive", "smin_over_smax", "rel_err"):
                vals = [r[obj].get(f) for r in rows if r[obj].get(f) is not None]
                if vals:
                    agg[f"{obj}_{f}"] = float(np.nanmean(vals))
        agg["n"] = len(rows)
        agg["sr_energy_true"] = float(np.mean([r["sr_column_energy_true"] for r in rows]))
        agg["sr_energy_corrected"] = float(np.mean([r["sr_column_energy_corrected"] for r in rows]))
        summary[k] = agg
    return summary


# ============================================================================================
# ledger and latency
# ============================================================================================


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency(plant: Plant, make_ctrl, L: int, reps: int, warmups: int, sens_seed: int = 0) -> Dict:
    """Complete bound-input inference, including preparation and final readout."""
    from .tasks import predictions
    if reps < 1 or warmups < 0:
        raise ValueError("Latency needs positive repetitions and nonnegative warmups")
    dev = plant.device
    times = []
    for i in range(warmups + reps):
        _sync(dev)
        t0 = time.perf_counter()
        plant.prepare(grad=False)
        ctrl = make_ctrl()
        output = run_pass(plant, ctrl, L, prepared=True, sens_seed=sens_seed)
        predictions(plant.graph, output["H"], plant.bb)
        _sync(dev)
        if i >= warmups:
            times.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": float(np.median(times)), "p10_ms": float(np.quantile(times, 0.1)), "p90_ms": float(np.quantile(times, 0.9)),
            "reps": reps, "L": L, "samples_ms": times, "n_graphs": plant.B,
            "scope": "bound_input_prepare_controller_forward_readout", "union_support": "cached",
            "sample_id": getattr(plant.graph, "sample_id", None)}


@torch.no_grad()
def component_profile(cfg: Config, plant: Plant, ctrl: Controller, L: int, sens_seed: int = 0) -> Dict:
    """Per-layer wall time of sensing, the controller decision (observer, base, planner, execution) and the block."""
    dev = plant.device
    plant.prepare(grad=False)
    ctrl.role_mask = plant.role_mask
    ctrl.reset(plant.B, plant.ctx)
    H = plant.H0
    acc = {"sensing_ms": 0.0, "decision_ms": 0.0, "block_ms": 0.0}
    for ell in range(L):
        _sync(dev); t0 = time.perf_counter()
        z_raw = plant.observe(H, sens_seed) if ctrl.needs_sensing else None
        _sync(dev); t1 = time.perf_counter()
        A, rec = ctrl.step(ell, z_raw, plant.ctx)
        _sync(dev); t2 = time.perf_counter()
        H = plant.layer(H, A)
        _sync(dev); t3 = time.perf_counter()
        ctrl.commit(ell, A, rec, plant.ctx)
        _sync(dev); t4 = time.perf_counter()
        acc["sensing_ms"] += (t1 - t0) * 1e3; acc["decision_ms"] += (t2 - t1 + t4 - t3) * 1e3; acc["block_ms"] += (t3 - t2) * 1e3
    return {k: v / L for k, v in acc.items()}


def mac_ledger(cfg: Config, plant: Plant) -> Dict:
    """Analytic multiply–accumulate counts per decision per graph (Table II rows)."""
    geom = plant.geom
    ns = int(plant.sensed().idx.numel())
    d, b, dz, dx, du, w, H, No, Npd, Kc, M = cfg.d, cfg.sketch_b, cfg.dz, cfg.dx, geom.du, cfg.head_width, cfg.H, cfg.No, cfg.Npd, geom.Kc, geom.M
    dim = H * Kc * (M + 2)
    sensing = ns * d * b + ns * b * b * (1 + Kc)
    observer = dz * dx + dx * dx
    base = cfg.wA * cfg.wA * 3 + cfg.wA * (dx + Kc * (M + 2) + 1) * 3
    rollout = No * H * (dx * dx + dx * du * (1 + cfg.rB) + dz * dx + dx * dx * H * du)
    heads = No * H * (dz * w + w * w + dz * w) * 2
    pdhg = No * Npd * (dim * dim + H * cfg.dy * dim * 2)
    cert = No * (dim * dim + H * cfg.dy * dim)
    proj = No * Npd * H * Kc * 18 * (2 * M + 1)
    execm = dz * w + dx
    planner = rollout + heads + pdhg + cert + proj
    return {"sensing": sensing, "observer": observer, "base_policy": base, "rollout": rollout, "heads": heads, "pdhg": pdhg, "certificates": cert,
            "projections": proj, "execution": execm, "planner_total": planner, "controller_excl_sensing": observer + base + planner + execm,
            "controller_with_sensing": sensing + observer + base + planner + execm, "sensed_nodes": ns, "n_nodes": plant.n, "nnz_union": plant.sup.nnz}


# ============================================================================================
# conservatism, harm, realized gains (Tables XII, XIII, XLIII)
# ============================================================================================


@torch.no_grad()
def conservatism(cfg: Config, plant: Plant, pb: PolicyBundle, env: Envelope, seeds: List[int], n_decisions: int, condition: str) -> Dict:
    """Per decision: true advantage of the executed action, of the full step b+d and of the hard rule."""
    rs = pb.rs
    gen = torch.Generator().manual_seed(cfg.seed + 303)
    rows = []
    obs_noise = 0.2 if condition == "sensor_shift" else 0.0
    L_run = min(cfg.L_a, 2 * cfg.L_max) if condition == "beyond_Lmax" else min(cfg.L_a, cfg.L_max)
    ell_lo = cfg.L_max if condition == "beyond_Lmax" else 0
    hz = cfg.hazard
    for s in seeds:
        if len(rows) >= n_decisions:
            break
        with evaluation_sample(plant,s,"tune"):
            plant.prepare(grad=False)
            ctrl = build_policy_ctrl(cfg, plant, pb, env)
            base_ctrl = build_base_ctrl(cfg, plant, pb)
            first = run_with_snapshots(plant, ctrl, L_run, [], s, obs_noise=obs_noise, noise_seed=s)
            cand = [i for i, r in enumerate(first["recs"]) if i >= ell_lo and "d" in r and float(plant.geom.norm_1sigma(r["b"] + r["d"], r["b"])[0]) > 1e-9]
            if not cand:
                continue
            k = min(len(cand), max(1, n_decisions // max(1, len(seeds))))
            chosen = sorted(cand[int(i)] for i in torch.randperm(len(cand), generator=gen)[:k])
            second = run_with_snapshots(plant, ctrl, max(chosen) + 1, chosen, s, obs_noise=obs_noise, noise_seed=s)
            for ell in chosen:
                ce = contrast_error(cfg, plant, ctrl, base_ctrl, second["snaps"][ell], ell, s, obs_noise=obs_noise, noise_seed=s, extra_full=True)
                if "full" not in ce["G"]:
                    continue
                G = ce["G"]
                adv_exec = (1 - hz) * (G["b"] - G["A"])                   # true advantage of the executed action
                adv_full = (1 - hz) * (G["b"] - G["full"])                # of the full step
                net = ce["delta_hat"] - float(env.L_bar(torch.tensor([ce["r"]]))[0]) * ce["full_norm"]
                hard_exec = net > 0
                adv_hard = adv_full if hard_exec else 0.0
                rows.append({"ell": ell, "rho": ce["rho"], "adv_exec": adv_exec, "adv_full": adv_full, "adv_hard": adv_hard, "hard_exec": hard_exec,
                             "covered": ce["e_abs"] <= float(env.L_bar(torch.tensor([ce["r"]]))[0]) * ce["dnorm"] if ce["dnorm"] > 1e-9 else True,
                             "bin": ce["bin"]})
    if not rows:
        return {"condition": condition, "n": 0, "sample_ids": list(seeds),
                "scope": "tuning_graph_decisions" if plant.is_family else "single_graph_decisions"}
    adv_exec = np.array([r["adv_exec"] for r in rows]); adv_full = np.array([r["adv_full"] for r in rows]); adv_hard = np.array([r["adv_hard"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    beneficial = adv_full > 1e-4
    harmful = adv_exec < 0
    harmful_hard = adv_hard < 0
    out = {"condition": condition, "n": len(rows), "sample_ids": list(seeds),
           "scope": "tuning_graph_decisions" if plant.is_family else "single_graph_decisions", "harmful_share": float(harmful.mean()), "harmful_share_hard": float(harmful_hard.mean()),
           "harm_magnitude_mean": float(-adv_exec[harmful].mean()) if harmful.any() else 0.0,
           "harm_magnitude_p95": float(np.quantile(-adv_exec[harmful], 0.95)) if harmful.any() else 0.0,
           "harm_magnitude_mean_hard": float(-adv_hard[harmful_hard].mean()) if harmful_hard.any() else 0.0,
           "false_rejection_rate": float(((rho == 0) & beneficial).sum() / max(1, beneficial.sum())),
           "partial_share_of_beneficial": float(((rho > 0) & (rho < 1) & beneficial).sum() / max(1, beneficial.sum())),
           "captured_advantage": float(adv_exec[beneficial].sum() / max(1e-12, adv_full[beneficial].sum())) if beneficial.any() else float("nan"),
           "captured_advantage_hard": float(adv_hard[beneficial].sum() / max(1e-12, adv_full[beneficial].sum())) if beneficial.any() else float("nan"),
           "coverage_pooled": float(np.mean([r["covered"] for r in rows])),
           "coverage_worst_bin": min([float(np.mean([r["covered"] for r in rows if r["bin"] == b])) for b in set(r["bin"] for r in rows)]),
           "rho_zero_share": float((rho == 0).mean()), "rho_partial_share": float(((rho > 0) & (rho < 1)).mean())}
    return out


@torch.no_grad()
def realized_gains(cfg: Config, plant: Plant, pb: PolicyBundle, env: Envelope, n_pairs: int, seed: int) -> Dict:
    if n_pairs < 1:
        raise ValueError("Realized gains need a positive pair budget")
    all_gains, all_sensitivity, used = [], [], []
    ids = list(plant.source.partitions["audit"]) if plant.is_family else [-1]
    with preserve_plant(plant):
        budget = max(1, math.ceil(n_pairs/len(ids)))
        for index, sample in enumerate(ids):
            count = min(budget,n_pairs-len(all_gains))
            if count <= 0:
                break
            plant.bind_sample(sample,"audit") if plant.is_family else plant.bind_graph(plant.reference_graph)
            gains,sensitivity = _realized_gains_one(cfg,plant,pb,env,count,seed+index*1009)
            all_gains.extend(gains); all_sensitivity.extend(sensitivity); used.append(int(sample))
    g, sensitivity = np.asarray(all_gains), np.asarray(all_sensitivity)
    return {"n_pairs": len(g), "gain_p50": float(np.median(g)), "gain_p99": float(np.quantile(g,.99)),
            "gain_max": float(g.max()), "action_sensitivity_p99": float(np.quantile(sensitivity,.99)),
            "certificate_level_alpha_min_over_R0": cfg.alpha_min/cfg.R0,
            "sample_ids": used, "scope": "held_out_audit_graph_pairs" if plant.is_family else "single_graph_state_pairs",
            "certified_global_bound": False}


@torch.no_grad()
def _realized_gains_one(cfg: Config, plant: Plant, pb: PolicyBundle, env: Envelope, n_pairs: int, seed: int) -> Dict:
    """Sampled one-step closed-loop gains g_ℓ = ‖ε_{ℓ+1}‖/‖ε_ℓ‖ over nearby augmented-state pairs
    (relative perturbation sizes 10⁻³–10⁻¹), and the directly sampled action sensitivity."""
    rs, geom, dev = pb.rs, plant.geom, plant.device
    gen = torch.Generator().manual_seed(seed)
    plant.prepare(grad=False)
    ctrl = build_policy_ctrl(cfg, plant, pb, env)
    ctrl2 = build_policy_ctrl(cfg, plant, pb, env)
    L = max(1,min(32,cfg.L_max))
    sampled_layers = torch.randint(0,L,(n_pairs,),generator=gen)
    counts = torch.bincount(sampled_layers,minlength=L)
    layers = torch.nonzero(counts,as_tuple=True)[0].tolist()
    ctrl.role_mask = ctrl2.role_mask = plant.role_mask
    ctrl.reset(plant.B, plant.ctx)
    H = plant.H0
    gains, sens = [], []
    sens_seed = 777 + seed
    for ell in range(max(layers) + 1):
        if ell in layers:
            reps = int(counts[ell])
            for _ in range(reps):
                eps = float(10 ** (torch.rand(1, generator=gen) * 2 - 3))
                mem = ControllerSnapshot(ctrl)
                mem.restore(ctrl2)
                Hp = H + eps * H.norm(dim=1, keepdim=True).mean() * torch.randn(H.shape, generator=gen).to(dev) / math.sqrt(H.shape[1])
                Hp = Hp * (cfg.R0/Hp.norm(dim=1,keepdim=True).clamp_min(cfg.R0))
                if ctrl2.x_prior is not None:
                    ctrl2.x_prior = ctrl2.x_prior + eps * rs.x_max * torch.randn(ctrl2.x_prior.shape, generator=gen).to(dev)
                    ctrl2.x_prior *= rs.x_max/ctrl2.x_prior.norm(dim=-1,keepdim=True).clamp_min(rs.x_max)
                ctrl2.eta = torch.clamp(ctrl2.eta + eps * torch.randn(ctrl2.eta.shape, generator=gen).to(dev), -1, 1)
                ctrl2.A_prev = geom.project(ctrl2.A_prev + eps * torch.randn(ctrl2.A_prev.shape, generator=gen).to(dev))[0]
                if ctrl2.V_prev is not None:
                    perturbation = eps*torch.randn(ctrl2.V_prev.shape,generator=gen).to(dev)
                    ctrl2.V_prev = geom.project(ctrl2.V_prev+perturbation)[0]
                d0 = block_distance(plant, cfg, H, Hp, ctrl, ctrl2, rs.x_max)
                z1 = plant.observe(H, sens_seed); z2 = plant.observe(Hp, sens_seed)
                A1, r1 = ctrl.step(ell, z1, plant.ctx)
                A2, r2 = ctrl2.step(ell, z2, plant.ctx)
                H1 = plant.layer(H, A1); H2 = plant.layer(Hp, A2)
                ctrl.commit(ell, A1, r1, plant.ctx); ctrl2.commit(ell, A2, r2, plant.ctx)
                d1 = block_distance(plant, cfg, H1, H2, ctrl, ctrl2, rs.x_max)
                gains.append(d1 / max(d0, 1e-12))
                dH = float((H - Hp).norm(dim=1).max())
                sens.append(float(geom.norm_1sigma(A1, A2)[0]) / max(dH, 1e-12))
                mem.restore(ctrl)
        z = plant.observe(H, sens_seed)
        A, r = ctrl.step(ell, z, plant.ctx)
        H = plant.layer(H, A)
        ctrl.commit(ell, A, r, plant.ctx)
    return gains, sens


# ============================================================================================
# manuscript targets for ogbn-arxiv (projected values of the submission; never measured results)
# ============================================================================================

ARXIV_TARGETS = {
    # (table, quantity, arm) -> manuscript value
    ("XXVIII", "P1_test_accuracy", "A0"): 73.42, ("XXVIII", "P1_test_accuracy", "A1t"): 73.84, ("XXVIII", "P1_test_accuracy", "A2"): 74.18,
    ("XXVIII", "P1_test_accuracy", "ROPC-A"): 74.17, ("XXVIII", "P1_test_accuracy", "ROPC"): 74.24,
    ("XXVII", "P2_DA", "A0"): 72.34, ("XXVII", "P2_PDD", "A0"): 2.71, ("XXVII", "P2_DA", "A1t"): 73.08, ("XXVII", "P2_PDD", "A1t"): 0.55,
    ("XXVII", "P2_DA", "A2"): 73.37, ("XXVII", "P2_PDD", "A2"): 0.26, ("XXVII", "P2_DA", "base"): 73.34, ("XXVII", "P2_PDD", "base"): 0.27,
    ("XXVII", "P2_DA", "ROPC-A"): 73.37, ("XXVII", "P2_PDD", "ROPC-A"): 0.28, ("XXVII", "P2_DA", "amortized_raw"): 73.36, ("XXVII", "P2_PDD", "amortized_raw"): 0.31,
    ("XXVII", "P2_DA", "student_tw"): 73.39, ("XXVII", "P2_PDD", "student_tw"): 0.23, ("XXVII", "P2_DA", "C1"): 73.34, ("XXVII", "P2_PDD", "C1"): 0.26,
    ("XXVII", "P2_DA", "Cr1"): 73.38, ("XXVII", "P2_PDD", "Cr1"): 0.23, ("XXVII", "P2_DA", "ROPC"): 73.44, ("XXVII", "P2_PDD", "ROPC"): 0.19,
    ("XXX", "latency_ms_L_star", "A0"): 36.0, ("XXX", "latency_ms_L_star", "ROPC-A"): 37.4, ("XXX", "latency_ms_L_star", "ROPC"): 37.9,
    ("XXX", "latency_ms_L_star", "compiled"): 36.0,
    ("LVIII", "drift_online_sigma_0.0", "ROPC"): 74.24, ("LVIII", "drift_compiled_sigma_0.0", "ROPC"): 74.24, ("LVIII", "drift_online_sigma_0.01", "ROPC"): 74.19,
    ("LVIII", "drift_compiled_sigma_0.01", "ROPC"): 74.17, ("LVIII", "drift_online_sigma_0.05", "ROPC"): 73.99, ("LVIII", "drift_compiled_sigma_0.05", "ROPC"): 73.86,
    ("LVIII", "drift_online_sigma_0.1", "ROPC"): 73.60, ("LVIII", "drift_compiled_sigma_0.1", "ROPC"): 73.26,
    ("XL", "coverage_pooled", "ROPC"): 0.95, ("XL", "coverage_worst_bin", "ROPC"): 0.93, ("XL", "harmful_share", "ROPC"): 0.0006,
    ("XL", "gain_p99", "ROPC"): 0.984, ("XIII", "action_sensitivity_p99", "ROPC"): 0.031, ("XIII", "gain_p50", "ROPC"): 0.95,
    ("XLIII", "false_rejection_rate", "ROPC"): 0.061, ("XLIII", "partial_share_of_beneficial", "ROPC"): 0.142, ("XLIII", "captured_advantage", "ROPC"): 0.84,
    ("XLIX", "rho_full_share_in_support", "ROPC"): 0.78, ("XLIX", "authority_p50", "ROPC"): 0.058, ("XLIX", "authority_p90", "ROPC"): 0.142,
    ("XLIX", "rho_zero_share", "ROPC"): 0.060, ("XLIX", "rho_partial_share", "ROPC"): 0.160, ("XLIX", "dnorm_over_dmax_median", "ROPC"): 0.31,
    ("L", "nmse1", "ROPC"): 0.41, ("L", "rank_corr", "ROPC"): 0.90, ("L", "order_accuracy", "ROPC"): 0.84, ("L", "tail", "ROPC"): 0.046,
    ("XXIII", "equivalence_margin_ROPC_vs_ROPC-A", "ROPC"): 0.2, ("XXIII", "equivalence_margin_ROPC_vs_A2", "ROPC"): 0.2,
    ("XXX", "training_gpu_hours", "A0"): 1.90, ("XXX", "training_gpu_hours", "ROPC"): 4.42, ("XXX", "training_gpu_hours", "ROPC-A"): 4.73,
    ("XXV", "distilled_share_of_depth_area_gain", "A2"): 0.94,
}

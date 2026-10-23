"""Task training, intervention collection and decision-focused identification.

Implements data-separated supervision, graph-family mini batches, stage/value
heads, response fitting, recurrent amortization and resumable task checkpoints.
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (ActionGeometry, Backbone, GraphData, KERNEL_NAMES, Normalizer, Sensor, SensedSet, SplineSchedule,
                        UnionSupport, build_sensed_set, effective_sample_size, exact_spectral_norm)
from .control import (BasePolicyGRU, ConvexHead, Controller, Envelope, HealthRegion, Planner, PlannerConfig,
                          ReliabilityScorer, RotatingDesign, SupportDistance, TransitionModel, depth_t, eroded_bounds,
                          excitation_profile, pooled_response, ridge_weights, slack_radii)

# ============================================================================================
# configuration (Appendix A) and scale presets
# ============================================================================================


@dataclass
class Config:
    dataset: str = "ogbn-arxiv"
    seed: int = 0
    scale: str = "full"
    mode: str = "joint"                 # joint (deployed) or fixed backbone (Lemma 7(a) exact)
    batch_size: int = 0                  # 0: use the dataset's Table IX batch size
    subset_nodes: int = 50000
    split_index: int = 0
    structural_samples: int = 65536
    # architecture
    d: int = 128
    dff: int = 128
    Kf: int = 16
    Kc: int = 4
    support_s: int = 2
    gamma_sl: float = 1.0
    alpha_min: float = 0.05
    alpha_max: float = 0.40
    b_min: float = 1e-3
    R0: float = 3.0
    eps_g: float = 0.5
    role_hidden: int = 64
    bank: str = "P0,Pr0,Pg,Pf"          # Table IX, ogbn-arxiv; 'auto' selects from the candidate bank
    candidate_bank: str = "P0,Pr0,Pf,Pc,Px,Pg"
    r_min: float = 2.0
    L_star: int = 32
    L_max: int = 64
    ref_knots: int = 2
    free_knots: int = 8
    # sensing / model
    sketch_b: int = 16
    n_sens: int = 1024
    dy: int = 3
    rB: int = 4
    gamma_A: float = 0.96
    gamma_B: float = 0.08
    Ro_max: float = 1.0
    sigma_min_C: float = 0.35
    ess_min: float = 16.0
    # planner
    H: int = 3
    No: int = 3
    Npd: int = 8
    delta_tr: float = 0.025
    beta_w: float = 0.5
    lam_delta: float = 0.02
    lam_v: float = 0.05
    lam_V: float = 0.3
    lam_sup: float = 0.5
    lam_psi: float = 4.0
    sup_quantile: float = 0.95
    L_V: float = 4.0
    eps_P: float = 1e-4
    ell0: float = 8.0
    # value / execution / audit
    hazard: float = 1.0 / 32
    T_cap: int = 320
    head_width: int = 64
    K_rounds: int = 3
    kappa_stop: float = 0.025
    wA: int = 48
    N_DA: int = 4
    eps_A: float = 1e-3
    q_t: float = 0.95
    q_c: float = 0.90
    L_a: int = 256
    K_lib: int = 8
    value_slice: float = 0.15
    continuation_frac: float = 0.25
    # identification
    Kb: int = 8
    delta: float = 0.01
    eta_s: float = 0.5
    tau_exc: float = 0.05
    kappa_max: float = 100.0
    lam0: float = 1e-3
    lam_adv: float = 1.0
    lam_p: float = 1.0
    # training
    lr: float = 1e-3
    wd: float = 1e-5
    lam_struct: float = 0.1
    epochs_warmup: int = 40
    warmup_ckpts: str = "5,10,20"
    epochs_openloop: int = 1500
    epochs_round: int = 150
    epochs_distill: int = 600
    epochs_heads: int = 400
    epochs_ident: int = 400
    epochs_dagger: int = 200
    branch_traj: int = 24
    states_per_traj: int = 32
    p_explore: float = 0.3
    explore_radius: float = 0.05
    dagger_traj: int = 16
    tune_samples: int = 32
    audit_samples: int = 128
    reserve_samples: int = 100
    ckpt_perturb: float = 1e-3
    audit_decisions_per_sample: int = 16
    audit_all_decisions: int = 1
    regret_states: int = 512
    diag_states: int = 256
    gain_pairs: int = 2000
    harm_decisions: int = 128
    latency_reps: int = 1000
    latency_warmups: int = 200
    val_every: int = 10
    checkpoint_layers: int = 1
    grad_ckpt_min_nodes: int = 20000
    variants: str = "ropc,ropc_a,cr1,c1"
    students: str = "tw"

    @property
    def dz(self) -> int:
        return 5 + 5 * self.Kc

    @property
    def bank_list(self) -> List[str]:
        return [k for k in self.bank.split(",") if k]

    @property
    def M(self) -> int:
        return len(self.bank_list)

    @property
    def dx(self) -> int:
        return self.Kc * (self.M + 1)

    @property
    def dc(self) -> int:
        return self.Kc + 3


SCALE_PRESETS = {
    "full": {},
    "pilot": dict(epochs_warmup=12, warmup_ckpts="2,4,8", epochs_openloop=150, epochs_round=30, epochs_distill=100, epochs_heads=200,
                  epochs_ident=200, epochs_dagger=100, branch_traj=6, states_per_traj=16, dagger_traj=4, T_cap=96, tune_samples=8,
                  audit_samples=16, reserve_samples=12, audit_decisions_per_sample=6, regret_states=32, diag_states=32, gain_pairs=100,
                  harm_decisions=12, latency_reps=50, latency_warmups=10, L_a=64, N_DA=2, val_every=5),
    "preflight": dict(epochs_warmup=2, warmup_ckpts="1", epochs_openloop=3, epochs_round=2, epochs_distill=3, epochs_heads=30,
                      epochs_ident=30, epochs_dagger=20, branch_traj=2, states_per_traj=8, dagger_traj=2, T_cap=12, tune_samples=3,
                      audit_samples=4, reserve_samples=3, audit_decisions_per_sample=3, regret_states=4, diag_states=4, gain_pairs=10,
                      harm_decisions=4, latency_reps=3, latency_warmups=1, L_a=12, N_DA=1, L_max=8, L_star=4, K_rounds=2, val_every=1),
}


def make_config(scale: str, overrides: Optional[Dict] = None) -> Config:
    if scale not in SCALE_PRESETS:
        raise ValueError(f"Unknown scale {scale!r}; expected {sorted(SCALE_PRESETS)}")
    cfg = Config()
    for k, v in SCALE_PRESETS[scale].items():
        setattr(cfg, k, v)
    cfg.scale = scale
    for k, v in (overrides or {}).items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown configuration field {k!r}")
        if v is not None:
            setattr(cfg, k, v)
    if not 1 <= cfg.sketch_b <= cfg.d:
        raise ValueError("Require 1 <= sketch_b <= d")
    if cfg.H < 1 or cfg.T_cap < cfg.H or not 0 < cfg.hazard <= 1:
        raise ValueError("Require H >= 1, T_cap >= H and 0 < hazard <= 1")
    if not 0 < cfg.value_slice < 1 or cfg.L_max < 1 or cfg.L_star < 1:
        raise ValueError("Invalid value fraction or depth")
    if cfg.mode not in ("joint", "fixed"):
        raise ValueError("mode must be joint or fixed")
    return cfg


def geometry_of(cfg: Config) -> ActionGeometry:
    return ActionGeometry(cfg.M, cfg.Kc, cfg.alpha_min, cfg.alpha_max, cfg.r_min)


def planner_config(cfg: Config, H: Optional[int] = None) -> PlannerConfig:
    return PlannerConfig(H=H or cfg.H, No=cfg.No, Npd=cfg.Npd, delta_tr=cfg.delta_tr, beta_w=cfg.beta_w, lam_delta=cfg.lam_delta,
                         lam_v=cfg.lam_v, lam_V=cfg.lam_V, lam_sup=cfg.lam_sup, L_V=cfg.L_V, eps_P=cfg.eps_P, hazard=cfg.hazard, ell0=cfg.ell0)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, str(path))


def _json_default(o):
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================================
# datasets
# ============================================================================================


def synthetic_graph(n: int = 240, F_in: int = 12, n_classes: int = 5, n_comm: int = 4, seed: int = 0) -> GraphData:
    """Planted-partition graph with community-dependent features and labels (semantic preflight)."""
    g = torch.Generator().manual_seed(seed)
    comm = torch.randint(0, n_comm, (n,), generator=g)
    centers = torch.randn(n_comm, F_in, generator=g)
    x = centers[comm] + 0.8 * torch.randn(n, F_in, generator=g)
    y = ((comm * 2 + (x[:, 0] > 0).long()) % n_classes)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = 0.12 if comm[i] == comm[j] else 0.012
            if torch.rand(1, generator=g).item() < p:
                src.append(j)
                dst.append(i)
    ei = torch.tensor([src, dst], dtype=torch.long)
    perm = torch.randperm(n, generator=g)
    ntr, nva = int(0.6 * n), int(0.2 * n)
    return GraphData(name="synthetic", x=x, y=y, edge_index=ei, num_nodes=n, num_classes=n_classes,
                     train_idx=perm[:ntr], valid_idx=perm[ntr:ntr + nva], test_idx=perm[ntr + nva:])


def _pick(d: Dict, *names):
    for k in names:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_cached_graph(args) -> GraphData:
    """Legacy arxiv-only cache reader; all dataset loading belongs to datasets.py."""
    name = args.dataset_name
    if name != "ogbn-arxiv":
        raise ValueError("Legacy cached-graph loading is arxiv-only; use ropc.datasets.load_dataset for other tasks")
    npz_path = Path(args.ropc_graph_npz) if args.ropc_graph_npz else None
    if npz_path is not None and npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        return GraphData(name=name, x=torch.from_numpy(z["x"]).float(), y=torch.from_numpy(z["y"]).long(),
                         edge_index=torch.from_numpy(z["edge_index"]).long(), num_nodes=int(z["num_nodes"]),
                         num_classes=int(z["num_classes"]), train_idx=torch.from_numpy(z["train_idx"]).long(),
                         valid_idx=torch.from_numpy(z["valid_idx"]).long(), test_idx=torch.from_numpy(z["test_idx"]).long())
    gd = None
    pt = Path(args.graph_pt) if args.graph_pt else None
    if pt is not None and pt.exists():
        try:
            obj = torch.load(pt, map_location="cpu", weights_only=False)
            gd = graph_from_object(obj, name)
            log(f"[DATA] loaded {pt} ({gd.num_nodes} nodes, {gd.edge_index.shape[1]} arcs)")
        except Exception as ex:  # noqa: BLE001
            log(f"[DATA] graph.pt unusable ({ex!r}); trying the next source")
    if gd is None and args.rtt_graph_npz and Path(args.rtt_graph_npz).exists():
        try:
            z = np.load(args.rtt_graph_npz, allow_pickle=False)
            gd = graph_from_object({k: z[k] for k in z.files}, name)
            log(f"[DATA] loaded RTT cache {args.rtt_graph_npz}")
        except Exception as ex:  # noqa: BLE001
            log(f"[DATA] RTT npz unusable ({ex!r}); trying OGB")
    if gd is None:
        from ogb.nodeproppred import NodePropPredDataset  # type: ignore
        ds = NodePropPredDataset(name=name, root=args.dataset_root)
        graph, labels = ds[0]
        split = ds.get_idx_split()
        gd = GraphData(name=name, x=torch.from_numpy(graph["node_feat"]).float(), y=torch.from_numpy(labels.reshape(-1)).long(),
                       edge_index=torch.from_numpy(graph["edge_index"]).long(), num_nodes=int(graph["num_nodes"]),
                       num_classes=int(labels.max()) + 1, train_idx=torch.from_numpy(split["train"]).long(),
                       valid_idx=torch.from_numpy(split["valid"]).long(), test_idx=torch.from_numpy(split["test"]).long())
    if npz_path is not None:
        ensure_dir(npz_path.parent)
        np.savez(npz_path, x=gd.x.numpy(), y=gd.y.numpy(), edge_index=gd.edge_index.numpy(), num_nodes=gd.num_nodes,
                 num_classes=gd.num_classes, train_idx=gd.train_idx.numpy(), valid_idx=gd.valid_idx.numpy(), test_idx=gd.test_idx.numpy())
        log(f"[DATA] wrote ROPC graph cache {npz_path}")
    return gd


def graph_from_object(obj, name: str, split_index: int = 0) -> GraphData:
    """Accepts a dict (keys x/y/edge_index and split indices or masks), a PyG-like object with those
    attributes, or an OGB-style (graph_dict, labels) pair."""
    if name != "ogbn-arxiv":
        raise ValueError("Legacy graph_from_object is arxiv-only; use the task-aware dataset adapters")
    if isinstance(obj, (tuple, list)) and len(obj) == 2 and isinstance(obj[0], dict):
        gdict, labels = obj
        d = {"x": gdict.get("node_feat"), "y": labels, "edge_index": gdict.get("edge_index"), "num_nodes": gdict.get("num_nodes"),
             "node_year": gdict.get("node_year")}
    elif isinstance(obj, dict):
        d = dict(obj)
    else:
        d = {k: getattr(obj, k, None) for k in ("x", "y", "edge_index", "train_idx", "valid_idx", "test_idx", "train_mask", "val_mask",
                                                  "test_mask", "num_nodes", "node_year", "split_idx")}
        if hasattr(obj, "train_mask") and d.get("train_mask") is None:
            d["train_mask"] = obj.train_mask

    def T(v, dtype):
        if v is None:
            return None
        v = torch.as_tensor(np.asarray(v) if not torch.is_tensor(v) else v)
        return v.to(dtype)

    x = T(_pick(d, "x", "node_feat", "features", "feat", "X"), torch.float32)
    y = T(_pick(d, "y", "labels", "label", "node_label", "Y"), torch.long)
    if y is not None:
        if y.ndim > 2 or (y.ndim == 2 and y.shape[1] != 1):
            raise ValueError("The legacy arxiv cache requires one categorical label per node")
        y = y.reshape(-1)
    ei = T(_pick(d, "edge_index", "edges", "edge_list", "adj_t"), torch.long)
    if ei is not None and ei.shape[0] != 2:
        ei = ei.t()
    n = int(d.get("num_nodes") or x.shape[0])
    tr = T(_pick(d, "train_idx", "train", "idx_train", "train_nodes"), torch.long)
    va = T(_pick(d, "valid_idx", "val_idx", "valid", "val", "idx_val", "idx_valid", "valid_nodes"), torch.long)
    te = T(_pick(d, "test_idx", "test", "idx_test", "test_nodes"), torch.long)
    split = d.get("split_idx")
    if (tr is None) and isinstance(split, dict):
        tr, va, te = T(split.get("train"), torch.long), T(split.get("valid"), torch.long), T(split.get("test"), torch.long)
    if tr is None and _pick(d, "train_mask") is not None:
        def mask_indices(value):
            mask = T(value, torch.bool)
            if mask is None:
                raise ValueError("All three split masks are required")
            if mask.ndim == 2:
                if not 0 <= split_index < mask.shape[1]:
                    raise ValueError("Split index is outside the available official masks")
                mask = mask[:, split_index]
            if mask.ndim != 1:
                raise ValueError("Split masks must have shape [N] or [N, splits]")
            return mask.nonzero(as_tuple=True)[0]
        tr = mask_indices(_pick(d, "train_mask"))
        va = mask_indices(_pick(d, "val_mask", "valid_mask"))
        te = mask_indices(_pick(d, "test_mask"))
    if tr is None and _pick(d, "node_year") is not None and name == "ogbn-arxiv":
        yr = T(_pick(d, "node_year"), torch.long).reshape(-1)
        tr, va, te = (yr <= 2017).nonzero(as_tuple=True)[0], (yr == 2018).nonzero(as_tuple=True)[0], (yr >= 2019).nonzero(as_tuple=True)[0]
    if tr is None or va is None or te is None:
        raise ValueError("no split indices found in the cached graph object")
    return GraphData(name=name, x=x, y=y, edge_index=ei, num_nodes=n, num_classes=int(y.max()) + 1, train_idx=tr, valid_idx=va, test_idx=te)


@dataclass
class Partition:
    """Training labels split into the task route and the value slice; tuning/audit/reserve samples are
    (sensing seed, checkpoint-perturbation seed) pairs on a single graph (Appendix B)."""
    task_idx: torch.Tensor
    value_idx: torch.Tensor
    tune_seeds: List[int]
    audit_seeds: List[int]
    reserve_seeds: List[int]


def make_partition(graph: GraphData, cfg: Config) -> Partition:
    source = getattr(graph, "source", None)
    if source is not None and getattr(source, "is_family", False):
        # The source splits whole graphs. Local indices therefore cover all
        # supervised items of this graph/batch without splitting its labels.
        ids = graph.train_idx
        value_ids = getattr(graph, "value_idx", None)
        def samples(name, limit):
            result = [int(i) for i in source.partitions[name]]
            return result if cfg.scale == "full" else result[:limit]
        return Partition(ids, ids if value_ids is None else value_ids,
                         samples("tune", cfg.tune_samples), samples("audit", cfg.audit_samples),
                         samples("reserve", cfg.reserve_samples))
    predefined_task = getattr(graph, "task_idx", None)
    predefined_value = getattr(graph, "value_idx", None)
    if predefined_task is not None and predefined_value is not None:
        return Partition(predefined_task, predefined_value,
                         [10_000+i for i in range(cfg.tune_samples)],
                         [20_000+i for i in range(cfg.audit_samples)],
                         [30_000+i for i in range(cfg.reserve_samples)])
    g = torch.Generator().manual_seed(1000 + cfg.seed)
    perm = graph.train_idx[torch.randperm(graph.train_idx.numel(), generator=g).to(graph.train_idx.device)]
    if perm.numel() < 2:
        raise ValueError("At least two training items are needed for disjoint task/value slices")
    nv = max(1, min(perm.numel()-1, int(round(cfg.value_slice * perm.numel()))))
    return Partition(task_idx=perm[nv:].sort().values, value_idx=perm[:nv].sort().values,
                     tune_seeds=[10_000 + i for i in range(cfg.tune_samples)],
                     audit_seeds=[20_000 + i for i in range(cfg.audit_samples)],
                     reserve_seeds=[30_000 + i for i in range(cfg.reserve_samples)])


# ============================================================================================
# plant: backbone + graph + support, forward passes
# ============================================================================================


class Plant:
    def __init__(self, cfg: Config, bb: Backbone, graph: GraphData, sup: UnionSupport, part: Partition, device, sens_seed: int = 0):
        self.cfg, self.bb, self.device = cfg, bb, torch.device(device)
        self.geom = bb.geom
        self.sensor = Sensor(cfg.d, cfg.sketch_b, cfg.Kc, seed=777, device=device)
        self.sens_seed = sens_seed
        self._sensed_cache: Dict[int, SensedSet] = {}
        self.normalizer = Normalizer(cfg.dz)
        self.loss_bar = 1.0
        self.source = getattr(graph, "source", None)
        self.reference_graph = graph
        self.bind_graph(graph, sup=sup, part=part)

    @property
    def is_family(self) -> bool:
        return self.source is not None and bool(getattr(self.source, "is_family", False))

    def bind_graph(self, graph: GraphData, *, sup=None, part=None):
        """Bind one graph or a disjoint family minibatch to the shared backbone.

        Derived graph state is rebuilt rather than reusing caches from another
        batch. Normalization and the task-loss scale remain training-frozen.
        """
        graph = graph.to(self.device)
        if getattr(graph, "source", None) is not None:
            self.source = graph.source
        elif self.source is not None:
            graph.source = self.source
        self.graph = graph
        self.sup = sup if sup is not None else UnionSupport(graph.edge_index, graph.num_nodes)
        self.part = part if part is not None else make_partition(graph, self.cfg)
        self.n, self.B, self.batch = graph.num_nodes, graph.num_graphs, graph.batch_vec
        self._sensed_cache.clear()
        self.H0 = None
        deg_in, deg_out = graph.degrees()
        counts = torch.bincount(self.batch, minlength=self.B).to(graph.x.dtype).clamp_min(1)
        self.node_counts = counts
        degree_features = torch.stack([torch.log1p(deg_in), torch.log1p(deg_out)], 1)
        self.logdeg_mean = torch.zeros(self.B, 2, device=self.device).index_add_(0, self.batch, degree_features) / counts[:, None]
        return self

    def bind_sample(self, sample_id: int, split: Optional[str] = None):
        if not self.is_family:
            return self
        return self.bind_graph(self.source.sample(int(sample_id), split=split))

    def item_graph_ids(self, indices=None):
        task_type = getattr(self.graph, "task_type", "node_classification")
        if task_type.startswith("graph_"):
            gids = torch.arange(self.B, device=self.device)
        elif task_type == "link_prediction":
            gids = self.batch[self.graph.edge_label_index[0]]
        else:
            gids = self.batch
        return gids if indices is None else gids[indices]

    # ---- per-pass preparation
    def sensed(self, seed: Optional[int] = None) -> SensedSet:
        s = self.sens_seed if seed is None else seed
        if s not in self._sensed_cache:
            self._sensed_cache[s] = build_sensed_set(self.graph, self.cfg.n_sens, s, gamma_sl=self.cfg.gamma_sl)
        return self._sensed_cache[s]

    def prepare(self, grad: bool = False):
        with torch.set_grad_enabled(grad):
            self.H0 = self.bb.encode(self.graph)
            self.roles = self.bb.role_inference(self.graph)
            self.kvals = self.bb.kernel_bank(self.sup, self.roles)
            Tc = self.roles["Theta_c"]
            total = torch.zeros(self.B, self.cfg.Kc, device=self.device).index_add_(0, self.batch, Tc)
            squares = torch.zeros_like(total).index_add_(0, self.batch, Tc.detach().square())
            mass = total / self.node_counts[:, None]
            ess = total.detach().square() / squares.clamp_min(1e-12)
            self.role_mask = (ess >= self.cfg.ess_min).float()
            self.ctx = torch.cat([mass.detach(), self.node_counts.log()[:, None] / math.log(1e6), self.logdeg_mean], 1)
        self.ctx = self.ctx.detach()

    def node_actions(self, A: torch.Tensor) -> torch.Tensor:
        """A (B,Kc,p) → (n,p): a_i = Σ_c θ^c_ic a_c, masked roles execute their own share (base action set by caller)."""
        Tc = self.roles["Theta_c"]
        if self.B == 1:
            return Tc @ A[0]
        return torch.einsum("nc,ncp->np", Tc, A[self.batch])

    def layer(self, H: torch.Tensor, A: torch.Tensor, ckpt: bool = False) -> torch.Tensor:
        na = self.node_actions(A)
        if ckpt and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._layer_fn, H, na, use_reentrant=False)
        return self._layer_fn(H, na)

    def _layer_fn(self, H, na):
        return self.bb.layer(H, self.H0, na, self.sup, self.kvals, self.roles, self.batch, self.B)

    @torch.no_grad()
    def observe(self, H: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        from .tasks import node_uncertainty
        s = self.sensed(seed)
        logits = self.bb.readout(H.detach()[s.idx])
        uncertainty = node_uncertainty(self.graph, H.detach(), self.bb)[s.idx]
        return self.sensor.observe(H.detach(), self.H0.detach(), self.roles["Theta_c"].detach(), logits, s, self.B,
                                   self.role_mask, uncertainty=uncertainty)

    @torch.no_grad()
    def stage_loss(self, H: torch.Tensor) -> torch.Tensor:
        """g(H) = min{ℓ_task(f_out(H))/ℓ̄, 1} on the value slice (B,)."""
        from .tasks import task_loss
        idx = self.part.value_idx
        losses = task_loss(self.graph, H, self.bb, indices=idx, reduction="none")
        gids = self.item_graph_ids(idx)
        valid = torch.isfinite(losses)
        total = torch.zeros(self.B, device=H.device).index_add_(0, gids[valid], losses[valid])
        count = torch.zeros_like(total).index_add_(0, gids[valid], torch.ones_like(losses[valid]))
        if (count == 0).any():
            raise ValueError("Every value graph must contain at least one supervised item")
        return torch.clamp(total / count / max(float(self.loss_bar), 1e-12), min=0.0, max=1.0)

    @torch.no_grad()
    def raw_value_loss(self, H: torch.Tensor) -> float:
        from .tasks import task_loss
        return float(task_loss(self.graph, H, self.bb, indices=self.part.value_idx))

    @torch.no_grad()
    def accuracy(self, H: torch.Tensor, idx: torch.Tensor) -> float:
        """Compatibility alias for the task's official metric (not always accuracy)."""
        from .tasks import metric_value
        return metric_value(self.graph, H, self.bb, indices=idx)

    def task_loss(self, H: torch.Tensor) -> torch.Tensor:
        from .tasks import task_loss
        return task_loss(self.graph, H, self.bb, indices=self.part.task_idx)

    def use_ckpt(self) -> bool:
        return bool(self.cfg.checkpoint_layers) and self.n >= self.cfg.grad_ckpt_min_nodes


def make_controller(cfg: Config, plant: Plant, mode: str, model=None, value=None, stage=None, health=None, support=None,
                    envelope: Optional[Envelope] = None, scorer=None, base=None, student=None, schedule=None, trace=None,
                    gate_value=None, planner_over_reference: bool = False, H: Optional[int] = None, x_max: float = 10.0) -> Controller:
    return Controller(mode, plant.geom, planner_config(cfg, H), model, value, stage, health, support, envelope or Envelope((0.0,)),
                      cfg.eps_A, scorer, plant.normalizer, plant.bb.ref_schedule, base=base, student=student, schedule=schedule,
                      trace=trace, wA=cfg.wA, x_max=x_max, gate_value=gate_value, planner_over_reference=planner_over_reference)


def run_pass(plant: Plant, ctrl: Controller, L: int, *, grad: bool = False, readout_layers: Optional[List[int]] = None,
             record: bool = False, sens_seed: Optional[int] = None, prepared: bool = False, H_start: Optional[torch.Tensor] = None,
             ell_start: int = 0, stage_losses: bool = False, train_mix: bool = False, readout_splits=("valid",)):
    """Run layers ℓ = ell_start … ell_start+L−1.  Returns dict with H (final), per-layer readouts (accuracy on
    valid/test) at readout_layers, g_ℓ stage losses if requested, and the controller records."""
    with torch.set_grad_enabled(grad):
        return _run_pass(plant, ctrl, L, grad=grad, readout_layers=readout_layers, record=record, sens_seed=sens_seed, prepared=prepared,
                         H_start=H_start, ell_start=ell_start, stage_losses=stage_losses, train_mix=train_mix,
                         readout_splits=readout_splits)


def _run_pass(plant: Plant, ctrl: Controller, L: int, *, grad: bool, readout_layers, record, sens_seed, prepared, H_start, ell_start,
              stage_losses, train_mix, readout_splits=("valid",)):
    if not prepared:
        plant.prepare(grad=grad)
    if H_start is None:
        ctrl.reset(plant.B, plant.ctx)
        H = plant.H0
    else:
        H = H_start
    out = {"readouts": {}, "g": [], "A": []}
    rl = set(readout_layers or [])
    ctrl.role_mask = plant.role_mask
    for j in range(L):
        ell = ell_start + j
        if stage_losses:
            # A T-term continuation always starts at its entry state, H_ell.
            out["g"].append(float(plant.stage_loss(H).mean()))
        if ctrl.needs_sensing:
            with torch.no_grad():
                z_raw = plant.observe(H, sens_seed)
            A, rec = ctrl.step(ell, z_raw, plant.ctx)
        else:
            A, rec = ctrl.step(ell, None, plant.ctx)
        if train_mix and ctrl.mode == "ropc" and ctrl.base is None and grad:
            # the task route sees (1 − sg ρ) b + sg ρ v* with b the trainable reference proposal
            b_grad = plant.bb.ref_schedule(depth_t(ell, ctrl.cfg.ell0).to(A.device)).expand_as(A)
            share = 1 - plant.role_mask[..., None] * rec["rho"].detach().reshape(-1, 1, 1)
            A = A.detach() + share * (b_grad - b_grad.detach())
        elif ctrl.mode == "schedule" and ctrl.schedule is not None and grad:
            A = ctrl.schedule(depth_t(ell, ctrl.cfg.ell0).to(A.device)).expand_as(A)
        elif ctrl.mode == "schedule" and ctrl.schedule is None and grad:
            A = plant.bb.ref_schedule(depth_t(ell, ctrl.cfg.ell0).to(A.device)).expand_as(A)
        H = plant.layer(H, A, ckpt=plant.use_ckpt() and grad)
        ctrl.commit(ell, A.detach(), rec, plant.ctx, record=record)
        if (ell + 1) in rl:
            out["readouts"][ell + 1] = {
                split: plant.accuracy(H, getattr(plant.graph, split+"_idx")) for split in readout_splits}
        out["A"].append(A.detach())
    out["H"] = H
    return out


def anytime_objective(g: List[float], hazard: float) -> float:
    """J = Σ_{k≥0} h(1−h)^k g(H_k) over the recorded stage losses (truncated)."""
    return float(sum(hazard * (1 - hazard) ** k * gk for k, gk in enumerate(g)))


def continuation_value(g_future: List[float], hazard: float) -> float:
    """V_{ℓ+1} target: Σ_{k'≥0} h(1−h)^{k'} g(H_{ℓ+1+k'})."""
    return anytime_objective(g_future, hazard)


@contextlib.contextmanager
def preserve_plant(plant: Plant):
    """Restore the bound graph even when sampling or evaluation raises."""
    graph, support, partition = plant.graph, plant.sup, plant.part
    try:
        yield
    finally:
        plant.bind_graph(graph, sup=support, part=partition)


def family_batch_size(plant: Plant) -> int:
    return int(plant.cfg.batch_size or getattr(plant.source, "batch_size", 1))


@torch.no_grad()
def evaluate_depths(plant: Plant, make_ctrl, depths, split="valid", sens_seed=0) -> Dict[int, float]:
    """Evaluate a complete official split, concatenating predictions before metrics.

    AP, ROC-AUC, F1 and log-MSE are not averaged over minibatches. PCQM uses
    the official arithmetic mean of per-graph filtered MRR, not a query pool.
    """
    from .tasks import predictions, metric_from_predictions, filtered_mrr
    depths = sorted(set(int(d) for d in depths))
    if not depths or min(depths) < 1:
        raise ValueError("Evaluation depths must be positive")
    values = {d: {"pred": [], "target": [], "mrr": []} for d in depths}
    spec_graph = plant.graph
    with preserve_plant(plant):
        if plant.is_family:
            bs = 1 if getattr(plant.graph, "task_type", "") == "link_prediction" else family_batch_size(plant)
            graphs = plant.source.iter_batches(split, bs, epoch=0, shuffle=False)
        else:
            graphs = [plant.reference_graph]
        count = 0
        for graph in graphs:
            count += 1
            plant.bind_graph(graph)
            plant.prepare(grad=False)
            ctrl = make_ctrl()
            ctrl.reset(plant.B, plant.ctx)
            ctrl.role_mask = plant.role_mask
            H = plant.H0
            for ell in range(max(depths)):
                raw = plant.observe(H, sens_seed) if ctrl.needs_sensing else None
                action, record = ctrl.step(ell, raw, plant.ctx)
                H = plant.layer(H, action)
                ctrl.commit(ell, action, record, plant.ctx)
                if ell+1 not in values:
                    continue
                target = values[ell+1]
                if getattr(graph, "task_type", "") == "link_prediction":
                    target["mrr"].append(float(filtered_mrr(plant.graph, H, plant.bb)))
                else:
                    pred = predictions(plant.graph, H, plant.bb)
                    truth = plant.graph.y
                    if not plant.is_family:
                        idx = getattr(plant.graph, split+"_idx")
                        pred, truth = pred[idx], truth[idx]
                    target["pred"].append(pred.detach().cpu())
                    target["target"].append(truth.detach().cpu())
        if not count:
            raise ValueError(f"Dataset partition {split!r} is empty")
    result = {}
    for d, rows in values.items():
        result[d] = (float(np.mean(rows["mrr"])) if rows["mrr"] else
                     float(metric_from_predictions(spec_graph, torch.cat(rows["pred"]), torch.cat(rows["target"]))))
    return result


@torch.no_grad()
def fit_observation_normalizer(plant: Plant, sample_count=None) -> Dict:
    """Fit raw-channel normalization and the loss scale on training/value data.

    Family samples come only from the disjoint value partition. The selected
    sample IDs are returned so an artifact records the fitting population.
    """
    cfg = plant.cfg
    if plant.is_family:
        ids = list(plant.source.partitions["value"])
        if not ids:
            raise ValueError("The family value partition is empty")
        limit = min(len(ids), cfg.branch_traj if sample_count is None else int(sample_count))
        rng = np.random.default_rng(cfg.seed+631)
        ids = [int(ids[i]) for i in rng.permutation(len(ids))[:max(1, limit)]]
    else:
        ids = [None]
    raw, losses = [], []
    with preserve_plant(plant):
        for sample in ids:
            if sample is not None:
                plant.bind_sample(sample, "value")
            plant.prepare(grad=False)
            ctrl = make_controller(cfg, plant, "schedule")
            ctrl.reset(plant.B, plant.ctx)
            H = plant.H0
            for ell in range(cfg.L_max+1):
                raw.append(plant.observe(H).cpu())
                losses.append(plant.raw_value_loss(H))
                if ell < cfg.L_max:
                    action, rec = ctrl.step(ell, None, plant.ctx)
                    H = plant.layer(H, action)
                    ctrl.commit(ell, action, rec, plant.ctx)
    raw_tensor = torch.cat(raw)
    if not torch.isfinite(raw_tensor).all() or not np.isfinite(losses).all():
        raise ValueError("Non-finite observation or value loss while fitting normalization")
    plant.normalizer.fit(raw_tensor)
    plant.loss_bar = max(float(np.quantile(losses, 0.99)), 1e-8)
    return {"loss_bar": plant.loss_bar, "n_rows": len(raw_tensor), "sample_ids": ids,
            "partition": "value", "loss_scale_quantile": 0.99}


# ============================================================================================
# backbone construction, checkpoints
# ============================================================================================


def build_backbone(cfg: Config, graph: GraphData, bank: Optional[List[str]] = None) -> Backbone:
    geom = geometry_of(cfg) if bank is None else ActionGeometry(len(bank), cfg.Kc, cfg.alpha_min, cfg.alpha_max, cfg.r_min)
    return Backbone(graph.x.shape[1], graph.num_classes, cfg.d, cfg.dff, cfg.Kf, cfg.Kc, cfg.b_min, cfg.R0, bank or cfg.bank_list, geom,
                    cfg.gamma_sl, cfg.ref_knots, cfg.support_s, cfg.eps_g, cfg.role_hidden)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_backbone(bb: Backbone, path, extra: Optional[Dict] = None):
    atomic_torch_save({"schema_version": 2, "state": bb.state_dict(), "bank": bb.bank, "extra": extra or {}}, path)


def load_backbone(cfg: Config, graph: GraphData, path, device) -> Tuple[Backbone, Dict]:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    bb = build_backbone(cfg, graph, ck["bank"])
    bb.load_state_dict(ck["state"])
    bb.to(device)
    bb.geom.to(device)
    return bb, ck.get("extra", {})


def perturb_backbone_(bb: Backbone, scale: float, seed: int):
    """Checkpoint perturbation for single-graph tuning/audit samples: W ← W(1 + scale·N(0,1))."""
    if scale <= 0:
        return
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for p in bb.parameters():
            p.mul_(1.0 + scale * torch.randn(p.shape, generator=g).to(p.device))
    bb.update_sigma()


# ============================================================================================
# training the task route (warm-up / open loop / distillation)
# ============================================================================================


def train_task_route(cfg: Config, plant: Plant, make_ctrl, epochs: int, out_dir: Path, tag: str, ckpt_epochs: Optional[List[int]] = None,
                     L_max: Optional[int] = None, val_depths: Optional[List[int]] = None, train_mix: bool = False, lr: Optional[float] = None,
                     params: Optional[List[nn.Parameter]] = None, extra_state_fn=None) -> Dict:
    """Random-depth task training with whole-split validation and no test access.

    Families use true disjoint-graph minibatches from the task partition. The
    selected checkpoint stores auxiliary schedules together with the backbone.
    """
    bb, dev = plant.bb, plant.device
    out_dir = ensure_dir(out_dir)
    L_max = int(L_max or cfg.L_max)
    val_depths = val_depths or sorted(set([d for d in (4, 8, 12, 16, 24, 32, 48, 64) if d <= L_max] + [L_max]))
    params = list(params if params is not None else [p for p in bb.parameters() if p.requires_grad])
    if not params:
        raise ValueError("The task optimizer has no trainable parameters")
    opt = torch.optim.AdamW(params, lr=lr or cfg.lr, weight_decay=cfg.wd)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed + 17)
    gen_cpu = torch.Generator().manual_seed(cfg.seed + 23)
    minimize = getattr(plant.graph, "metric_mode", "max") == "min"
    best = {"val": float("inf") if minimize else -float("inf"), "epoch": -1, "depth": None}
    history, ckpt_epochs = [], set(ckpt_epochs or [])
    t0 = time.time()
    resume_path = out_dir/f"{tag}_resume.pt"
    identity = {"config": asdict(cfg), "tag": tag, "L_max": L_max, "val_depths": list(val_depths),
                "train_mix": bool(train_mix), "lr": lr or cfg.lr,
                "data_fingerprint": getattr(plant.source, "fingerprint", None),
                "bank": list(bb.bank), "parameter_shapes": [list(p.shape) for p in params]}
    start_epoch = 1
    resumed_from = None
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("identity") != identity:
            raise ValueError(f"Task checkpoint configuration/data mismatch: {resume_path}")
        bb.load_state_dict(checkpoint["backbone"])
        with torch.no_grad():
            for parameter, saved in zip(params, checkpoint["parameters"]):
                parameter.copy_(saved.to(parameter))
        opt.load_state_dict(checkpoint["optimizer"])
        gen.set_state(checkpoint["generator_device"])
        gen_cpu.set_state(checkpoint["generator_cpu"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if dev.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        history, best = checkpoint["history"], checkpoint["best"]
        resumed_from = int(checkpoint["epoch"])
        start_epoch = resumed_from+1
    def save_resume(epoch):
        atomic_torch_save({"identity": identity, "epoch": epoch, "backbone": bb.state_dict(),
            "parameters": [p.detach().cpu() for p in params], "optimizer": opt.state_dict(),
            "generator_device": gen.get_state(), "generator_cpu": gen_cpu.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "history": history, "best": best}, resume_path)
    def extra(epoch):
        result = {"epoch": epoch, "config": asdict(cfg), "metric": getattr(plant.graph, "metric", "accuracy"),
                  "metric_mode": "min" if minimize else "max", "selection_split": "valid"}
        if extra_state_fn is not None:
            result["aux_state"] = extra_state_fn()
        return result
    with preserve_plant(plant):
        for ep in range(start_epoch, max(1, epochs) + 1):
            losses, depths, batches = [], [], 0
            if epochs > 0:
                bb.train()
                if plant.is_family:
                    graphs = plant.source.iter_batches("task", family_batch_size(plant), epoch=ep-1, shuffle=True)
                elif plant.source is not None and hasattr(plant.source, "iter_training_batches"):
                    graphs = plant.source.iter_training_batches(cfg.subset_nodes, epoch=ep-1)
                else:
                    graphs = [plant.graph]
                for graph in graphs:
                    plant.bind_graph(graph)
                    bb.update_sigma()
                    L = int(torch.randint(1, L_max + 1, (1,), generator=gen_cpu))
                    ctrl = make_ctrl()
                    out = run_pass(plant, ctrl, L, grad=True, train_mix=train_mix)
                    loss = plant.task_loss(out["H"])
                    if cfg.lam_struct > 0:
                        count = min(cfg.structural_samples, max(256, 4*plant.n))
                        loss = loss + cfg.lam_struct * bb.roles.struct_loss(plant.graph, plant.roles, count, count, gen)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite task loss in {tag}, epoch {ep}")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
                    opt.step()
                    losses.append(float(loss.detach())); depths.append(L); batches += 1
                if not batches:
                    raise ValueError("The task-training partition is empty")
            bb.update_sigma()
            row = {"epoch": ep if epochs else 0, "loss": float(np.mean(losses)) if losses else None,
                   "L": depths[-1] if depths else None, "n_batches": batches}
            history.append(row)
            if ep in ckpt_epochs:
                save_backbone(bb, out_dir / f"{tag}_ep{ep}.pt", extra(ep))
            if ep % cfg.val_every == 0 or ep == max(1, epochs):
                bb.eval()
                vals = evaluate_depths(plant, make_ctrl, val_depths, split="valid")
                finite = {d:v for d,v in vals.items() if math.isfinite(v)}
                if not finite:
                    raise ValueError("Validation metric is undefined at every candidate depth")
                d_best = min(finite, key=finite.get) if minimize else max(finite, key=finite.get)
                row["valid"] = vals
                better = finite[d_best] < best["val"] if minimize else finite[d_best] > best["val"]
                if better:
                    best = {"val": finite[d_best], "epoch": row["epoch"], "depth": d_best,
                            "sweep": {int(k): {"valid": v} for k,v in vals.items()}}
                    meta = extra(row["epoch"]) | {"depth": d_best, "valid": finite[d_best]}
                    save_backbone(bb, out_dir / f"{tag}_best.pt", meta)
                log(f"[{tag}] epoch {row['epoch']}/{epochs} valid@{d_best}={finite[d_best]:.6g} best={best['val']:.6g}@ep{best['epoch']}")
            save_resume(ep)
        save_backbone(bb, out_dir / f"{tag}_final.pt", extra(epochs))
    return {"best": best, "history": history[-50:], "epochs": epochs, "seconds": time.time()-t0,
            "selection_split": "valid", "test_evaluated": False, "metric_mode": "min" if minimize else "max",
            "resumed_from_epoch": resumed_from, "checkpoint_scope": "completed_epoch_optimizer_and_rng"}


# ============================================================================================
# bank selection: E σ̄(C_H) per unit cost over warm-up checkpoints (Section V-B)
# ============================================================================================


@torch.no_grad()
def response_jacobian(plant: Plant, H: torch.Tensor, ell: int, A_center: torch.Tensor, step: float = 1e-4,
                      sens_seed: Optional[int] = None) -> torch.Tensor:
    """Finite-difference C_H = ∂z_{ℓ+1}/∂u (dz × du) of the network's observation map at (H, ℓ)."""
    geom = plant.geom
    u0 = geom.to_tangent(A_center)
    cols = []
    for j in range(geom.du):
        e = torch.zeros_like(u0)
        e[:, j] = step
        zp = plant.normalizer(plant.observe(plant.layer(H, geom.from_tangent(u0 + e)), sens_seed))
        zm = plant.normalizer(plant.observe(plant.layer(H, geom.from_tangent(u0 - e)), sens_seed))
        cols.append((zp - zm)[0] / (2 * step))
    return torch.stack(cols, 1)


def sigma_bar(C: torch.Tensor, subspace: Optional[torch.Tensor] = None) -> float:
    """σ̄(C) = (det CᵀC)^{1/(2du)}; on the identifiable subspace at ℓ = 0 (restart–residual columns removed)."""
    if subspace is not None:
        C = C[:, subspace]
    s = torch.linalg.svdvals(C.double())
    return float(torch.exp(torch.log(s.clamp_min(1e-30)).mean()))


def bank_cost(bank: List[str], n: int, nnz_union: int, Kf: int, d: int) -> float:
    """Relative per-layer cost model: one fused SpMM on the union support plus the factorised global route."""
    c = nnz_union * d
    if "Pg" in bank:
        c += n * Kf * d + Kf * Kf * d
    c += len(bank) * nnz_union
    return float(c)


# ============================================================================================
# branch collection (Section V-B, Lemma 12) on a single graph
# ============================================================================================


@dataclass
class BranchData:
    """Saved states, antithetic branches and continuations of one collection pass (all CPU tensors)."""
    z: torch.Tensor; x: torch.Tensor; x_prior: torch.Tensor; e: torch.Tensor; t: torch.Tensor; ell: torch.Tensor
    A_prev: torch.Tensor; eta: torch.Tensor; b: torch.Tensor; g: torch.Tensor; V_tgt: torch.Tensor; ctx_c: torch.Tensor
    block: torch.Tensor; same_state_block: torch.Tensor
    z_next_c: torch.Tensor; zH_c: torch.Tensor; uH_c: torch.Tensor      # centre: next obs, next-H obs, executed tangent actions (H)
    u_br: torch.Tensor; z_br: torch.Tensor; row_br: torch.Tensor; sign_br: torch.Tensor; delta_br: torch.Tensor
    cont_br: torch.Tensor; G_br: torch.Tensor; G_c: torch.Tensor; zH_br: torch.Tensor; uH_br: torch.Tensor
    exec_u: torch.Tensor; traj_e: torch.Tensor; traj_x: torch.Tensor; traj_z: torch.Tensor; traj_t: torch.Tensor
    ctx_br: Optional[torch.Tensor] = None       # actual post-action memory, for value targets
    plant_ctx: Optional[torch.Tensor] = None    # one graph context per saved state
    traj_ctx: Optional[torch.Tensor] = None     # one context per retained trajectory state
    traj_id: Optional[torch.Tensor] = None
    sample_ids: Optional[torch.Tensor] = None
    observation_space: str = "raw"
    schema_version: int = 2

    def n_states(self) -> int:
        return int(self.z.shape[0])


class ControllerSnapshot:
    """Copy/restore of the controller memory (for branches and continuations)."""

    def __init__(self, ctrl: Controller):
        self.x_prior = None if ctrl.x_prior is None else ctrl.x_prior.clone()
        self.eta = ctrl.eta.clone()
        self.eta_student = ctrl.eta_student.clone()
        self.A_prev = ctrl.A_prev.clone()
        self.V_prev = None if ctrl.V_prev is None else ctrl.V_prev.clone()
        self.pending = copy.deepcopy(getattr(ctrl, "_pending", None))

    def restore(self, ctrl: Controller):
        ctrl.x_prior = None if self.x_prior is None else self.x_prior.clone()
        ctrl.eta = self.eta.clone()
        ctrl.eta_student = self.eta_student.clone()
        ctrl.A_prev = self.A_prev.clone()
        ctrl.V_prev = None if self.V_prev is None else self.V_prev.clone()
        if self.pending is None:
            if hasattr(ctrl, "_pending"):
                del ctrl._pending
        else:
            ctrl._pending = copy.deepcopy(self.pending)


@torch.no_grad()
def base_continuation(plant: Plant, ctrl: Controller, H: torch.Tensor, ell_from: int, T: int, sens_seed: Optional[int],
                      n_obs: int = 0, hazard: float = 1 / 32):
    """Continue under the controller (base) from H at layer ell_from for T layers; returns the
    hazard-weighted return Σ_{k'=0}^{T−1} h(1−h)^{k'} g(H_{ell_from+k'}), the first n_obs raw
    observations and executed tangent actions along the way."""
    if T < max(1, n_obs+1):
        raise ValueError("Continuation must include every requested future observation")
    snap = ControllerSnapshot(ctrl)
    gs, zs, us = [], [], []
    Hc = H
    try:
        for j in range(T):
            ell = ell_from + j
            gs.append(float(plant.stage_loss(Hc).mean()))
            z_raw = plant.observe(Hc, sens_seed) if ctrl.needs_sensing else None
            if 1 <= j <= n_obs:                       # z at ell_from + j, j = 1..n_obs (the step-j predictions' targets)
                zs.append(z_raw[0].clone() if z_raw is not None else plant.observe(Hc, sens_seed)[0].clone())
            A, rec = ctrl.step(ell, z_raw, plant.ctx)
            if j < n_obs:                             # executed tangent action at ell_from + j, j = 0..n_obs−1
                us.append(plant.geom.to_tangent(A)[0].clone())
            Hc = plant.layer(Hc, A)
            ctrl.commit(ell, A, rec, plant.ctx)
    finally:
        snap.restore(ctrl)
    return continuation_value(gs, hazard), zs, us


def bind_training_sample(plant: Plant, trajectory: int, seed: int, partition="value"):
    """Sample training graphs only; preserve the global value mask on node subgraphs."""
    if plant.is_family:
        ids = list(plant.source.partitions[partition])
        if not ids:
            raise ValueError(f"Empty family {partition} partition")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        sample = int(ids[int(order[trajectory % len(ids)])])
        plant.bind_sample(sample, partition)
        return sample
    if plant.source is not None and hasattr(plant.source, "iter_training_batches"):
        for graph in plant.source.iter_training_batches(plant.cfg.subset_nodes, epoch=seed+trajectory):
            plant.bind_graph(graph)
            if plant.part.value_idx.numel():
                return int(getattr(graph, "sample_id", -1) or -1)
        raise ValueError("No training subgraph contains value labels")
    plant.bind_graph(plant.reference_graph)
    return -1


@torch.no_grad()
def collect_branches(cfg: Config, plant: Plant, make_base_ctrl, n_traj: int, states_per_traj: int, T_cap: int, seed: int,
                     explore: bool = True) -> BranchData:
    """Collect complete, jointly eroded Hadamard blocks and true continuations.

    Each block uses a single radius vector computed from all of its saved states.
    Backbone states are temporarily kept on CPU and released after each trajectory.
    The observation fields are raw; actual and nominal memory contexts are distinct.
    """
    if n_traj < 1 or states_per_traj < 1 or T_cap < cfg.H:
        raise ValueError("Positive trajectories/states and T_cap >= H are required")
    geom, dev = plant.geom, plant.device
    design = RotatingDesign(geom.du, cfg.Kb)
    spb = max(1, math.ceil(design.NH / design.Kb))
    n_blocks = max(1, math.ceil(states_per_traj / spb))
    gen = torch.Generator().manual_seed(seed)
    names = ("z", "x", "x_prior", "e", "t", "ell", "A_prev", "eta", "b", "g", "V_tgt", "ctx_c", "block", "same",
             "z_next_c", "zH_c", "uH_c", "u_br", "z_br", "row_br", "sign_br", "delta_br", "cont_br", "G_br", "G_c",
             "zH_br", "uH_br", "ctx_br", "plant_ctx", "sample_ids")
    recs = {k: [] for k in names}
    trajectories = {k: [] for k in ("e", "x", "z", "t", "u", "ctx", "id")}
    block_id = 0
    with preserve_plant(plant):
        for tr in range(n_traj):
            sample_id = bind_training_sample(plant, tr, seed)
            plant.prepare(grad=False)
            if plant.B != 1:
                raise ValueError("Branch collection samples individual graphs; source.sample must return num_graphs=1")
            sens_seed = 500 + seed*97 + tr
            ctrl = make_base_ctrl()
            ctrl.reset(1, plant.ctx)
            ctrl.role_mask = plant.role_mask
            blocks = []
            for k in range(n_blocks):
                if tr == 0 and k == 0:
                    layers = [0]*spb
                elif cfg.L_max < spb:
                    depth = int(torch.randint(1, cfg.L_max+1, (1,), generator=gen))
                    layers = [depth]*spb
                else:
                    start = int(torch.randint(1, cfg.L_max-spb+2, (1,), generator=gen))
                    layers = list(range(start, start+spb))
                blocks.append((block_id, layers))
                block_id += 1
            wanted = set(ell for _, layers in blocks for ell in layers)
            first_branch = min(wanted)
            saved, gs, zs, us = {}, [], [], []
            H = plant.H0
            total = cfg.L_max + max(T_cap, cfg.H) + 1
            for ell in range(total):
                raw = plant.observe(H, sens_seed)
                A, rec = ctrl.step(ell, raw, plant.ctx)
                executed = A
                if explore and ell < first_branch and float(torch.rand(1, generator=gen)) < cfg.p_explore:
                    noise = torch.randn(A.shape, generator=gen).to(dev)*cfg.explore_radius
                    executed = geom.project(A+noise)[0]
                if ell in wanted:
                    saved[ell] = {"H": H.detach().cpu(), "memory": ControllerSnapshot(ctrl),
                                  "rec": {k:v.clone() if torch.is_tensor(v) else v for k,v in rec.items()},
                                  "raw": raw[0].cpu(), "eta": ctrl.eta[0].cpu(), "b": A.clone()}
                gs.append(float(plant.stage_loss(H)[0]))
                zs.append(raw[0].cpu())
                u = geom.to_tangent(executed)[0].cpu()
                us.append(u)
                if ell <= cfg.L_max:
                    for key, value in (("e", rec["e"][0]), ("x", rec["x"][0]), ("z", raw[0]), ("u", u),
                                       ("ctx", plant.ctx[0])):
                        trajectories[key].append(value.cpu())
                    trajectories["t"].append(float(depth_t(ell, cfg.ell0)))
                    trajectories["id"].append(tr)
                H = plant.layer(H, executed)
                ctrl.commit(ell, executed, rec, plant.ctx)
            for bid, layers in blocks:
                centers = torch.cat([geom.to_tangent(saved[l]["b"]) for l in layers])
                delta = slack_radii(geom, centers, cfg.delta, cfg.eta_s)
                lo, hi, beta = eroded_bounds(geom, delta)
                for k, ell in enumerate(layers):
                    state = saved[ell]
                    H = state["H"].to(dev)
                    rec, b = state["rec"], state["b"]
                    rows = torch.arange(k*cfg.Kb, min((k+1)*cfg.Kb, design.NH))
                    if rows.numel() != cfg.Kb:
                        raise ValueError("Kb must divide the Hadamard order for rectangular saved branch tensors")
                    directions = design.H[rows].to(dev)
                    eroded = geom.project(b, lo.unsqueeze(0), hi.unsqueeze(0), beta.unsqueeze(0))[0]
                    center = geom.to_tangent(eroded)
                    nbranch = 2*len(rows)
                    continued = torch.zeros(nbranch, dtype=torch.bool)
                    n_cont = min(nbranch, max(1, round(cfg.continuation_frac*nbranch)))
                    continued[torch.randperm(nbranch, generator=gen)[:n_cont]] = True
                    zbr, ubr, gbr, zhbr, uhbr, cb = [], [], [], [], [], []
                    t = depth_t(ell, cfg.ell0).to(dev).expand(1)
                    xn_base = ctrl.model.predict(rec["x"], t, geom.to_tangent(b), plant.ctx,
                                                 torch.tensor([ell == 0], device=dev))
                    nominal_ctx = ctrl._ctx(depth_t(ell+1, cfg.ell0).to(dev), xn_base, rec["eta_next"], b)
                    for row, direction in enumerate(directions):
                        for sign in (1.0, -1.0):
                            u = center + sign*delta[None, :]*direction[None, :]
                            action = geom.from_tangent(u)
                            if float(geom.feasibility_violation(action).max()) > 2e-6:
                                raise AssertionError("A jointly eroded intervention is infeasible")
                            state["memory"].restore(ctrl)
                            Hn = plant.layer(H, action)
                            ctrl.commit(ell, action, rec, plant.ctx)
                            actual_ctx = ctrl._ctx(depth_t(ell+1, cfg.ell0).to(dev), ctrl.x_prior, ctrl.eta, ctrl.A_prev)
                            cb.append(actual_ctx[0].cpu())
                            zbr.append(plant.observe(Hn, sens_seed)[0].cpu()); ubr.append(u[0].cpu())
                            j = 2*row + (sign < 0)
                            if continued[j]:
                                value, future_z, future_u = base_continuation(plant, ctrl, Hn, ell+1, T_cap,
                                                                             sens_seed, cfg.H-1, cfg.hazard)
                                zh = torch.stack(future_z).cpu() if future_z else torch.zeros(0, cfg.dz)
                                uh = torch.stack(future_u).cpu() if future_u else torch.zeros(0, geom.du)
                            else:
                                value, zh, uh = float("nan"), torch.zeros(cfg.H-1, cfg.dz), torch.zeros(cfg.H-1, geom.du)
                            gbr.append(value); zhbr.append(zh); uhbr.append(uh)
                    row_values = {"z": state["raw"], "x": rec["x"][0].cpu(), "x_prior": rec["x_prior"][0].cpu(),
                        "e": rec["e"][0].cpu(), "t": float(t[0]), "ell": ell, "A_prev": rec["A_prev"][0].cpu(),
                        "eta": state["eta"], "b": b[0].cpu(), "g": gs[ell],
                        "V_tgt": continuation_value(gs[ell:ell+T_cap], cfg.hazard), "ctx_c": nominal_ctx[0].cpu(),
                        "block": bid, "same": len(set(layers)) == 1, "z_next_c": zs[ell+1],
                        "zH_c": torch.stack(zs[ell+2:ell+cfg.H+1]) if cfg.H > 1 else torch.zeros(0, cfg.dz),
                        "uH_c": torch.stack(us[ell:ell+cfg.H]), "u_br": torch.stack(ubr), "z_br": torch.stack(zbr),
                        "row_br": rows.repeat_interleave(2), "sign_br": torch.tensor([1.0,-1.0]).repeat(len(rows)),
                        "delta_br": delta.cpu(), "cont_br": continued, "G_br": torch.tensor(gbr),
                        "G_c": continuation_value(gs[ell+1:ell+1+T_cap], cfg.hazard), "zH_br": torch.stack(zhbr),
                        "uH_br": torch.stack(uhbr), "ctx_br": torch.stack(cb), "plant_ctx": plant.ctx[0].cpu(),
                        "sample_ids": sample_id}
                    for key, value in row_values.items():
                        recs[key].append(value)
            del saved
    def stack(values):
        return torch.stack(values) if torch.is_tensor(values[0]) else torch.tensor(values)
    fields = {("same_state_block" if k == "same" else k): stack(v) for k,v in recs.items()}
    fields.update(exec_u=stack(trajectories["u"]), traj_e=stack(trajectories["e"]), traj_x=stack(trajectories["x"]),
                  traj_z=stack(trajectories["z"]), traj_t=stack(trajectories["t"]), traj_ctx=stack(trajectories["ctx"]),
                  traj_id=stack(trajectories["id"]))
    return BranchData(**fields)


# ============================================================================================
# identification (10), heads, health, support, scorer
# ============================================================================================


def pooled_blocks(bd: BranchData, normalizer: Normalizer, design: RotatingDesign) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """Per completed block: (depth of its first state, pooled estimate Ĉ, mean δ).  Observations are
    standardised before pooling."""
    out = []
    for bid in bd.block.unique().tolist():
        rows = (bd.block == bid).nonzero(as_tuple=True)[0]
        zp, zm, xi, djs = [], [], [], []
        for r in rows.tolist():
            zb = normalizer(bd.z_br[r])                                   # (2·n_rows, dz)
            n_rows = zb.shape[0] // 2
            zp.append(zb[0::2]); zm.append(zb[1::2])
            xi.append(design.H[bd.row_br[r][0::2]])
            djs.append(bd.delta_br[r])
        zp, zm, xi = torch.cat(zp), torch.cat(zm), torch.cat(xi)
        dj = djs[0]
        if not all(torch.equal(dj, other) for other in djs[1:]):
            raise ValueError("A completed Hadamard block must use one common radius vector")
        if zp.shape[0] < design.NH:
            continue
        if len(torch.unique(torch.cat([bd.row_br[r][0::2] for r in rows.tolist()]))) != design.NH:
            raise ValueError("Incomplete or duplicated rows in a Hadamard block")
        Chat = pooled_response(zp[: design.NH], zm[: design.NH], xi[: design.NH], dj)
        out.append((int(bd.ell[rows[0]]), Chat, dj))
    return out


def fit_heads(cfg: Config, value: ConvexHead, stage: ConvexHead, bd: BranchData, normalizer: Normalizer, device, epochs: int) -> Dict:
    """Fit V_ω(z;t,ζ) to hazard-weighted continuation targets and ĝ_ω(z) to stage losses, from the
    visited states and the continued branches' next states; certify L_V afterwards."""
    if getattr(bd, "observation_space", None) != "raw":
        raise ValueError("Branch observations must be raw; regenerate legacy branch artifacts")
    if epochs < 1:
        raise ValueError("Head fitting needs at least one optimizer step")
    z = normalizer(bd.z).to(device)
    # context of a visited state is (t, x⁻_ℓ, η_ℓ, A_{ℓ−1}); the stored ctx_c is the next-layer context
    ctx_state = torch.cat([bd.t.unsqueeze(1), bd.x_prior, bd.eta, bd.A_prev.flatten(1)], 1).to(device)
    V_t = bd.V_tgt.to(device).float()
    g_t = bd.g.to(device).float()
    # Fit values on the memory that actually generated each continuation. The
    # nominal base context is held fixed only while comparing planned actions.
    zb, Vb, cb = [], [], []
    for r in range(bd.n_states()):
        m = bd.cont_br[r]
        if m.any():
            zb.append(normalizer(bd.z_br[r][m]))
            Vb.append(bd.G_br[r][m])
            if bd.ctx_br is None:
                raise ValueError("Continued branch targets require their actual post-action memory context")
            cb.append(bd.ctx_br[r][m])
    if zb:
        z_all = torch.cat([z, torch.cat(zb).to(device)])
        V_all = torch.cat([V_t, torch.cat(Vb).to(device).float()])
        c_all = torch.cat([ctx_state, torch.cat(cb).to(device)])
    else:
        z_all, V_all, c_all = z, V_t, ctx_state
    optV = torch.optim.Adam(value.parameters(), lr=3e-3)
    optg = torch.optim.Adam(stage.parameters(), lr=3e-3)
    for ep in range(epochs):
        lossV = F.mse_loss(value(z_all, c_all), V_all)
        optV.zero_grad(); lossV.backward(); optV.step(); value.project_()
        lossg = F.mse_loss(stage(z), g_t)
        optg.zero_grad(); lossg.backward(); optg.step(); stage.project_()
    LVv = value.certify_(cfg.L_V)
    LVg = stage.certify_(cfg.L_V)
    with torch.no_grad():
        mseV = float(F.mse_loss(value(z_all, c_all), V_all))
        mseg = float(F.mse_loss(stage(z), g_t))
    return {"value_mse": mseV, "stage_mse": mseg, "LV_bound_value": LVv, "LV_bound_stage": LVg, "n_value_rows": int(z_all.shape[0]),
            "Gf_value": value.grad_bound(), "Gf_stage": stage.grad_bound()}


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    from scipy.stats import rankdata
    if a.numel() < 3 or a.numel() != b.numel():
        return float("nan")
    ra = torch.as_tensor(rankdata(a.detach().cpu().numpy(), method="average"), dtype=torch.float64)
    rb = torch.as_tensor(rankdata(b.detach().cpu().numpy(), method="average"), dtype=torch.float64)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = ra.norm() * rb.norm()
    if float(den) == 0:
        return float("nan")
    return float((ra * rb).sum() / den)


def channel_scores(bd: BranchData, normalizer: Normalizer) -> torch.Tensor:
    """Rank correlation between channel contrasts Δz_j = z_{ℓ+1}^{(k)} − z_{ℓ+1}^{(c)} and continuation-loss
    contrasts G_k − G_c over continued branches (channel selection of Ξ)."""
    dzs, dGs = [], []
    for r in range(bd.n_states()):
        m = bd.cont_br[r]
        if not m.any():
            continue
        zc = normalizer(bd.z_next_c[r].unsqueeze(0))[0]
        dzs.append(normalizer(bd.z_br[r][m]) - zc)
        dGs.append(bd.G_br[r][m] - bd.G_c[r])
    if not dzs:
        return torch.zeros(bd.z.shape[1])
    dz, dG = torch.cat(dzs), torch.cat(dGs)
    return torch.tensor([spearman(dz[:, j], dG) for j in range(dz.shape[1])]).nan_to_num(0.0)


def fit_model(cfg: Config, model: TransitionModel, value: ConvexHead, health: HealthRegion, bd: BranchData, normalizer: Normalizer,
              device, epochs: int, design: RotatingDesign) -> Dict:
    """Decision-focused identification on detached branches, with a trained observer.

    Heads remain parameter-frozen while their input derivatives train the model.
    Posterior states are recomputed under the current C and noise parameters;
    the depth-zero prior is the learned graph-context initialization.
    """
    if epochs < 1 or bd.n_states() < 1:
        raise ValueError("Identification requires records and positive epochs")
    if getattr(bd, "observation_space", None) != "raw":
        raise ValueError("Regenerate branch artifacts with raw observations")
    S, n_br, du = bd.n_states(), bd.u_br.shape[1], model.du
    context = plant_ctx_from_bd(bd, cfg, device)
    tensors = {"z": normalizer(bd.z).to(device), "prior": bd.x_prior.to(device), "t": bd.t.to(device),
               "ell": bd.ell.to(device), "ctx": bd.ctx_c.to(device), "context": context,
               "ub": bd.u_br.to(device), "zb": normalizer(bd.z_br).to(device),
               "uc": bd.uH_c[:,0].to(device), "zc": normalizer(bd.z_next_c).to(device),
               "cont": bd.cont_br.to(device), "gb": bd.G_br.to(device), "gc": bd.G_c.to(device),
               "zhc": normalizer(bd.zH_c).to(device), "uhc": bd.uH_c.to(device),
               "zhb": normalizer(bd.zH_br).to(device), "uhb": bd.uH_br.to(device)}
    blocks = pooled_blocks(bd, normalizer, design)
    excitation = torch.stack([excitation_profile(C) for _,C,_ in blocks]).mean(0) if blocks else torch.zeros(du)
    lambdas = ridge_weights(excitation, cfg.lam0, cfg.tau_exc, cfg.kappa_max).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    huber = nn.HuberLoss(delta=1.0)
    head_flags = [p.requires_grad for p in value.parameters()]
    for p in value.parameters():
        p.requires_grad_(False)
        p.grad = None
    history = []
    def predict(batch):
        zero = batch["ell"] == 0
        prior = torch.where(zero[:,None], model.x_init(batch["context"]), batch["prior"])
        x, innovation = model.correct(prior, batch["z"])
        xn_c = model.predict(x, batch["t"], batch["uc"], batch["context"], zero)
        xb = x[:,None].expand(-1,n_br,-1).reshape(-1,model.dx)
        tb = batch["t"][:,None].expand(-1,n_br).reshape(-1)
        cb = batch["context"][:,None].expand(-1,n_br,-1).reshape(-1,model.dc)
        dzb = zero[:,None].expand(-1,n_br).reshape(-1)
        xn_b = model.predict(xb, tb, batch["ub"].reshape(-1,du), cb, dzb)
        zp_c, zp_b = xn_c @ model.C.T, (xn_b @ model.C.T).reshape(-1,n_br,model.dz)
        return x, xn_c, xn_b, zp_c, zp_b, cb
    def losses(batch):
        x, xn_c, xn_b, zp_c, zp_b, cb = predict(batch)
        ns = len(x)
        dynamic = huber(zp_c,batch["zc"]) + huber(zp_b,batch["zb"])
        xr = xn_c
        for j in range(batch["zhc"].shape[1]):
            tj = depth_t(batch["ell"]+1+j,cfg.ell0).to(device)
            xr = model.predict(xr,tj,batch["uhc"][:,j+1],batch["context"],torch.zeros(ns,dtype=torch.bool,device=device))
            dynamic = dynamic+huber(xr@model.C.T,batch["zhc"][:,j])
        selected = batch["cont"].reshape(-1)
        if selected.any():
            xr = xn_b[selected]
            future_actions = batch["uhb"].reshape(ns*n_br,cfg.H-1,du)[selected]
            future_obs = batch["zhb"].reshape(ns*n_br,cfg.H-1,model.dz)[selected]
            start = batch["ell"][:,None].expand(-1,n_br).reshape(-1)[selected]
            for j in range(cfg.H-1):
                tj = depth_t(start+1+j,cfg.ell0).to(device)
                xr = model.predict(xr,tj,future_actions[:,j],cb[selected],torch.zeros(len(xr),dtype=torch.bool,device=device))
                dynamic = dynamic+huber(xr@model.C.T,future_obs[:,j])
        tnext = depth_t(batch["ell"]+1,cfg.ell0).to(device)
        tnb = tnext[:,None].expand(-1,n_br).reshape(-1)
        contexts = batch["ctx"][:,None].expand(-1,n_br,-1).reshape(-1,batch["ctx"].shape[-1])
        vc = value(zp_c,batch["ctx"])
        vb = value(zp_b.reshape(-1,model.dz),contexts).reshape(ns,n_br)
        pred = vb+health.psi(zp_b.reshape(-1,model.dz),tnb).reshape(ns,n_br)-(vc+health.psi(zp_c,tnext))[:,None]
        with torch.no_grad():
            observed = (value(batch["zb"].reshape(-1,model.dz),contexts)+health.psi(batch["zb"].reshape(-1,model.dz),tnb)).reshape(ns,n_br)
            observed -= (value(batch["zc"],batch["ctx"])+health.psi(batch["zc"],tnext))[:,None]
        contrast = huber(pred,observed)
        # Consecutive rows are the positive/negative members of one intervention.
        contrast += huber(pred[:,0::2]-pred[:,1::2], observed[:,0::2]-observed[:,1::2])
        advantage = huber((vb-vc[:,None])[batch["cont"]],(batch["gb"]-batch["gc"][:,None])[batch["cont"]]) if selected.any() else x.new_zeros(())
        response = model.Bmat(x,batch["t"],batch["ell"]==0)
        ridge = (response.square().mean(0).sum(0)*lambdas).sum()
        total = dynamic+cfg.lam_p*contrast+cfg.lam_adv*advantage+ridge
        return total, dynamic, contrast, advantage, ridge
    try:
        for ep in range(epochs):
            totals = np.zeros(5)
            order = torch.randperm(S,device=device)
            for ids in order.split(min(32,S)):
                batch = {k:v[ids] for k,v in tensors.items()}
                values = losses(batch)
                if not torch.isfinite(values[0]):
                    raise FloatingPointError(f"Non-finite identification loss at epoch {ep}")
                opt.zero_grad(set_to_none=True)
                values[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.0,error_if_nonfinite=True)
                opt.step(); model.constrain_()
                totals += np.array([float(v.detach()) for v in values])*len(ids)/S
            if ep % max(1,epochs//5)==0 or ep==epochs-1:
                history.append(dict(zip(("loss","L_dyn","L_p","L_adv","ridge"),totals.tolist()))|{"epoch":ep})
        with torch.no_grad():
            predictions, observed, advp, advt = [], [], [], []
            for ids in torch.arange(S,device=device).split(min(32,S)):
                batch={k:v[ids] for k,v in tensors.items()}
                _,_,_,zc,zb,_ = predict(batch)
                predictions.append(zb.cpu()); observed.append(batch["zb"].cpu())
                if batch["cont"].any():
                    ctxb=batch["ctx"][:,None].expand(-1,n_br,-1).reshape(-1,batch["ctx"].shape[-1])
                    pred=value(zb.reshape(-1,model.dz),ctxb).reshape(-1,n_br)-value(zc,batch["ctx"])[:,None]
                    advp.append(pred[batch["cont"]].cpu())
                    advt.append((batch["gb"]-batch["gc"][:,None])[batch["cont"]].cpu())
            yp,yt=torch.cat(predictions),torch.cat(observed)
            nmse=float((yp-yt).square().mean()/yt.var(unbiased=False).clamp_min(1e-9))
            rank=spearman(torch.cat(advp),torch.cat(advt)) if advp else float("nan")
            ordering=pairwise_order_accuracy(torch.cat(advp),torch.cat(advt)) if advp else {}
    finally:
        for parameter,flag in zip(value.parameters(),head_flags):
            parameter.requires_grad_(flag)
    return {"history":history,"nmse1":nmse,"rank_corr":rank,"order_accuracy":ordering,
            "excitation":excitation.tolist(),"ridge":lambdas.cpu().tolist(),
            "inactive_coords":int((excitation<cfg.tau_exc).sum()),"n_blocks":len(blocks),"norms":model.norms(),
            "diagnostic_partition":"training_branches","observer_fitted":True,"head_parameters_frozen_during_identification":True}


def plant_ctx_from_bd(bd: BranchData, cfg: Config, device) -> torch.Tensor:
    """Return persisted per-state contexts; never silently substitute a zero graph."""
    c = getattr(bd, "plant_ctx", None)
    if c is None:
        c = getattr(bd, "_plant_ctx", None)
    if c is None:
        raise ValueError("Branch artifact lacks graph contexts; collect fresh branches")
    if c.shape != (bd.n_states(), cfg.dc):
        raise ValueError("Branch graph-context shape is inconsistent with its configuration")
    return c.to(device)


def pairwise_order_accuracy(pred: torch.Tensor, true: torch.Tensor, n_pairs: int = 20000, n_deciles: int = 10) -> Dict:
    n = pred.numel()
    if n < 4:
        return {}
    g = torch.Generator().manual_seed(0)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    m = i != j
    i, j = i[m], j[m]
    dt = true[i] - true[j]
    dp = pred[i] - pred[j]
    ok = (torch.sign(dt) == torch.sign(dp)).float()
    mag = dt.abs()
    q = torch.quantile(mag, torch.linspace(0, 1, n_deciles + 1))
    out = {"overall": float(ok.mean())}
    for d in range(n_deciles):
        sel = (mag >= q[d]) & (mag <= q[d + 1])
        out[f"decile_{d + 1}"] = float(ok[sel].mean()) if sel.any() else float("nan")
    return out


def fit_support_and_scorer(cfg: Config, support: SupportDistance, scorer: ReliabilityScorer, bd: BranchData, model: TransitionModel,
                           normalizer: Normalizer, device):
    support.fit(bd.exec_u)
    with torch.no_grad():
        z = normalizer(bd.traj_z).to(device)
        if bd.traj_ctx is None or bd.traj_id is None or len(bd.exec_u) != len(z):
            raise ValueError("Trajectory contexts, boundaries and executed actions must be persisted together")
        contexts = bd.traj_ctx.to(device)
        x_prior = model.x_init(contexts[:1])
        es, xs = [], []
        for i in range(z.shape[0]):
            if i == 0 or bd.traj_id[i] != bd.traj_id[i-1]:
                x_prior = model.x_init(contexts[i:i+1])
            xcor, e = model.correct(x_prior, z[i:i + 1])
            es.append(e[0]); xs.append(xcor[0])
            ell_i = int(round(float(bd.traj_t[i]) * cfg.ell0 / max(1e-9, 1 - float(bd.traj_t[i]))))
            u_i = bd.exec_u[i:i+1].to(device)
            x_prior = model.predict(xcor, bd.traj_t[i:i + 1].to(device), u_i, contexts[i:i+1], torch.tensor([ell_i == 0], device=device))
        scorer.fit_training(torch.stack(es), torch.stack(xs))


# ============================================================================================
# DAgger amortization (Section V-E, Appendix E)
# ============================================================================================


def dagger_train(cfg: Config, plant: Plant, make_planner_ctrl, student: BasePolicyGRU, n_rounds: int, n_traj: int, seed: int,
                 label: str = "executed", epochs: int = 200) -> Dict:
    """DAgger on value/training graphs, with the student's own recurrent memory.

    Later rounds execute the student with probability 1-2**(-round). The teacher
    labels visited states; no validation, audit or test graph enters optimization.
    """
    if n_rounds < 1 or n_traj < 1 or epochs < 1 or label not in ("executed", "proposal"):
        raise ValueError("Invalid DAgger budget or label type")
    geom, dev = plant.geom, plant.device
    gen = torch.Generator().manual_seed(seed)
    dataset, hist, sample_ids = [], [], []
    opt = torch.optim.Adam(student.parameters(),lr=2e-3)
    with preserve_plant(plant):
        for j in range(1,n_rounds+1):
            p_student=0.0 if j==1 else 1.0-2.0**(-j)
            for tr in range(n_traj):
                sample_ids.append(bind_training_sample(plant,tr,seed+j*104729))
                plant.prepare(grad=False)
                sens_seed=900+seed*31+j*7+tr
                L=int(torch.randint(max(1,cfg.L_max//2),cfg.L_max+1,(1,),generator=gen))
                ctrl=make_planner_ctrl();ctrl.reset(plant.B,plant.ctx);ctrl.role_mask=plant.role_mask
                if plant.B != 1:
                    raise ValueError("DAgger source samples must be individual graphs")
                eta_s=torch.zeros(1,cfg.wA,device=dev)
                H=plant.H0
                xs,previous,ts,labels=[],[],[],[]
                with torch.no_grad():
                    for ell in range(L):
                        raw=plant.observe(H,sens_seed)
                        teacher,rec=ctrl.step(ell,raw,plant.ctx)
                        t=depth_t(ell,cfg.ell0).to(dev)
                        target=teacher if label=="executed" else rec["b"]+rec["d"]
                        proposed,eta_s=student(rec["x"],ctrl.A_prev,eta_s,t)
                        # Masked roles execute the base under both teachers and students.
                        proposed=torch.where(plant.role_mask[...,None].bool(),proposed,rec["b"])
                        xs.append(rec["x"][0].cpu());previous.append(ctrl.A_prev[0].cpu())
                        ts.append(float(t));labels.append(target[0].cpu())
                        action=proposed if float(torch.rand(1,generator=gen))<p_student else teacher
                        H=plant.layer(H,action);ctrl.commit(ell,action,rec,plant.ctx)
                dataset.append({"x":torch.stack(xs),"A_prev":torch.stack(previous),"t":torch.tensor(ts),"label":torch.stack(labels)})
            for ep in range(epochs):
                total=0.0;opt.zero_grad(set_to_none=True)
                for sequence in dataset:
                    eta=torch.zeros(1,cfg.wA,device=dev)
                    loss=0.0
                    for i in range(len(sequence["x"])):
                        action,eta=student(sequence["x"][i:i+1].to(dev),sequence["A_prev"][i:i+1].to(dev),eta,sequence["t"][i].to(dev))
                        loss=loss+geom.norm_1sigma(action,sequence["label"][i:i+1].to(dev)).mean()
                    loss=loss/len(sequence["x"])/len(dataset)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite DAgger loss")
                    loss.backward();total+=float(loss.detach())
                torch.nn.utils.clip_grad_norm_(student.parameters(),5.0,error_if_nonfinite=True)
                opt.step()
                if ep%max(1,epochs//4)==0 or ep==epochs-1:
                    hist.append({"dagger_round":j,"epoch":ep,"loss":total})
            log(f"[DAGGER:{label}] round {j}/{n_rounds} p_student={p_student:.3f} sequences={len(dataset)} loss={total:.5g}")
    return {"history":hist,"n_sequences":len(dataset),"sample_ids":sample_ids,"partition":"value"}


# ============================================================================================
# round artefacts
# ============================================================================================


class RoundState:
    """Everything a round produces (model, heads, health, support, scorer, base policy, envelope)."""

    def __init__(self, cfg: Config, geom: ActionGeometry, device):
        self.cfg, self.geom, self.device = cfg, geom, device
        sr = [c * (cfg.M + 1) for c in range(cfg.Kc)]                           # restart–residual coordinate of each role
        self.model = TransitionModel(cfg.dx, cfg.dz, geom.du, cfg.dc, cfg.rB, cfg.gamma_A, cfg.gamma_B, cfg.Ro_max, cfg.sigma_min_C,
                                     geom.tau, sr).to(device)
        ctx_dim = 1 + cfg.dx + cfg.wA + cfg.Kc * geom.p
        self.value = ConvexHead(cfg.dz, cfg.head_width, ctx_dim).to(device)
        self.stage = ConvexHead(cfg.dz, cfg.head_width, 0).to(device)
        self.health = HealthRegion(cfg.dz, cfg.dy, cfg.lam_psi)
        self.support = SupportDistance(geom.du, cfg.lam_sup, cfg.sup_quantile)
        self.scorer = ReliabilityScorer(cfg.dz, cfg.dx)
        self.base: Optional[BasePolicyGRU] = None
        self.student_tw: Optional[BasePolicyGRU] = None
        self.envelope = Envelope((0.0,))
        self.x_max = 10.0

    def new_policy(self, ref_raw: torch.Tensor) -> BasePolicyGRU:
        return BasePolicyGRU(self.geom, self.cfg.dx, self.cfg.wA, init_raw=ref_raw).to(self.device)

    def state_dict(self) -> Dict:
        return {"model": self.model.state_dict(), "value": self.value.state_dict(), "stage": self.stage.state_dict(),
                "health": self.health.state_dict(), "support": self.support.state_dict(), "scorer": self.scorer.state_dict(),
                "base": None if self.base is None else self.base.state_dict(), "student_tw": None if self.student_tw is None else self.student_tw.state_dict(),
                "envelope": asdict(self.envelope), "x_max": self.x_max}

    def load_state_dict(self, d: Dict, ref_raw: torch.Tensor):
        self.model.load_state_dict(d["model"]); self.value.load_state_dict(d["value"]); self.stage.load_state_dict(d["stage"])
        self.health.load_state_dict(d["health"]); self.support.load_state_dict(d["support"]); self.scorer.load_state_dict(d["scorer"])
        self.base = None
        self.student_tw = None
        if d.get("base") is not None:
            self.base = self.new_policy(ref_raw); self.base.load_state_dict(d["base"])
        if d.get("student_tw") is not None:
            self.student_tw = self.new_policy(ref_raw); self.student_tw.load_state_dict(d["student_tw"])
        e = d["envelope"]
        self.envelope = Envelope(tuple(e["slopes"]), tuple(e["edges"]), tuple(tuple(r) for r in e["ramps"]))
        self.x_max = d.get("x_max", 10.0)


def provisional_envelope(cfg: Config, rs: RoundState, bd: BranchData, normalizer: Normalizer, device, planner_cfg: PlannerConfig) -> Envelope:
    """Single provisional slope from the continued branches: the q_t-quantile of |e|/‖Δu‖₁ with
    e = [Q̂(c) − Q̂(k)] − [Q(c) − Q(k)] (used only inside rounds; the frozen policy is calibrated later)."""
    geom = rs.geom
    scores = []
    with torch.no_grad():
        for r in range(bd.n_states()):
            m = bd.cont_br[r]
            if not m.any():
                continue
            t = bd.t[r:r + 1].to(device)
            ell = int(bd.ell[r])
            c = plant_ctx_from_bd(bd, cfg, device)[r:r + 1]
            prior = rs.model.x_init(c) if ell == 0 else bd.x_prior[r:r+1].to(device)
            x, _ = rs.model.correct(prior, normalizer(bd.z[r:r+1].to(device)))
            ctx1 = bd.ctx_c[r:r + 1].to(device)
            dz0 = torch.tensor([ell == 0], device=device)
            u_c = bd.uH_c[r][0:1].to(device)
            u_k = bd.u_br[r][m].to(device)
            xc = rs.model.predict(x, t, u_c, c, dz0)
            xk = rs.model.predict(x.expand(u_k.shape[0], -1), t.expand(u_k.shape[0]), u_k, c.expand(u_k.shape[0], -1), dz0.expand(u_k.shape[0]))
            q_c = (1 - cfg.hazard) * rs.value(xc @ rs.model.C.t(), ctx1)
            q_k = (1 - cfg.hazard) * rs.value(xk @ rs.model.C.t(), ctx1.expand(u_k.shape[0], -1))
            pred = q_c - q_k
            real = (1 - cfg.hazard) * (bd.G_c[r] - bd.G_br[r][m]).to(device)
            e = (pred - real).abs() + 2 * (1 - cfg.hazard) ** cfg.T_cap
            dn = geom.norm_1sigma(geom.from_tangent(u_k), geom.from_tangent(u_c.expand(u_k.shape[0], -1)))
            scores.append((e / dn.clamp_min(1e-9)).cpu())
    if not scores:
        return Envelope((float("inf"),))
    s = torch.cat(scores)
    k = int(math.ceil((s.numel() + 1) * cfg.q_t))
    lam = float(s.sort().values[k-1]) if k <= s.numel() else float("inf")
    return Envelope((lam,))


def anytime_of_policy(cfg: Config, plant: Plant, make_ctrl, sample_seeds: List[int], T: int, bb_template: Backbone) -> List[float]:
    """Truncated tuning objective; family samples are actual held-out graphs."""
    Js = []
    base_state = copy.deepcopy(bb_template.state_dict())
    with preserve_plant(plant):
        try:
            for sample in sample_seeds:
                if plant.is_family:
                    plant.bind_sample(sample, "tune")
                else:
                    plant.bind_graph(plant.reference_graph)
                    bb_template.load_state_dict(base_state)
                    perturb_backbone_(bb_template, cfg.ckpt_perturb, sample)
                plant.prepare(grad=False)
                out = run_pass(plant, make_ctrl(), T, prepared=True, sens_seed=sample, stage_losses=True)
                Js.append(anytime_objective(out["g"], cfg.hazard))
        finally:
            bb_template.load_state_dict(base_state)
            bb_template.update_sigma()
    return Js


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def fit_round_components(cfg: Config, plant: Plant, rs: RoundState, bd: BranchData, device, design: RotatingDesign) -> Dict:
    """Heads → health → model (contrast through frozen heads) → heads refit → support/scorer → L_V."""
    normalizer = plant.normalizer
    plant_ctx_from_bd(bd, cfg, device)
    info = {}
    info["heads_pre"] = fit_heads(cfg, rs.value, rs.stage, bd, normalizer, device, cfg.epochs_heads)
    cs = channel_scores(bd, normalizer)
    rs.health.fit(normalizer(bd.traj_z), bd.traj_t, cs)
    info["health"] = {"selected_channels": rs.health.sel.tolist(), "rho_c": rs.health.rho_c, "L_psi": rs.health.L_psi(),
                      "channel_scores": cs.tolist()}
    info["model"] = fit_model(cfg, rs.model, rs.value, rs.health, bd, normalizer, device, cfg.epochs_ident, design)
    info["heads"] = fit_heads(cfg, rs.value, rs.stage, bd, normalizer, device, max(1, cfg.epochs_heads // 2))
    fit_support_and_scorer(cfg, rs.support, rs.scorer, bd, rs.model, normalizer, device)
    rs.x_max = float(bd.traj_x.norm(dim=1).max()) * 1.5 + 1.0
    info["support"] = {"chi_q": rs.support.chi_q, "Minv_norm": rs.support.Minv_norm}
    info["gamma_o"] = rs.model.observer_contraction()
    return info

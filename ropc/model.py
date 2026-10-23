"""ROPC graph container, role-conditioned operator bank,
propagation block (5), action geometry (8) with exact projection, tie-inclusive sensing and the
observation map (Appendix B), and the reference / free schedules.

Every equation number refers to "Predicting Before Propagating: Role-Conditioned Operator Predictive
Control for Deep Graph Neural Networks" and its supplement. Numerical and architectural
conventions are documented in docs/METHOD.md.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

# ============================================================================================
# action geometry: polytope (8), Helmert tangent chart, exact projection (18)
# ============================================================================================


def helmert_basis(p: int) -> torch.Tensor:
    """Orthonormal basis T ∈ R^{p×(p−1)} of 1⊥.  Column k−1 is (1,…,1,−k,0,…)/√(k(k+1)) on the
    first k+1 coordinates, so column 0 = (e_0 − e_1)/√2: with coordinates ordered (s, r, transport…)
    it is the restart–residual contrast that Lemma 2 nulls at ℓ = 0."""
    T = torch.zeros(p, p - 1, dtype=torch.float64)
    for k in range(1, p):
        T[:k, k - 1] = 1.0
        T[k, k - 1] = -float(k)
        T[:, k - 1] /= math.sqrt(k * (k + 1))
    return T


class ActionGeometry:
    """Role-wise actions a = (a_s, a_r, a_1, …, a_M) ∈ A of (8); A ∈ R^{Kc×(M+2)}; tangent chart
    u = (I_Kc ⊗ Tᵀ) vec(A) ∈ R^{du}, du = Kc(M+1); role-summed norm ‖·‖_{1,Σ}."""

    def __init__(self, M: int, Kc: int, alpha_min: float, alpha_max: float, r_min: float):
        self.M, self.Kc = int(M), int(Kc)
        self.p = self.M + 2
        self.du = self.Kc * (self.M + 1)
        self.alpha_min, self.alpha_max, self.r_min = float(alpha_min), float(alpha_max), float(r_min)
        if self.M < 1 or self.Kc < 1 or not (0 <= self.alpha_min <= self.alpha_max <= 1) or self.r_min < 0:
            raise ValueError("Invalid action dimensions, source bounds, or transport floor")
        if (1 + self.r_min) * self.alpha_min > 1 + 1e-12:
            raise ValueError("The source lower bound and transport floor make the action polytope empty")
        self.T = helmert_basis(self.p)                               # retain a precise chart; cast at tensor-use sites
        self.n_vec = torch.zeros(self.p)
        self.n_vec[0] = 1.0 + self.r_min
        self.n_vec[1] = 1.0
        self.as_cap = min(self.alpha_max, 1.0 / (1.0 + self.r_min))  # a_r ≥ 0 and the floor imply a_s ≤ 1/(1+r_min)
        self.tau = self.T.abs().max(0).values                         # |u_j| ≤ τ_j on the simplex
        self.lo = torch.zeros(self.p)
        self.lo[0] = self.alpha_min
        self.hi = torch.ones(self.p)
        self.hi[0] = self.alpha_max

    # ---- device helpers
    def to(self, device):
        self.T = self.T.to(device)
        self.n_vec = self.n_vec.to(device)
        self.tau = self.tau.to(device)
        self.lo = self.lo.to(device)
        self.hi = self.hi.to(device)
        return self

    # ---- charts
    def to_tangent(self, A: torch.Tensor) -> torch.Tensor:
        """A: (..., Kc, p) → u: (..., Kc·(M+1))."""
        return (A @ self.T.to(A)).flatten(-2)

    def from_tangent(self, u: torch.Tensor) -> torch.Tensor:
        U = u.reshape(*u.shape[:-1], self.Kc, self.M + 1)
        return U @ self.T.to(u).T + 1.0 / self.p

    def norm_1sigma(self, A: torch.Tensor, Bm: torch.Tensor) -> torch.Tensor:
        """‖A − B‖_{1,Σ} = Σ_c ‖a_c − b_c‖_1 over the last two dims."""
        return (A - Bm).abs().sum(dim=(-1, -2))

    def feasibility_violation(self, A: torch.Tensor) -> torch.Tensor:
        """Max violation of (8) over the last two dims (0 when feasible)."""
        v = torch.zeros(A.shape[:-2], device=A.device, dtype=A.dtype)
        v = torch.maximum(v, (-A).clamp_min(0).amax(dim=(-1, -2)))
        v = torch.maximum(v, (A.sum(-1) - 1.0).abs().amax(-1))
        v = torch.maximum(v, (self.alpha_min - A[..., 0]).clamp_min(0).amax(-1))
        v = torch.maximum(v, (A[..., 0] - self.alpha_max).clamp_min(0).amax(-1))
        v = torch.maximum(v, ((1 + self.r_min) * A[..., 0] + A[..., 1] - 1.0).clamp_min(0).amax(-1))
        return v

    # ---- smooth, surjective-onto-the-interior parameterisation (used by schedules and the GRU base)
    def parameterize(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (..., Kc, p) unconstrained → feasible A.  a_s ∈ [α_min, min(α_max, 1/(1+r_min))],
        a_r ∈ [0, 1 − (1+r_min) a_s], transport mass 1 − a_s − a_r split by a softmax."""
        a_s = self.alpha_min + (self.as_cap - self.alpha_min) * torch.sigmoid(raw[..., 0])
        a_r = (1.0 - (1.0 + self.r_min) * a_s) * torch.sigmoid(raw[..., 1])
        a_t = (1.0 - a_s - a_r).unsqueeze(-1) * torch.softmax(raw[..., 2:], dim=-1)
        return torch.cat([a_s.unsqueeze(-1), a_r.unsqueeze(-1), a_t], dim=-1)

    def unparameterize(self, A: torch.Tensor) -> torch.Tensor:
        """Approximate inverse of parameterize (for initialising schedules / students from actions)."""
        eps = 1e-5
        s = ((A[..., 0] - self.alpha_min) / max(self.as_cap - self.alpha_min, eps)).clamp(eps, 1 - eps)
        cap_r = (1.0 - (1.0 + self.r_min) * A[..., 0]).clamp_min(eps)
        r = (A[..., 1] / cap_r).clamp(eps, 1 - eps)
        t = A[..., 2:].clamp_min(eps)
        raw = torch.cat([torch.logit(s).unsqueeze(-1), torch.logit(r).unsqueeze(-1), torch.log(t)], dim=-1)
        return raw

    # ---- projection (18): KKT enumeration with a checked numerical fallback
    def project(self, A_prime: torch.Tensor, lo: Optional[torch.Tensor] = None, hi: Optional[torch.Tensor] = None,
                beta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        """Project A_prime (..., Kc, p) blockwise onto {lo ≤ a ≤ hi, 1ᵀa = 1, nᵀa ≤ beta}.
        lo/hi default to the polytope bounds, beta to 1 (trust-region boxes tighten lo/hi; erosion
        tightens all three).  Returns (A, number_of_fallback_rows)."""
        shp = A_prime.shape
        ap = A_prime.reshape(-1, self.p)
        lo_ = (self.lo.expand_as(ap) if lo is None else lo.reshape(-1, self.p).expand_as(ap)).to(ap)
        hi_ = (self.hi.expand_as(ap) if hi is None else hi.reshape(-1, self.p).expand_as(ap)).to(ap)
        bt = torch.ones(ap.shape[0], device=ap.device, dtype=ap.dtype) if beta is None else beta.reshape(-1).expand(ap.shape[0]).to(ap)
        a, nfb = project_block(ap, lo_, hi_, self.n_vec.to(ap), bt)
        return a.reshape(shp), nfb


def _box_simplex_multiplier(ap: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """y such that Σ_k clip(ap_k − y, lo_k, hi_k) = target (piecewise-linear, non-increasing in y)."""
    Bn, p = ap.shape
    bp, _ = torch.cat([ap - hi, ap - lo], 1).sort(dim=1)                  # (B, 2p) ascending
    vals = torch.minimum(torch.maximum(ap[:, None, :] - bp[:, :, None], lo[:, None, :]), hi[:, None, :]).sum(-1)
    tgt = target[:, None]
    cnt = (vals >= tgt - 1e-12).sum(1)
    j = (cnt - 1).clamp(0, 2 * p - 2)
    idx = torch.arange(Bn, device=ap.device)
    F0, F1, b0, b1 = vals[idx, j], vals[idx, j + 1], bp[idx, j], bp[idx, j + 1]
    denom = F0 - F1
    frac = torch.where(denom > 1e-12, (F0 - target) / denom.clamp_min(1e-12), torch.zeros_like(denom)).clamp(0, 1)
    y = b0 + frac * (b1 - b0)
    y = torch.where(cnt == 0, bp[:, 0], y)
    return y


def project_block(ap: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, n_vec: torch.Tensor, beta: torch.Tensor,
                  tol: float = 1e-7) -> Tuple[torch.Tensor, int]:
    """Euclidean projection of each row of ap (B, p) onto {lo ≤ a ≤ hi, 1ᵀa = 1, nᵀa ≤ beta} with
    n = (n_s, n_r, 0, …, 0).  KKT system (18): a = clip(ap − y1·1 − y2·n, lo, hi)."""
    # Solve the small KKT systems in double precision. Thin trust boxes and r_min=16
    # otherwise amplify float32 cancellation and can return an infeasible action.
    result_dtype = ap.dtype
    ap, lo, hi, n_vec, beta = (v.to(torch.float64) for v in (ap, lo, hi, n_vec, beta))
    if not all(bool(torch.isfinite(v).all()) for v in (ap, lo, hi, n_vec, beta)):
        raise ValueError("Projection inputs must be finite")
    if bool((lo > hi + 1e-12).any()) or bool((lo.sum(1) > 1 + 2e-7).any()) or bool((hi.sum(1) < 1 - 2e-7).any()):
        raise ValueError("Empty box-simplex intersection")
    # Minimize n^T a over the box-simplex to reject an empty eroded polytope.
    cheapest = lo.clone()
    remaining = (1.0 - lo.sum(1)).clamp_min(0)
    for k in n_vec.argsort().tolist():
        add = torch.minimum(remaining, hi[:, k] - lo[:, k])
        cheapest[:, k] += add
        remaining -= add
    if bool((cheapest @ n_vec > beta + 1e-9).any()):
        raise ValueError("The action/trust/erosion constraints have an empty intersection")
    Bn, p = ap.shape
    M = p - 2
    dev, dt = ap.device, ap.dtype
    n0, n1 = float(n_vec[0]), float(n_vec[1])
    # ---------------- case A: y2 = 0 (floor inactive)
    y1A = _box_simplex_multiplier(ap, lo, hi, torch.ones_like(beta))
    aA = torch.minimum(torch.maximum(ap - y1A[:, None], lo), hi)
    okA = (aA @ n_vec <= beta + tol) & ((aA.sum(1) - 1.0).abs() <= 1e-6)
    # ---------------- case B: y2 > 0 (floor active): enumerate 3×3 bound states × (2M+1) transport segments
    apT, loT, hiT = ap[:, 2:], lo[:, 2:], hi[:, 2:]
    if M > 0:
        bpT, _ = torch.cat([apT - hiT, apT - loT], 1).sort(dim=1)           # (B, 2M)
        left = torch.cat([bpT[:, :1] - 1.0, bpT], 1)
        right = torch.cat([bpT, bpT[:, -1:] + 1.0], 1)
    else:
        bpT = torch.zeros(Bn, 0, device=dev, dtype=dt)
        left = torch.full((Bn, 1), -1e9, device=dev, dtype=dt)
        right = torch.full((Bn, 1), 1e9, device=dev, dtype=dt)
    S = left.shape[1]                                                       # 2M+1 segments
    ymid = 0.5 * (left + right)
    sh = apT[:, None, :] - ymid[:, :, None]                                 # (B, S, M)
    aTmid = torch.minimum(torch.maximum(sh, loT[:, None, :]), hiT[:, None, :])
    free = (sh > loT[:, None, :]) & (sh < hiT[:, None, :])
    kappa = free.to(dt).sum(-1)                                             # (B, S)
    alpha = torch.where(free, apT[:, None, :], aTmid).sum(-1)               # (B, S)

    def coeffs(apk, lok, hik, nk):
        c = torch.stack([lok, apk, hik], 1)                                   # (B, 3)
        pc = torch.tensor([0.0, -1.0, 0.0], device=dev, dtype=dt)[None].expand(Bn, 3)
        q = torch.tensor([0.0, -nk, 0.0], device=dev, dtype=dt)[None].expand(Bn, 3)
        return c, pc, q

    cs, ps, qs = coeffs(ap[:, 0], lo[:, 0], hi[:, 0], n0)
    cr, pr, qr = coeffs(ap[:, 1], lo[:, 1], hi[:, 1], n1)
    cs_, ps_, qs_ = cs[:, :, None, None], ps[:, :, None, None], qs[:, :, None, None]
    cr_, pr_, qr_ = cr[:, None, :, None], pr[:, None, :, None], qr[:, None, :, None]
    kap_, alp_ = kappa[:, None, None, :], alpha[:, None, None, :]
    P11 = ps_ + pr_ - kap_
    P12 = qs_ + qr_
    R1 = 1.0 - cs_ - cr_ - alp_
    P21 = n0 * ps_ + n1 * pr_
    P22 = n0 * qs_ + n1 * qr_
    R2 = beta[:, None, None, None] - n0 * cs_ - n1 * cr_
    det = P11 * P22 - P12 * P21
    nonsing = det.abs() > 1e-10
    det_s = torch.where(nonsing, det, torch.ones_like(det))
    y1 = (R1 * P22 - P12 * R2) / det_s
    y2 = (P11 * R2 - P21 * R1) / det_s
    # singular (both s and r clipped): y1 from the simplex equation, y2 from the clip consistency bounds
    kap_s = torch.where(kap_ > 0, kap_, torch.ones_like(kap_))
    y1_sing = torch.where(kap_ > 0, (cs_ + cr_ + alp_ - 1.0) / kap_s, torch.zeros_like(kap_s)).expand_as(y1)
    # lower bounds on y2 from "lo" states, upper from "hi" states (state index 0 = lo, 2 = hi)
    big = torch.full_like(y1, 1e9)
    ap_s, ap_r = ap[:, 0][:, None, None, None], ap[:, 1][:, None, None, None]
    lo_s, hi_s = lo[:, 0][:, None, None, None], hi[:, 0][:, None, None, None]
    lo_r, hi_r = lo[:, 1][:, None, None, None], hi[:, 1][:, None, None, None]
    st_s = torch.arange(3, device=dev)[None, :, None, None]
    st_r = torch.arange(3, device=dev)[None, None, :, None]
    lb_s = torch.where(st_s == 0, (ap_s - y1_sing - lo_s) / max(n0, 1e-12), -big)
    lb_r = torch.where(st_r == 0, (ap_r - y1_sing - lo_r) / max(n1, 1e-12), -big)
    ub_s = torch.where(st_s == 2, (ap_s - y1_sing - hi_s) / max(n0, 1e-12), big)
    ub_r = torch.where(st_r == 2, (ap_r - y1_sing - hi_r) / max(n1, 1e-12), big)
    y2_sing = torch.maximum(torch.maximum(lb_s, lb_r), torch.full_like(y1, 1e-9))
    sing_ok = (ub_s >= y2_sing - 1e-9) & (ub_r >= y2_sing - 1e-9) & (R2.abs() <= 1e-7) & (st_s != 1) & (st_r != 1)
    y1 = torch.where(nonsing, y1, y1_sing)
    y2 = torch.where(nonsing, y2, y2_sing)
    # candidate actions and violations
    y1b = y1[..., None]
    y2b = y2[..., None]
    a_cand = ap[:, None, None, None, :] - y1b - y2b * n_vec[None, None, None, None, :]
    a_cand = torch.minimum(torch.maximum(a_cand, lo[:, None, None, None, :]), hi[:, None, None, None, :])
    v_sum = (a_cand.sum(-1) - 1.0).abs()
    v_floor = (a_cand @ n_vec - beta[:, None, None, None]).abs()
    # segment consistency of y1 and bound-state consistency of s and r
    seg_ok = (y1 >= left[:, None, None, :] - 1e-9) & (y1 <= right[:, None, None, :] + 1e-9)
    raw_s = ap_s - y1 - n0 * y2
    raw_r = ap_r - y1 - n1 * y2
    cons_s = torch.where(st_s == 1, (raw_s >= lo_s - 1e-9) & (raw_s <= hi_s + 1e-9),
                         torch.where(st_s == 0, raw_s <= lo_s + 1e-9, raw_s >= hi_s - 1e-9))
    cons_r = torch.where(st_r == 1, (raw_r >= lo_r - 1e-9) & (raw_r <= hi_r + 1e-9),
                         torch.where(st_r == 0, raw_r <= lo_r + 1e-9, raw_r >= hi_r - 1e-9))
    ok = seg_ok & cons_s & cons_r & (y2 > 1e-12) & (nonsing | sing_ok)
    viol = v_sum + v_floor + torch.where(ok, torch.zeros_like(v_sum), torch.full_like(v_sum, 1e3))
    viol_flat = viol.reshape(Bn, -1)
    best = viol_flat.argmin(1)
    aB = a_cand.reshape(Bn, -1, p)[torch.arange(Bn, device=dev), best]
    okB = viol_flat[torch.arange(Bn, device=dev), best] <= 1e-6
    a = torch.where(okA[:, None], aA, aB)
    need_fb = ~(okA | okB)
    nfb = int(need_fb.sum())
    if nfb > 0:
        a[need_fb] = _dykstra_fallback(ap[need_fb], lo[need_fb], hi[need_fb], n_vec, beta[need_fb])
    residual = torch.maximum((a.sum(1) - 1).abs(), (a @ n_vec - beta).clamp_min(0))
    residual = torch.maximum(residual, torch.maximum((lo - a).clamp_min(0).amax(1), (a - hi).clamp_min(0).amax(1)))
    if bool((residual > 2e-7).any()):
        raise RuntimeError("Action projection did not reach its feasibility tolerance")
    return a.to(result_dtype), nfb


def _dykstra_fallback(ap, lo, hi, n_vec, beta, iters: int = 80):
    """Compatibility name for a bounded dual bisection fallback, always in float64.

    For a fixed nonnegative half-space multiplier, the remaining problem is a
    box-simplex projection. Its floor value is monotone in that multiplier.
    Unlike a fixed number of alternating projections, this preserves the box
    and simplex at every iteration; the caller also verifies the final residual.
    """
    def at(mult):
        shifted = ap - mult[:, None] * n_vec
        y = _box_simplex_multiplier(shifted, lo, hi, torch.ones_like(beta))
        return torch.minimum(torch.maximum(shifted - y[:, None], lo), hi)

    lower = torch.zeros_like(beta)
    upper = torch.ones_like(beta)
    for _ in range(80):
        need = at(upper) @ n_vec > beta + 1e-12
        if not bool(need.any()):
            break
        upper = torch.where(need, 2 * upper, upper)
    else:
        raise RuntimeError("Could not bracket the action projection multiplier")
    for _ in range(iters):
        mid = (lower + upper) / 2
        need = at(mid) @ n_vec > beta
        lower = torch.where(need, mid, lower)
        upper = torch.where(need, upper, mid)
    return at(upper)


# ============================================================================================
# graph container and union-support sparse structure
# ============================================================================================


@dataclass
class GraphData:
    name: str
    x: torch.Tensor                 # (n, F) float32
    y: torch.Tensor                 # task-appropriate targets; NaN masks missing regression/multilabel entries
    edge_index: torch.Tensor        # (2, m) long; arc j → i stored as (edge_index[0]=j (source), edge_index[1]=i (target))
    num_nodes: int
    num_classes: int
    train_idx: torch.Tensor
    valid_idx: torch.Tensor
    test_idx: torch.Tensor
    batch: Optional[torch.Tensor] = None   # graph id per node (families); None = one graph
    num_graphs: int = 1
    task_type: str = "node_classification"
    metric: str = "accuracy"
    metric_mode: str = "max"
    edge_label_index: Optional[torch.Tensor] = None
    edge_label: Optional[torch.Tensor] = None
    graph_ids: Optional[torch.Tensor] = None
    source: object = None
    edge_attr: Optional[torch.Tensor] = None
    sample_id: object = None
    task_idx: Optional[torch.Tensor] = None
    value_idx: Optional[torch.Tensor] = None

    def to(self, device):
        for k in ("x", "y", "edge_index", "train_idx", "valid_idx", "test_idx", "batch", "edge_label_index", "edge_label", "graph_ids", "edge_attr", "task_idx", "value_idx"):
            v = getattr(self, k, None)
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        self.__dict__.pop("_struct_cache", None)
        return self

    @property
    def batch_vec(self) -> torch.Tensor:
        if self.batch is None:
            return torch.zeros(self.num_nodes, dtype=torch.long, device=self.x.device)
        return self.batch

    def degrees(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.num_nodes
        src, dst = self.edge_index[0], self.edge_index[1]
        deg_out = torch.zeros(n, device=self.x.device).index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg_in = torch.zeros(n, device=self.x.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        return deg_in, deg_out


class UnionSupport:
    """CSR structure on the union support of A, Aᵀ and I (row = receiving node), with the positions of
    the A arcs, the Aᵀ arcs and the diagonal inside the union, and the transpose permutation.  All
    sparse kernels share this one structure and are fused before a single SpMM (Appendix I)."""

    def __init__(self, edge_index: torch.Tensor, n: int):
        dev = edge_index.device
        src, dst = edge_index[0].long(), edge_index[1].long()
        keyA = dst * n + src                # (row=i=target, col=j=source): arc j → i
        keyAT = src * n + dst               # reverse arcs
        keyI = torch.arange(n, device=dev) * (n + 1)
        keys = torch.unique(torch.cat([keyA, keyAT, keyI]))   # sorted ascending = CSR order
        self.n = n
        self.nnz = int(keys.numel())
        self.row = (keys // n).long()
        self.col = (keys % n).long()
        counts = torch.bincount(self.row, minlength=n)
        self.crow = torch.zeros(n + 1, dtype=torch.long, device=dev)
        self.crow[1:] = torch.cumsum(counts, 0)
        self.posA = torch.searchsorted(keys, keyA)              # entries of A (arc j→i), may repeat
        self.posAT = torch.searchsorted(keys, keyAT)
        self.posI = torch.searchsorted(keys, keyI)
        self.tperm = torch.searchsorted(keys, self.col * n + self.row)   # entry of (col,row)
        self.keys = keys
        self.m = int(src.numel())
        self.src, self.dst = src, dst

    def row_sums(self, vals: torch.Tensor) -> torch.Tensor:
        """Local CSR segment sums, without subtracting large global float32 prefixes."""
        return torch.segment_reduce(vals, "sum", lengths=self.crow[1:] - self.crow[:-1])

    def normalized_values(self, arc_pos: torch.Tensor, arc_w: torch.Tensor, self_loop: float) -> torch.Tensor:
        """Values of R(A_w + γ_sl I) on the union support, arcs given by positions/weights."""
        v = torch.zeros(self.nnz, device=arc_w.device, dtype=arc_w.dtype)
        v = v.index_add(0, arc_pos, arc_w)
        v = v.index_add(0, self.posI, torch.full((self.n,), float(self_loop), device=v.device, dtype=v.dtype))
        rs = self.row_sums(v)
        return v / rs[self.row].clamp_min(1e-12)


class FusedSpMM(torch.autograd.Function):
    """Y = S X on a fused CSR support; backward uses the explicit transpose
    permutation and chunked value gradients. Backend determinism is measured by
    replay tests on the actual device, rather than inferred from the CSR format.
    """

    @staticmethod
    def _spmm(crow, col, vals, n, X):
        S = torch.sparse_csr_tensor(crow, col, vals, (n, n))
        try:
            return torch.sparse.mm(S, X)
        except RuntimeError:                       # CSR matmul unavailable for this build/device: COO path
            return torch.sparse.mm(S.to_sparse_coo().coalesce(), X)

    @staticmethod
    def forward(ctx, vals, X, crow, col, tperm, row):
        n = X.shape[0]
        Y = FusedSpMM._spmm(crow, col, vals, n, X)
        ctx.save_for_backward(vals, X, crow, col, tperm, row)
        return Y

    @staticmethod
    def backward(ctx, gY):
        vals, X, crow, col, tperm, row = ctx.saved_tensors
        n = X.shape[0]
        gX = gV = None
        if ctx.needs_input_grad[1]:
            gX = FusedSpMM._spmm(crow, col, vals[tperm].contiguous(), n, gY.contiguous())
        if ctx.needs_input_grad[0]:
            nnz = vals.numel()
            gV = torch.empty_like(vals)
            chunk = max(1, int(4.0e7 // max(1, X.shape[1])))
            for s in range(0, nnz, chunk):
                e = min(nnz, s + chunk)
                gV[s:e] = (gY[row[s:e]] * X[col[s:e]]).sum(1)
        return gV, gX, None, None, None, None


def fused_spmm(vals: torch.Tensor, X: torch.Tensor, sup: UnionSupport) -> torch.Tensor:
    return FusedSpMM.apply(vals, X, sup.crow, sup.col, sup.tperm, sup.row)


# ============================================================================================
# spectrally bounded linear maps (‖W‖₂ ≤ 1), checkpoint-safe
# ============================================================================================


class SpectralLinear(nn.Module):
    """Linear map normalized by a certified upper bound on the spectral norm.

    update_sigma() computes an SVD at the optimizer-step boundary. Between updates,
    ||W_ref||_2 + ||W-W_ref||_F remains an upper bound, including after an external
    parameter edit. Forward calls never change state, so checkpoint replay uses
    the same weights and normalizer. The reference SVD is detached from training.
    """

    def __init__(self, in_f: int, out_f: int, n_power: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.orthogonal_(self.weight)
        self.register_buffer("u", F.normalize(torch.randn(out_f), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(in_f), dim=0))
        self.register_buffer("weight_ref", self.weight.detach().clone())
        self.register_buffer("sigma_ref", torch.ones(()))
        self.n_power = n_power
        self.update_sigma(20)

    @torch.no_grad()
    def update_sigma(self, iters: Optional[int] = None):
        U, S, Vh = torch.linalg.svd(self.weight, full_matrices=False)
        self.u.copy_(U[:, 0])
        self.v.copy_(Vh[0])
        self.weight_ref.copy_(self.weight)
        self.sigma_ref.copy_(S[0] * (1 + 4 * torch.finfo(S.dtype).eps))

    def sigma(self) -> torch.Tensor:
        return self.sigma_ref + torch.linalg.vector_norm(self.weight - self.weight_ref)

    def effective_weight(self) -> torch.Tensor:
        s = self.sigma()
        return self.weight / torch.clamp(s, min=1.0)

    def forward(self, x):
        return x @ self.effective_weight().t()


def exact_spectral_norm(W: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(W.detach().float(), ord=2))


# ============================================================================================
# roles: degree-corrected mixed memberships (Section IV-A), structural likelihood (Appendix E)
# ============================================================================================


class RoleEncoder(nn.Module):
    """Θ_f (simplex rows, fine support s), Π (row-stochastic), Θ_c = Θ_f Π, compatibilities
    B ∈ [b_min, 1]^{Kf×Kf} (symmetric), propensities ν_i, intercept β₀ and slope β₁ ≥ 0."""

    def __init__(self, F_in: int, Kf: int, Kc: int, b_min: float, hidden: int = 64, support_s: int = 2, eps_g: float = 0.5):
        super().__init__()
        self.Kf, self.Kc, self.b_min, self.support_s, self.eps_g = Kf, Kc, b_min, support_s, eps_g
        self.mlp = nn.Sequential(nn.Linear(F_in + 2, hidden), nn.ReLU(), nn.Linear(hidden, Kf))
        self.Pi_raw = nn.Parameter(torch.randn(Kf, Kc) * 0.5)
        self.B_raw = nn.Parameter(torch.eye(Kf) * 3.0 - 1.5)
        self.nu = nn.Linear(2, 1)
        self.beta0 = nn.Parameter(torch.tensor(-6.0))
        self.beta1_raw = nn.Parameter(torch.tensor(2.0))

    def compat(self) -> torch.Tensor:
        Bs = 0.5 * (self.B_raw + self.B_raw.t())
        return self.b_min + (1.0 - self.b_min) * torch.sigmoid(Bs)

    def hierarchy(self) -> torch.Tensor:
        return torch.softmax(self.Pi_raw, dim=1)

    def beta1(self) -> torch.Tensor:
        return F.softplus(self.beta1_raw)

    def forward(self, x: torch.Tensor, logdeg: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.mlp(torch.cat([x, logdeg], 1))
        theta = torch.softmax(logits, dim=1)
        if 0 < self.support_s < self.Kf:
            top = theta.topk(self.support_s, dim=1).indices
            mask = torch.zeros_like(theta).scatter_(1, top, 1.0)
            theta = theta * mask
            theta = theta / theta.sum(1, keepdim=True).clamp_min(1e-12)
        Pi = self.hierarchy()
        Theta_c = theta @ Pi
        return {"Theta_f": theta, "Theta_c": Theta_c, "Pi": Pi, "B": self.compat(), "nu": self.nu(logdeg).squeeze(1)}

    def gate(self, Theta: torch.Tensor, Bm: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """g_ij = ε_g + (1 − ε_g) θ_iᵀ B θ_j on the given arcs (role shrinkage toward P0 through ε_g)."""
        p = ((Theta[dst] @ Bm) * Theta[src]).sum(1)
        return self.eps_g + (1.0 - self.eps_g) * p

    def struct_loss(self, graph: GraphData, roles: Dict[str, torch.Tensor], S_pos: int, S_neg: int,
                    gen: torch.Generator) -> torch.Tensor:
        """Unbiased Bernoulli NLL over ordered, distinct, within-graph node pairs.

        Positive arcs and true nonedges are sampled uniformly with replacement in
        their respective populations. Population/sample weights give the stated
        Horvitz–Thompson estimator; cross-graph pairs are never negative examples.
        """
        n = graph.num_nodes
        dev = graph.x.device
        Theta, Bm, nu = roles["Theta_f"], roles["B"], roles["nu"]
        b0, b1 = self.beta0, self.beta1()
        batch = graph.batch_vec
        signature = (graph.edge_index.data_ptr(), graph.edge_index._version, str(dev))
        cache = getattr(graph, "_struct_cache", None)
        if cache is None or cache[0] != signature:
            src, dst = graph.edge_index
            if bool((batch[src] != batch[dst]).any()):
                raise ValueError("Graph batches must not contain cross-graph edges")
            use = src != dst
            keys = torch.unique(src[use] * n + dst[use], sorted=True)
            sizes = torch.bincount(batch, minlength=graph.num_graphs)
            nodes = batch.argsort(stable=True)
            offsets = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), sizes.cumsum(0)])
            population = sizes * (sizes - 1)
            cache = (signature, keys, sizes, nodes, offsets, population)
            graph._struct_cache = cache
        _, keys, sizes, nodes, offsets, population = cache
        total, positive = int(population.sum()), keys.numel()
        negative = total - positive
        zero = b0 * 0.0
        if total == 0:
            return zero

        def pair_loss(pair_keys, positive_label):
            i, j = pair_keys // n, pair_keys % n
            prob = ((Theta[i] @ Bm) * Theta[j]).sum(1)
            logits = nu[i] + nu[j] + b0 + b1 * prob
            return F.softplus(-logits if positive_label else logits).mean()

        loss = zero
        if positive:
            if S_pos < 1:
                raise ValueError("Positive structural sample budget must be positive")
            ids = torch.randint(positive, (min(S_pos, positive),), generator=gen, device=dev)
            loss = loss + (positive / total) * pair_loss(keys[ids], True)
        if negative:
            if S_neg < 1:
                raise ValueError("Negative structural sample budget must be positive")
            chosen, remaining = [], int(S_neg)
            weights = population.to(torch.float64)
            # Rejection is uniform on true nonedges. Sparse graph datasets accept
            # nearly every proposal; a bounded dense-graph fallback is below.
            for _ in range(64):
                count = max(32, 2 * remaining)
                gids = torch.multinomial(weights, count, replacement=True, generator=gen)
                ns = sizes[gids]
                ii = (torch.rand(count, generator=gen, device=dev) * ns).long()
                jj = (torch.rand(count, generator=gen, device=dev) * (ns - 1)).long()
                jj += (jj >= ii).long()
                candidates = nodes[offsets[gids] + ii] * n + nodes[offsets[gids] + jj]
                loc = torch.searchsorted(keys, candidates)
                is_edge = (loc < positive) & (keys[loc.clamp_max(max(positive - 1, 0))] == candidates) if positive else torch.zeros_like(loc, dtype=torch.bool)
                accepted = candidates[~is_edge][:remaining]
                chosen.append(accepted)
                remaining -= accepted.numel()
                if remaining == 0:
                    break
            if remaining:
                if total > 1_000_000:
                    raise RuntimeError("Structural negative sampling exhausted its dense-graph budget")
                all_negative = []
                for gid in range(graph.num_graphs):
                    nd = nodes[offsets[gid]:offsets[gid + 1]]
                    pairs = nd[:, None] * n + nd[None, :]
                    candidates = pairs[~torch.eye(len(nd), device=dev, dtype=torch.bool)]
                    loc = torch.searchsorted(keys, candidates)
                    is_edge = (loc < positive) & (keys[loc.clamp_max(max(positive - 1, 0))] == candidates) if positive else torch.zeros_like(loc, dtype=torch.bool)
                    all_negative.append(candidates[~is_edge])
                pool = torch.cat(all_negative)
                chosen.append(pool[torch.randint(pool.numel(), (remaining,), generator=gen, device=dev)])
            loss = loss + (negative / total) * pair_loss(torch.cat(chosen), False)
        return loss


def effective_sample_size(theta_c: torch.Tensor) -> torch.Tensor:
    """ESS per coarse role: (Σ_i θ_ic)² / Σ_i θ_ic²."""
    return theta_c.sum(0) ** 2 / (theta_c.pow(2).sum(0) + 1e-12)


# ============================================================================================
# operator bank (Section IV-A): P0, P←0, Pf, Pc, P×, Pg
# ============================================================================================

KERNEL_NAMES = ["P0", "Pr0", "Pf", "Pc", "Px", "Pg"]   # P←0 is spelled Pr0; P× is spelled Px
SPARSE_KERNELS = ["P0", "Pr0", "Pf", "Pc", "Px"]


def kernel_values(name: str, sup: UnionSupport, roles: Dict[str, torch.Tensor], role_enc: RoleEncoder,
                  gamma_sl: float) -> torch.Tensor:
    """Row-stochastic values on the union support for one sparse kernel of the bank."""
    src, dst = sup.src, sup.dst
    ones = torch.ones(sup.m, device=src.device)
    if name == "P0":
        return sup.normalized_values(sup.posA, ones, gamma_sl)
    if name == "Pr0":
        return sup.normalized_values(sup.posAT, ones, gamma_sl)
    if name == "Pf":
        g = role_enc.gate(roles["Theta_f"], roles["B"], src, dst)
        return sup.normalized_values(sup.posA, g, gamma_sl)
    if name == "Pc":
        Pi = roles["Pi"]
        mass = Pi.sum(0).clamp_min(torch.finfo(Pi.dtype).tiny)
        Bc = (Pi.t() @ roles["B"] @ Pi) / (mass[:, None] * mass[None, :])
        g = role_enc.gate(roles["Theta_c"], Bc, src, dst)
        return sup.normalized_values(sup.posA, g, gamma_sl)
    if name == "Px":
        g = role_enc.gate(roles["Theta_f"], roles["B"], src, dst)
        return sup.normalized_values(sup.posAT, g, gamma_sl)
    raise ValueError(name)


def global_role_route(XW: torch.Tensor, Theta_f: torch.Tensor, Bm: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """P_g XW = diag(Θ_f B Θ_fᵀ 1)⁻¹ Θ_f B (Θ_fᵀ XW), per graph (7); node rank ≤ K_f."""
    TB = Theta_f @ Bm                                                          # (n, Kf)
    if num_graphs == 1:
        pooled = Theta_f.t() @ XW                                              # Θ_fᵀ XW  (Kf, d)
        mass = Theta_f.sum(0)                                                  # Θ_fᵀ 1   (Kf,)
        return (TB @ pooled) / (TB @ mass).unsqueeze(1).clamp_min(1e-12)
    out = torch.empty_like(XW)
    for gidx in range(num_graphs):
        nodes = (batch == gidx).nonzero(as_tuple=True)[0]
        Tg = Theta_f[nodes]
        pooled = Tg.t() @ XW[nodes]
        mass = Tg.sum(0)
        out[nodes] = (TB[nodes] @ pooled) / (TB[nodes] @ mass).unsqueeze(1).clamp_min(1e-12)
    return out


# ============================================================================================
# propagation block (5)
# ============================================================================================


class PropagationBlock(nn.Module):
    """H_{ℓ+1} = D_s H_0 + D_r H_ℓ + N(Σ_m D_m P̄_m H_ℓ W), N(y) = W₂ tanh(W₁ y), spectrally bounded."""

    def __init__(self, d: int, dff: int):
        super().__init__()
        self.W = SpectralLinear(d, d)
        self.W1 = SpectralLinear(d, dff)
        self.W2 = SpectralLinear(dff, d)

    def update_sigma(self):
        self.W.update_sigma()
        self.W1.update_sigma()
        self.W2.update_sigma()

    def forward(self, H, H0, node_act, sup: UnionSupport, kvals: Optional[torch.Tensor], sparse_slots: List[int],
                pg_slot: Optional[int], Theta_f, Bm, batch, num_graphs):
        HW = self.W(H)
        Y = torch.zeros_like(HW)
        if kvals is not None and len(sparse_slots) > 0:
            coef = node_act[:, sparse_slots]                       # (n, Ms)
            v = (coef[sup.row] * kvals.t()).sum(1)                 # fused edge values Σ_m a_{i,m} (P̄_m)_ij
            Y = fused_spmm(v, HW, sup)
        if pg_slot is not None:
            Y = Y + node_act[:, pg_slot:pg_slot + 1] * global_role_route(HW, Theta_f, Bm, batch, num_graphs)
        Nn = self.W2(torch.tanh(self.W1(Y)))
        return node_act[:, 0:1] * H0 + node_act[:, 1:2] * H + Nn


def project_ball_rows(H: torch.Tensor, R0: float) -> torch.Tensor:
    nrm = H.norm(dim=1, keepdim=True)
    return H * torch.clamp(R0 / (nrm + 1e-12), max=1.0)


# ============================================================================================
# schedules: reference β₁ (monotone 2-knot spline in t) and free spline (A1t)
# ============================================================================================


class SplineSchedule(nn.Module):
    """Piecewise-linear interpolation of feasible knot actions in t = ℓ/(ℓ+ℓ₀).

    The default two-knot reference is monotone in every action coordinate.
    Multi-knot instances are free feasible splines, used by schedule comparators.
    """

    def __init__(self, geom: ActionGeometry, n_knots: int, init_action: Optional[torch.Tensor] = None):
        super().__init__()
        self.geom = geom
        self.n_knots = n_knots
        if n_knots < 2:
            raise ValueError("A schedule needs at least two knots")
        if init_action is None:
            init_action = default_reference_actions(geom, n_knots)
        raw = geom.unparameterize(init_action)
        self.raw = nn.Parameter(raw)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: scalar tensor or (B,) → (B, Kc, p) (B = t.numel())."""
        t = t.reshape(-1).clamp(0.0, 1.0)
        pos = t * (self.n_knots - 1)
        i0 = pos.floor().long().clamp(0, self.n_knots - 2)
        w = (pos - i0.float()).unsqueeze(-1).unsqueeze(-1)
        knots = self.geom.parameterize(self.raw)
        return (1 - w) * knots[i0] + w * knots[i0 + 1]


def default_reference_actions(geom: ActionGeometry, n_knots: int) -> torch.Tensor:
    """Monotone reference initialisation: restart mass decays with depth, transport grows; feasible
    under the floor for every r_min (a_s ≤ 1/(1+r_min))."""
    out = torch.zeros(n_knots, geom.Kc, geom.p)
    for k in range(n_knots):
        f = k / max(1, n_knots - 1)
        width = geom.as_cap - geom.alpha_min
        start = max(geom.alpha_min, min(0.20, geom.as_cap - 0.05 * width))
        end = min(start, geom.alpha_min + 0.25 * width)
        a_s = (1 - f) * start + f * end
        max_r = 1.0 - (1.0 + geom.r_min) * a_s
        a_r = 0.5 * max_r
        a_t = 1.0 - a_s - a_r
        out[k, :, 0] = a_s
        out[k, :, 1] = a_r
        out[k, :, 2:] = a_t / geom.M
    return out


# ============================================================================================
# tie-inclusive sensing (Lemma 3) and the observation map (Appendix B)
# ============================================================================================


def equivariant_hash(x: torch.Tensor, edge_index: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Label-free, permutation-equivariant node score κ_i: feature/degree projection refined by two
    Weisfeiler–Leman rounds. Integer modular sums make the neighbor reduction
    independent of edge ordering, including on parallel devices. Hash collisions
    are retained by the tie-inclusive cutoff, never broken with node identifiers.
    """
    dev = x.device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.randint(-(1 << 30), 1 << 30, (x.shape[1],), generator=g, dtype=torch.int64).to(dev)
    src, dst = edge_index[0], edge_index[1]
    din = torch.bincount(dst, minlength=n)
    dout = torch.bincount(src, minlength=n)

    def mix(v):
        v = (v ^ (v >> 30)) * -4658895280553007687
        v = (v ^ (v >> 27)) * -7723592293110705685
        return v ^ (v >> 31)

    # Features enter as their float32 bit patterns; no graph/node index is used.
    bits = x.detach().float().contiguous().view(torch.int32).to(torch.int64)
    s = mix((bits * w).sum(1) + 1000003 * din + 1000033 * dout + int(seed) % (1 << 31))
    for r in range(2):
        inc = torch.zeros(n, device=dev, dtype=torch.int64).index_add_(0, dst, mix(s[src] + 97 + r))
        out = torch.zeros(n, device=dev, dtype=torch.int64).index_add_(0, src, mix(s[dst] + 193 + r))
        s = mix(s + inc * 1000037 + out * 1000039 + r)
    return s


@dataclass
class SensedSet:
    idx: torch.Tensor           # sensed node indices (sorted)
    graph_of: torch.Tensor      # graph id of each sensed node
    e_src: torch.Tensor         # sensed arcs (positions into idx), both endpoints sensed
    e_dst: torch.Tensor
    e_w: torch.Tensor           # (P0)_{dst,src} weights
    sizes: torch.Tensor         # (B,) number of sensed nodes per graph
    overshoot: float            # max |S_g| / n_sens over graphs above the cut


def build_sensed_set(graph: GraphData, n_sens: int, seed: int, gamma_sl: float = 1.0) -> SensedSet:
    if n_sens < 1 or gamma_sl <= 0:
        raise ValueError("The sensor budget and self-loop mass must be positive")
    n, dev = graph.num_nodes, graph.x.device
    batch = graph.batch_vec
    B = graph.num_graphs
    sizes_full = torch.bincount(batch, minlength=B)
    keep = torch.ones(n, dtype=torch.bool, device=dev)
    overshoot = 1.0
    if int(sizes_full.max()) > n_sens:
        kappa = equivariant_hash(graph.x, graph.edge_index, n, seed)
        for gidx in range(B):
            nodes = (batch == gidx).nonzero(as_tuple=True)[0]
            if nodes.numel() > n_sens:
                kg = kappa[nodes]
                thr = kg.sort().values[n_sens - 1]
                sel = kg <= thr                                   # tie-inclusive cut
                keep[nodes[~sel]] = False
                overshoot = max(overshoot, float(sel.sum()) / n_sens)
    idx = keep.nonzero(as_tuple=True)[0]
    pos = torch.full((n,), -1, dtype=torch.long, device=dev)
    pos[idx] = torch.arange(idx.numel(), device=dev)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    both = keep[src] & keep[dst]
    es, ed = pos[src[both]], pos[dst[both]]
    deg_in = torch.zeros(n, device=dev).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    w = 1.0 / (deg_in[dst[both]] + gamma_sl)                       # original P0 weights, without renormalizing the induced edges
    sizes = torch.bincount(batch[idx], minlength=B)
    return SensedSet(idx=idx, graph_of=batch[idx], e_src=es, e_dst=ed, e_w=w, sizes=sizes, overshoot=overshoot)


class Sensor:
    """Observation z = O(H) ∈ R^{dz}, dz = 5 + 5 Kc: five global channels (participation rank,
    source alignment, edge energy, log-variance ratio, predictive uncertainty) and five per coarse
    role (log-dispersion ratio, centroid separation, weighted RV, role uncertainty, role edge energy)."""

    def __init__(self, d: int, b: int, Kc: int, seed: int, device):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        Q, _ = torch.linalg.qr(torch.randn(d, b, generator=g))
        self.S = Q.to(device)                       # orthonormal sketch (d, b)
        self.b, self.Kc = b, Kc
        self.dz = 5 + 5 * Kc

    @staticmethod
    def _cov(O: torch.Tensor, w: torch.Tensor, gid: torch.Tensor, B: int):
        """Weighted per-graph mean and covariance.  O: (ns, b); w: (ns,) weights; gid: (ns,)."""
        b = O.shape[1]
        mass = torch.zeros(B, device=O.device, dtype=O.dtype).index_add_(0, gid, w).clamp_min(1e-12)
        mean = torch.zeros(B, b, device=O.device, dtype=O.dtype).index_add_(0, gid, w[:, None] * O) / mass[:, None]
        D = O - mean[gid]
        C = torch.zeros(B, b, b, device=O.device, dtype=O.dtype).index_add_(0, gid, (w[:, None, None] * D[:, :, None] * D[:, None, :]))
        return mean, C / mass[:, None, None], D, mass

    def observe(self, H: torch.Tensor, H0: torch.Tensor, Theta_c: torch.Tensor, logits: torch.Tensor,
                sensed: SensedSet, num_graphs: int, role_mask: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Raw observations (B,dz). logits and optional scalar uncertainty are on
        the sensed nodes. Task adapters supply multilabel/regression uncertainty;
        the default is categorical entropy (Bernoulli entropy for one logit).
        """
        idx, gid, B = sensed.idx, sensed.graph_of, num_graphs
        O = H[idx] @ self.S.to(H)
        O0 = H0[idx] @ self.S.to(H0)
        ns = idx.numel()
        ones = torch.ones(ns, device=H.device, dtype=H.dtype)
        mean, C, D, mass = self._cov(O, ones, gid, B)
        mean0, C0, D0, _ = self._cov(O0, ones, gid, B)
        Cx = torch.zeros(B, self.b, self.b, device=H.device, dtype=H.dtype).index_add_(0, gid, D[:, :, None] * D0[:, None, :]) / mass[:, None, None]
        trC = C.diagonal(dim1=1, dim2=2).sum(1)
        trC0 = C0.diagonal(dim1=1, dim2=2).sum(1)
        trC2 = (C * C).sum(dim=(1, 2))
        PR = trC ** 2 / (self.b * trC2 + EPS)
        RV = (Cx * Cx).sum(dim=(1, 2)) / (torch.sqrt(trC2) * torch.sqrt((C0 * C0).sum(dim=(1, 2))) + EPS)
        # edge energy on sensed arcs
        e_src, e_dst, e_w = sensed.e_src, sensed.e_dst, sensed.e_w.to(H)
        de = ((O[e_dst] - O[e_src]) ** 2).sum(1) * e_w if e_src.numel() > 0 else torch.zeros(0, device=H.device, dtype=H.dtype)
        gid_e = gid[e_dst] if e_src.numel() > 0 else torch.zeros(0, dtype=torch.long, device=H.device)
        nrm = (O * O).sum(1)
        E_num = torch.zeros(B, device=H.device, dtype=H.dtype).index_add_(0, gid_e, de)
        E_den = torch.zeros(B, device=H.device, dtype=H.dtype).index_add_(0, gid, nrm)
        Eg = E_num / (E_den + EPS)
        logvar = torch.log((trC + EPS) / (trC0 + EPS))
        if uncertainty is not None:
            ent = uncertainty.reshape(-1).to(H)
            if ent.numel() != ns or not bool(torch.isfinite(ent).all()):
                raise ValueError("Uncertainty must contain one finite scalar per sensed node")
        elif logits.shape[1] == 1:
            prob = torch.sigmoid(logits[:, 0])
            ent = -(prob * F.logsigmoid(logits[:, 0]) + (1 - prob) * F.logsigmoid(-logits[:, 0])) / math.log(2)
        else:
            pr = torch.softmax(logits, dim=1)
            ent = -(pr * torch.log(pr.clamp_min(1e-12))).sum(1) / math.log(logits.shape[1])
        U = torch.zeros(B, device=H.device, dtype=H.dtype).index_add_(0, gid, ent) / mass
        glob = torch.stack([PR, RV, Eg, logvar, U], 1)                           # (B, 5)
        # role channels
        Th = Theta_c[idx].to(H)                                                    # (ns, Kc)
        chans = []
        for c in range(self.Kc):
            w = Th[:, c]
            mc, Cc, Dc, mass_c = self._cov(O, w, gid, B)
            m0c, C0c, D0c, _ = self._cov(O0, w, gid, B)
            trCc = Cc.diagonal(dim1=1, dim2=2).sum(1)
            trC0c = C0c.diagonal(dim1=1, dim2=2).sum(1)
            Cxc = torch.zeros(B, self.b, self.b, device=H.device, dtype=H.dtype).index_add_(0, gid, w[:, None, None] * Dc[:, :, None] * D0c[:, None, :]) / mass_c[:, None, None]
            ldr = torch.log((trCc + EPS) / (trC0c + EPS))
            sep = ((mc - mean) ** 2).sum(1) / (trC + EPS)
            rv = (Cxc * Cxc).sum(dim=(1, 2)) / (torch.sqrt((Cc * Cc).sum(dim=(1, 2))) * torch.sqrt((C0c * C0c).sum(dim=(1, 2))) + EPS)
            unc = torch.zeros(B, device=H.device, dtype=H.dtype).index_add_(0, gid, w * ent) / mass_c
            if e_src.numel() > 0:
                den_c = torch.zeros(B, device=H.device, dtype=H.dtype).index_add_(0, gid, w * nrm)
                num_c = torch.zeros(B, device=H.device, dtype=H.dtype).index_add_(0, gid_e, w[e_dst] * de)
                ee = num_c / (den_c + EPS)
            else:
                ee = torch.zeros(B, device=H.device, dtype=H.dtype)
            ch = torch.stack([ldr, sep, rv, unc, ee], 1)
            if role_mask is not None:
                ch = ch * role_mask[:, c:c + 1].to(ch.dtype)
            chans.append(ch)
        return torch.cat([glob] + chans, 1)                                       # (B, dz)


class Normalizer:
    """Channel standardisation by training medians and floored MAD scales, then a smooth clip
    c·tanh(x/c); frozen after fitting (Appendix B)."""

    def __init__(self, dz: int, clip: float = 4.0, floor: float = 1e-3):
        self.median = torch.zeros(dz)
        self.scale = torch.ones(dz)
        self.clip, self.floor, self.fitted = clip, floor, False

    def fit(self, raw: torch.Tensor):
        raw = raw.detach().float().cpu()
        med = raw.median(0).values
        mad = (raw - med).abs().median(0).values * 1.4826
        self.median = med
        self.scale = torch.maximum(mad, torch.full_like(mad, self.floor) + 0.05 * med.abs())
        self.fitted = True

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        x = (raw - self.median.to(raw)) / self.scale.to(raw)
        return self.clip * torch.tanh(x / self.clip)

    def state_dict(self):
        return {"median": self.median, "scale": self.scale, "clip": self.clip, "floor": self.floor, "fitted": self.fitted}

    def load_state_dict(self, d):
        self.median, self.scale, self.clip, self.floor, self.fitted = d["median"], d["scale"], d["clip"], d["floor"], d["fitted"]


# ============================================================================================
# backbone container
# ============================================================================================


class Backbone(nn.Module):
    """Encoder, roles, weight-tied block, readout and reference schedule (θ of (2)) for one bank."""

    def __init__(self, F_in: int, n_classes: int, d: int, dff: int, Kf: int, Kc: int, b_min: float, R0: float,
                 bank: List[str], geom: ActionGeometry, gamma_sl: float, ref_knots: int, support_s: int, eps_g: float,
                 role_hidden: int = 64):
        super().__init__()
        self.F_in, self.n_classes, self.d, self.R0 = F_in, n_classes, d, R0
        self.bank = list(bank)
        self.gamma_sl = gamma_sl
        self.encoder = nn.Sequential(nn.Linear(F_in, d), nn.ReLU(), nn.Linear(d, d))
        self.roles = RoleEncoder(F_in, Kf, Kc, b_min, hidden=role_hidden, support_s=support_s, eps_g=eps_g)
        self.block = PropagationBlock(d, dff)
        self.readout = nn.Linear(d, n_classes)
        self.geom = geom
        self.ref_schedule = SplineSchedule(geom, ref_knots)
        self.sparse_slots = [2 + i for i, k in enumerate(self.bank) if k in SPARSE_KERNELS]
        self.sparse_names = [k for k in self.bank if k in SPARSE_KERNELS]
        self.pg_slot = (2 + self.bank.index("Pg")) if "Pg" in self.bank else None

    def update_sigma(self):
        self.block.update_sigma()

    def encode(self, graph: GraphData) -> torch.Tensor:
        return project_ball_rows(self.encoder(graph.x), self.R0)

    def role_inference(self, graph: GraphData) -> Dict[str, torch.Tensor]:
        deg_in, deg_out = graph.degrees()
        logdeg = torch.stack([torch.log1p(deg_in), torch.log1p(deg_out)], 1)
        return self.roles(graph.x, logdeg)

    def kernel_bank(self, sup: UnionSupport, roles: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.sparse_names:
            return None
        return torch.stack([kernel_values(k, sup, roles, self.roles, self.gamma_sl) for k in self.sparse_names], 0)

    def layer(self, H, H0, node_act, sup, kvals, roles, batch, num_graphs):
        return self.block(H, H0, node_act, sup, kvals, self.sparse_slots, self.pg_slot, roles["Theta_f"], roles["B"], batch, num_graphs)

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

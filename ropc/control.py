"""ROPC observer and bilinear transition model (9), anytime
value / stage heads (Section V-C, Appendix E), health region and support distance, the certified
majorization planner (Section V-D, Lemma 5, Lemma 10, Appendix D), margin-scaled execution (14),
the GRU base policy βₖ, interventional designs (Lemma 12), reliability score and envelope
(Appendix B), and the Controller that executes one decision (Algorithm 1)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ActionGeometry, Normalizer, SplineSchedule

# ============================================================================================
# transition model and observer
# ============================================================================================


class TransitionModel(nn.Module):
    """x⁻_{ℓ+1} = A_x(t) x_ℓ + B(x_ℓ,t) u + E c + b_x(t), B(x,t) = B₀(t) + U diag(Γx) R_u,
    x_ℓ = x⁻_ℓ + K_o(z_ℓ − C x⁻_ℓ), K_o = Cᵀ(CCᵀ + R_o)⁻¹ (9).  Depth dependence is a convex
    combination of endpoint matrices, so ‖A_x(t)‖₂ ≤ γ_A for every t once both endpoints satisfy it.
    constrain_() enforces ‖A_x‖₂ ≤ γ_A, ‖U‖₂,‖Γ‖₂ ≤ 1, max_u ‖R_u u‖_∞ ≤ γ_B, σ_min(C) ≥ ς_min,
    λ_max(R_o) ≤ 1, so that (γ_A+γ_B)‖I − K_oC‖₂ < 1 (Proposition 5)."""

    def __init__(self, dx: int, dz: int, du: int, dc: int, rB: int, gamma_A: float, gamma_B: float,
                 Ro_max: float, sigma_min_C: float, tau: torch.Tensor, sr_coords: List[int]):
        super().__init__()
        if not (0 < dx <= dz) or min(du, dc, rB) < 1:
            raise ValueError("Transition dimensions must be positive and dz must be at least dx")
        if not (0 < Ro_max <= 1 and 0 < sigma_min_C <= 1 and 0 <= gamma_A and 0 <= gamma_B):
            raise ValueError("Invalid observer/transition constraints")
        self.dx, self.dz, self.du, self.dc, self.rB = dx, dz, du, dc, rB
        self.gamma_A, self.gamma_B, self.Ro_max, self.sigma_min_C = gamma_A, gamma_B, Ro_max, sigma_min_C
        self.Aa = nn.Parameter(0.9 * torch.eye(dx))
        self.Ab = nn.Parameter(0.9 * torch.eye(dx))
        self.Ba = nn.Parameter(torch.randn(dx, du) * 0.05)
        self.Bb = nn.Parameter(torch.randn(dx, du) * 0.05)
        self.U = nn.Parameter(torch.randn(dx, rB) * 0.1)
        self.Gam = nn.Parameter(torch.randn(rB, dx) * 0.1)
        self.Ru = nn.Parameter(torch.randn(rB, du) * 0.01)
        self.E = nn.Parameter(torch.zeros(dx, dc))
        self.ba = nn.Parameter(torch.zeros(dx))
        self.bb = nn.Parameter(torch.zeros(dx))
        C0 = torch.zeros(dz, dx)
        C0[:dx, :dx] = torch.eye(dx)
        self.C = nn.Parameter(C0)
        self.logRo = nn.Parameter(torch.full((dz,), math.log(0.1)))    # small initial noise: the first filtered state tracks z
        self.Xinit = nn.Parameter(torch.zeros(dx, dc))
        self.x0b = nn.Parameter(torch.zeros(dx))
        tau = tau.detach().reshape(-1)
        if tau.numel() != du:
            if tau.numel() == 0 or du % tau.numel():
                raise ValueError("tau must describe one role or the entire tangent action")
            tau = tau.repeat(du // tau.numel())
        if not bool(torch.isfinite(tau).all()) or bool((tau <= 0).any()):
            raise ValueError("Tangent action bounds must be finite and positive")
        self.register_buffer("tau", tau.to(self.Ru).clone())
        m0 = torch.ones(du)
        m0[sr_coords] = 0.0
        self.register_buffer("mask0", m0)
        self.constrain_()

    @torch.no_grad()
    def constrain_(self):
        for P in (self.Aa, self.Ab):
            s = float(torch.linalg.matrix_norm(P, ord=2))
            if s > self.gamma_A:
                P.mul_(self.gamma_A / s)
        for P in (self.U, self.Gam):
            s = float(torch.linalg.matrix_norm(P, ord=2))
            if s > 1.0:
                P.mul_(1.0 / s)
        rown = (self.Ru.abs() * self.tau[None, :]).sum(1)                  # max_u ‖R_u u‖_∞ ≤ max_r Σ_j |R_rj| τ_j
        fac = torch.clamp(self.gamma_B / rown.clamp_min(1e-12), max=1.0)
        self.Ru.mul_(fac[:, None])
        Uc, Sc, Vh = torch.linalg.svd(self.C, full_matrices=False)
        Sc = Sc.clamp(self.sigma_min_C, 1.0)
        self.C.copy_(Uc @ torch.diag(Sc) @ Vh)
        self.logRo.clamp_(math.log(1e-3), math.log(self.Ro_max))

    # ---- pieces
    def Ro(self) -> torch.Tensor:
        return torch.exp(self.logRo)

    def Ko(self) -> torch.Tensor:
        C = self.C
        return torch.linalg.solve(C @ C.t() + torch.diag(self.Ro()), C).t()

    def observer_contraction(self) -> float:
        kappa = float(torch.linalg.matrix_norm(torch.eye(self.dx, device=self.C.device, dtype=self.C.dtype) - self.Ko() @ self.C, ord=2))
        return (self.gamma_A + self.gamma_B) * kappa

    def A_of(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1, 1)
        return (1 - t) * self.Aa + t * self.Ab

    def B0_of(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1, 1)
        return (1 - t) * self.Ba + t * self.Bb

    def b_of(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)
        return (1 - t) * self.ba + t * self.bb

    def Bmat(self, x: torch.Tensor, t: torch.Tensor, depth_zero: torch.Tensor) -> torch.Tensor:
        """(B, dx, du): B₀(t) + U diag(Γx) R_u, restart–residual columns masked at ℓ = 0 (Lemma 2)."""
        gx = x @ self.Gam.t()                                                  # (B, rB)
        Bm = self.B0_of(t) + self.U.unsqueeze(0) @ (gx.unsqueeze(2) * self.Ru.unsqueeze(0))
        mask = torch.where(depth_zero.reshape(-1, 1, 1), self.mask0.reshape(1, 1, -1), torch.ones_like(self.mask0).reshape(1, 1, -1))
        return Bm * mask

    def M_of(self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor, depth_zero: Optional[torch.Tensor] = None) -> torch.Tensor:
        """∂x⁻_{next}/∂x = A_x(t) + U diag(R_u u) Γ  (B, dx, dx)."""
        if depth_zero is not None:
            u = u * torch.where(depth_zero[:, None], self.mask0[None, :], torch.ones_like(self.mask0)[None, :])
        ru = u @ self.Ru.t()                                                   # (B, rB)
        return self.A_of(t) + self.U.unsqueeze(0) * ru.unsqueeze(1) @ self.Gam.unsqueeze(0)

    def predict(self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor, c: torch.Tensor, depth_zero: torch.Tensor) -> torch.Tensor:
        Ax = torch.bmm(self.A_of(t), x.unsqueeze(2)).squeeze(2)
        Bu = torch.bmm(self.Bmat(x, t, depth_zero), u.unsqueeze(2)).squeeze(2)
        return Ax + Bu + c @ self.E.t() + self.b_of(t)

    def correct(self, x_prior: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = z - x_prior @ self.C.t()
        return x_prior + e @ self.Ko().t(), e

    def x_init(self, c: torch.Tensor) -> torch.Tensor:
        return c @ self.Xinit.t() + self.x0b

    def response(self, x: torch.Tensor, t: torch.Tensor, depth_zero: torch.Tensor) -> torch.Tensor:
        """Model response C B(x,t) ∈ R^{dz×du} (the object compared with C_H)."""
        return self.C.unsqueeze(0) @ self.Bmat(x, t, depth_zero)

    def norms(self) -> Dict[str, float]:
        with torch.no_grad():
            return {
                "A": max(float(torch.linalg.matrix_norm(self.Aa, 2)), float(torch.linalg.matrix_norm(self.Ab, 2))),
                "B0": max(float(torch.linalg.matrix_norm(self.Ba, 2)), float(torch.linalg.matrix_norm(self.Bb, 2))),
                "U": float(torch.linalg.matrix_norm(self.U, 2)), "Gam": float(torch.linalg.matrix_norm(self.Gam, 2)),
                "Ru": float(torch.linalg.matrix_norm(self.Ru, 2)), "Ru_row": float(self.Ru.norm(dim=1).max()),
                "Gam_row": float(self.Gam.norm(dim=1).max()), "C": float(torch.linalg.matrix_norm(self.C, 2)),
                "C_smin": float(torch.linalg.svdvals(self.C).min()), "Ro_max": float(self.Ro().max()),
                "gamma_o": self.observer_contraction(),
            }


# ============================================================================================
# input-convex heads (value Vω with context, stage head ĝω)
# ============================================================================================


class ConvexHead(nn.Module):
    """Partially input-convex head: convex and smooth in z (softplus, nonnegative hidden-to-hidden
    and output weights); a context network (t, ζ) produces the biases of every z-layer.
    V(z;ctx) = w₃ᵀ softplus(W₂ h₁ + U₂ z + b₂) + u₃ᵀ z + b₃,  h₁ = softplus(W₁ z + b₁)."""

    def __init__(self, dz: int, width: int, ctx_dim: int = 0, ctx_hidden: int = 64):
        super().__init__()
        self.dz, self.width, self.ctx_dim = dz, width, ctx_dim
        self.W1 = nn.Linear(dz, width)
        self.W2 = nn.Parameter(torch.rand(width, width) * 0.1)
        self.U2 = nn.Linear(dz, width, bias=True)
        self.w3 = nn.Parameter(torch.rand(width) * 0.1)
        self.u3 = nn.Parameter(torch.zeros(dz))
        self.b3 = nn.Parameter(torch.zeros(1))
        if ctx_dim > 0:
            self.ctx = nn.Sequential(nn.Linear(ctx_dim, ctx_hidden), nn.Tanh(), nn.Linear(ctx_hidden, 2 * width + 1))
            nn.init.zeros_(self.ctx[-1].weight)
            nn.init.zeros_(self.ctx[-1].bias)
        else:
            self.ctx = None
        self.project_()

    @torch.no_grad()
    def project_(self):
        self.W2.clamp_(min=0.0)
        self.w3.clamp_(min=0.0)

    def forward(self, z: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.ctx is not None:
            cb = self.ctx(ctx)
            b1, b2, b3 = cb[:, :self.width], cb[:, self.width:2 * self.width], cb[:, 2 * self.width:]
        else:
            b1 = b2 = 0.0
            b3 = 0.0
        h1 = F.softplus(self.W1(z) + b1)
        h2 = F.softplus(h1 @ self.W2.t() + self.U2(z) + b2)
        return (h2 @ self.w3 + z @ self.u3 + self.b3 + (b3.squeeze(-1) if torch.is_tensor(b3) else 0.0))

    def grad_z(self, z: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            zz = z.detach().requires_grad_(True)
            v = self.forward(zz, ctx)
            g, = torch.autograd.grad(v.sum(), zz)
        return v.detach(), g.detach()

    @torch.no_grad()
    def hessian_bound(self) -> float:
        W1 = float(torch.linalg.matrix_norm(self.W1.weight, 2))
        W2 = float(torch.linalg.matrix_norm(self.W2, 2))
        U2 = float(torch.linalg.matrix_norm(self.U2.weight, 2))
        w3inf = float(self.w3.abs().max())
        t1 = 0.25 * W1 * W1 * float((self.W2.t() @ self.w3).abs().max())
        t2 = 0.25 * w3inf * (W2 * W1 + U2) ** 2
        return t1 + t2

    @torch.no_grad()
    def grad_bound(self) -> float:
        W1 = float(torch.linalg.matrix_norm(self.W1.weight, 2))
        W2 = float(torch.linalg.matrix_norm(self.W2, 2))
        U2 = float(torch.linalg.matrix_norm(self.U2.weight, 2))
        w3 = float(self.w3.norm())
        return W1 * W2 * w3 + U2 * w3 + float(self.u3.norm())

    @torch.no_grad()
    def certify_(self, L_target: float) -> float:
        """Scale the output weights so that the Hessian bound is at most L_target; returns the bound."""
        self.project_()
        b = self.hessian_bound()
        if b > L_target:
            self.w3.mul_(L_target / b)
            b = self.hessian_bound()
        return b


# ============================================================================================
# health region Ψ_t (Section V-D), support distance m(v), envelope and execution (14)
# ============================================================================================


class HealthRegion:
    """Ψ_t(y) = λ_ψ[‖Σ(t)^{-1/2}(y − µ(t))‖₂ − ρ_c]₊ on the regulated outputs y = Ξz (d_y channels
    selected by the rank correlation of channel contrasts with continuation-loss contrasts)."""

    def __init__(self, dz: int, dy: int, lam_psi: float, n_bins: int = 8, eig_floor: float = 1e-3):
        if not (1 <= dy <= dz) or n_bins < 1 or eig_floor <= 0 or lam_psi < 0:
            raise ValueError("Invalid health-region dimensions, covariance floor or penalty")
        self.dz, self.dy, self.lam_psi, self.n_bins, self.eig_floor = dz, dy, lam_psi, n_bins, eig_floor
        self.sel = torch.arange(dy)
        self.mu = torch.zeros(n_bins, dy)
        self.Sigma = torch.eye(dy).repeat(n_bins, 1, 1)
        self.rho_c = 3.0
        self.centers = (torch.arange(n_bins) + 0.5) / n_bins
        self.fitted = False

    def fit(self, z: torch.Tensor, t: torch.Tensor, channel_score: torch.Tensor, rho_quantile: float = 0.95):
        z, t = z.detach().float().cpu(), t.detach().float().cpu()
        self.sel = channel_score.abs().argsort(descending=True)[: self.dy].sort().values.cpu()
        y = z[:, self.sel]
        for b in range(self.n_bins):
            lo, hi = b / self.n_bins, (b + 1) / self.n_bins
            m = (t >= lo) & (t < hi) if b < self.n_bins - 1 else (t >= lo)
            if int(m.sum()) < 2:
                near = (t - self.centers[b]).abs().argsort()[: max(2, len(t) // self.n_bins)]
                m = torch.zeros_like(t, dtype=torch.bool)
                m[near] = True
            yb = y[m]
            self.mu[b] = yb.mean(0)
            S = torch.cov(yb.t()) if yb.shape[0] > 1 else torch.eye(self.dy)
            ev, U = torch.linalg.eigh(S + 1e-9 * torch.eye(self.dy))
            ev = ev.clamp_min(self.eig_floor * max(1.0, float(ev.max())))
            self.Sigma[b] = U @ torch.diag(ev) @ U.t()
        d = self.distance(y, t)
        self.rho_c = float(torch.quantile(d, rho_quantile)) if d.numel() > 1 else 3.0
        self.fitted = True

    def _interp(self, t: torch.Tensor):
        t = t.reshape(-1).float().cpu().clamp(0, 1)
        if self.n_bins == 1:
            return self.mu.expand(t.numel(), -1), self.Sigma.expand(t.numel(), -1, -1)
        pos = (t * self.n_bins - 0.5).clamp(0, self.n_bins - 1)
        i0 = pos.floor().long().clamp(0, self.n_bins - 2)
        w = (pos - i0.float())
        mu = (1 - w)[:, None] * self.mu[i0] + w[:, None] * self.mu[i0 + 1]
        Sig = (1 - w)[:, None, None] * self.Sigma[i0] + w[:, None, None] * self.Sigma[i0 + 1]
        return mu, Sig

    def inv_sqrt(self, t: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(µ(t), Σ(t)^{-1/2}) on device, batched over t."""
        mu, Sig = self._interp(t)
        ev, U = torch.linalg.eigh(Sig)
        L = U @ torch.diag_embed(ev.clamp_min(1e-12).rsqrt()) @ U.transpose(1, 2)
        return mu.to(device), L.to(device)

    def distance(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mu, L = self.inv_sqrt(t, y.device)
        return torch.bmm(L.to(y), (y - mu.to(y)).unsqueeze(2)).squeeze(2).norm(dim=1)

    def psi(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.lam_psi <= 0 or not self.fitted:
            return torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        y = z[:, self.sel.to(z.device)]
        return self.lam_psi * (self.distance(y, t) - self.rho_c).clamp_min(0)

    def L_psi(self) -> float:
        if self.lam_psi <= 0 or not self.fitted:
            return 0.0
        ev = torch.linalg.eigvalsh(self.Sigma)
        return self.lam_psi * float(ev.clamp_min(1e-12).min().rsqrt())

    def state_dict(self):
        return {"sel": self.sel, "mu": self.mu, "Sigma": self.Sigma, "rho_c": self.rho_c, "fitted": self.fitted, "lam_psi": self.lam_psi}

    def load_state_dict(self, d):
        self.sel, self.mu, self.Sigma, self.rho_c, self.fitted = d["sel"], d["mu"], d["Sigma"], d["rho_c"], d["fitted"]
        self.lam_psi = d.get("lam_psi", self.lam_psi)


class SupportDistance:
    """m(v) = ‖Σ_sup^{-1/2}(u(v) − µ_sup)‖₂ with training-fixed centre and metric over executed
    tangent actions; penalty λ_sup[m − χ_q]²₊ (convex)."""

    def __init__(self, du: int, lam_sup: float, quantile: float = 0.95):
        if du < 1 or lam_sup < 0 or not 0 < quantile <= 1:
            raise ValueError("Invalid support dimension, penalty or quantile")
        self.du, self.lam_sup, self.quantile = du, lam_sup, quantile
        self.mu = torch.zeros(du)
        self.L = torch.eye(du)
        self.Minv_norm = 1.0
        self.chi_q = 10.0
        self.fitted = False

    def fit(self, u: torch.Tensor):
        u = u.detach().float().cpu()
        self.mu = u.mean(0)
        S = torch.cov(u.t()) if u.shape[0] > 1 else torch.eye(self.du)
        ev, U = torch.linalg.eigh(S + 1e-9 * torch.eye(self.du))
        ev = ev.clamp_min(1e-3 * max(1.0, float(ev.max())))
        self.L = U @ torch.diag(ev.rsqrt()) @ U.t()
        self.Minv_norm = float(ev.min().reciprocal())
        d = self.m(u)
        self.chi_q = float(torch.quantile(d, self.quantile)) if d.numel() > 1 else 10.0
        self.fitted = True

    def m(self, u: torch.Tensor) -> torch.Tensor:
        return ((u - self.mu.to(u)) @ self.L.to(u).t()).norm(dim=-1)

    def penalty(self, u: torch.Tensor) -> torch.Tensor:
        return self.lam_sup * (self.m(u) - self.chi_q).clamp_min(0) ** 2

    def grad(self, u: torch.Tensor) -> torch.Tensor:
        mu, L = self.mu.to(u), self.L.to(u)
        diff = u - mu
        mm = (diff @ L.t()).norm(dim=-1, keepdim=True)
        act = (mm - self.chi_q).clamp_min(0)
        return 2.0 * self.lam_sup * act * (diff @ (L.t() @ L)) / mm.clamp_min(1e-9)

    def lipschitz(self) -> float:
        return 4.0 * self.lam_sup * self.Minv_norm

    def state_dict(self):
        return {"mu": self.mu, "L": self.L, "Minv_norm": self.Minv_norm, "chi_q": self.chi_q, "fitted": self.fitted, "lam_sup": self.lam_sup}

    def load_state_dict(self, d):
        self.mu, self.L, self.Minv_norm, self.chi_q, self.fitted = d["mu"], d["L"], d["Minv_norm"], d["chi_q"], d["fitted"]
        self.lam_sup = d.get("lam_sup", self.lam_sup)


@dataclass
class Envelope:
    """Continuous envelope L̄_c(r): λ₁ below 0.75, ramp to λ₂ on [0.75,0.80], λ₂ to 0.925, ramp to λ₃
    on [0.925,0.95], λ₃ above; bins at the reliability quantiles (0.80, 0.95)."""
    slopes: Tuple[float, ...] = (0.0, 0.0, 0.0)
    edges: Tuple[float, ...] = (0.80, 0.95)
    ramps: Tuple[Tuple[float, float], ...] = ((0.75, 0.80), (0.925, 0.95))

    def __post_init__(self):
        if not self.slopes or any(math.isnan(s) or s < 0 for s in self.slopes):
            raise ValueError("Envelope slopes must be nonnegative, with +infinity allowed for unsupported bins")
        if any(b < a for a, b in zip(self.slopes, self.slopes[1:])):
            raise ValueError("Envelope slopes must be nondecreasing")
        if len(self.slopes) == 1:
            self.edges, self.ramps = (), ()
        if len(self.edges) != len(self.slopes) - 1 or len(self.ramps) != len(self.slopes) - 1:
            raise ValueError("Envelope bins, slopes and ramps have incompatible lengths")
        if any(not a < b for a, b in self.ramps):
            raise ValueError("Every envelope ramp must have positive width")

    @property
    def n_bins(self) -> int:
        return len(self.slopes)

    def to_dict(self) -> Dict:
        """Strict-JSON representation; an unsupported slope is explicit, not NaN/Infinity."""
        return {"slopes": [None if math.isinf(s) else float(s) for s in self.slopes],
                "infinite_slopes": [math.isinf(s) for s in self.slopes],
                "edges": list(self.edges), "ramps": [list(ramp) for ramp in self.ramps]}

    @classmethod
    def from_dict(cls, data: Dict) -> "Envelope":
        flags = data.get("infinite_slopes", [False] * len(data["slopes"]))
        if len(flags) != len(data["slopes"]):
            raise ValueError("Envelope infinity mask must match the slope count")
        slopes = tuple(float("inf") if flag or slope is None else float(slope)
                       for slope, flag in zip(data["slopes"], flags))
        return cls(slopes, tuple(data.get("edges", (0.80, 0.95))),
                   tuple(tuple(ramp) for ramp in data.get("ramps", ((0.75, 0.80), (0.925, 0.95)))))

    def bin_of(self, r: torch.Tensor) -> torch.Tensor:
        b = torch.zeros_like(r, dtype=torch.long)
        for e in self.edges:
            b = b + (r >= e).long()
        return b

    def L_bar(self, r: torch.Tensor) -> torch.Tensor:
        lam = [float(s) for s in self.slopes]
        if len(lam) == 1:
            return torch.full_like(r, lam[0])
        out = torch.full_like(r, lam[0])
        for k in range(1, len(lam)):
            a, b = self.ramps[k - 1]
            w = ((r - a) / max(b - a, 1e-9)).clamp(0, 1)
            if math.isinf(lam[k]):
                # An unsupported bin and the part of its ramp with positive
                # weight force fallback; never form 0*inf or inf-inf.
                transition = torch.where(w > 0, torch.full_like(r, float("inf")), torch.full_like(r, lam[k - 1]))
            else:
                transition = (1 - w) * lam[k - 1] + w * lam[k]
            out = torch.where(r >= a, transition, out)
        return out

    def in_ramp(self, r: torch.Tensor) -> torch.Tensor:
        m = torch.zeros_like(r, dtype=torch.bool)
        if len(self.slopes) > 1:
            for a, b in self.ramps:
                m = m | ((r > a) & (r < b))
        return m

    def with_doubled(self, k: int) -> "Envelope":
        s = list(self.slopes)
        s[k] = 2.0 * s[k]
        for j in range(k + 1, len(s)):
            s[j] = max(s[j], s[k])
        return Envelope(tuple(s), self.edges, self.ramps)

    def lipschitz(self) -> float:
        lam = list(self.slopes)
        if any(math.isinf(v) for v in lam):
            return float("inf")
        if len(lam) == 1:
            return 0.0
        return max((lam[k] - lam[k - 1]) / max(self.ramps[k - 1][1] - self.ramps[k - 1][0], 1e-9) for k in range(1, len(lam)))


def margin_scaled_execution(b: torch.Tensor, d: torch.Tensor, delta_hat: torch.Tensor, r: torch.Tensor,
                            env: Envelope, eps_A: float, geom: ActionGeometry):
    """(14): ρ = clip((Δ̂ − L̄_c(r)‖d‖_{1,Σ})/ε_A, 0, 1), A = b + ρ d, certified advantage ρ·net margin."""
    Lc = env.L_bar(r)
    dn = geom.norm_1sigma(b + d, b)
    if eps_A <= 0:
        raise ValueError("The execution ramp must be positive")
    supported = torch.isfinite(Lc) & torch.isfinite(delta_hat) & torch.isfinite(dn) & torch.isfinite(r)
    finite_net = torch.nan_to_num(delta_hat, nan=0.0, posinf=0.0, neginf=0.0) - torch.where(torch.isfinite(Lc), Lc, torch.zeros_like(Lc)) * torch.nan_to_num(dn)
    rho = torch.where(supported, (finite_net / eps_A).clamp(0.0, 1.0), torch.zeros_like(finite_net))
    A = torch.where(supported.reshape(-1, 1, 1), b + rho.reshape(-1, 1, 1) * d, b)
    certificate = rho * torch.where(supported, finite_net, torch.zeros_like(finite_net))
    net = torch.where(supported, finite_net, torch.full_like(finite_net, -float("inf")))
    return A, rho, certificate, Lc, dn, net


class ReliabilityScorer:
    """r_ℓ = F̂(max{s^e/q_e, s^d/q_d}): s^e = d_z⁻¹‖ē‖² (standardised innovation), s^d = d_x⁻¹ m²
    (state support distance); q_e, q_d are tuning 0.95-quantiles; F̂ the tuning CDF of the maximum,
    linearly interpolated between distinct knots with tied values pooled."""

    def __init__(self, dz: int, dx: int):
        self.dz, self.dx = dz, dx
        self.e_scale = torch.ones(dz)
        self.x_mu = torch.zeros(dx)
        self.x_L = torch.eye(dx)
        self.q_e = 1.0
        self.q_d = 1.0
        self.knots = torch.tensor([0.0, 1.0])
        self.cdf = torch.tensor([0.0, 1.0])
        self.fitted = False

    def fit_training(self, e: torch.Tensor, x: torch.Tensor):
        e, x = e.detach().float().cpu(), x.detach().float().cpu()
        if not len(e) or not len(x) or not bool(torch.isfinite(e).all() & torch.isfinite(x).all()):
            raise ValueError("Reliability fitting needs nonempty finite training records")
        self.e_scale = e.std(0, unbiased=False).clamp_min(1e-3)
        self.x_mu = x.mean(0)
        S = torch.cov(x.t()) if x.shape[0] > 1 else torch.eye(self.dx)
        ev, U = torch.linalg.eigh(S + 1e-9 * torch.eye(self.dx))
        ev = ev.clamp_min(1e-3 * max(1.0, float(ev.max())))
        self.x_L = U @ torch.diag(ev.rsqrt()) @ U.t()

    def raw_scores(self, e: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        se = ((e / self.e_scale.to(e)) ** 2).mean(1)
        sd = (((x - self.x_mu.to(x)) @ self.x_L.to(x).t()) ** 2).sum(1) / self.dx
        return se, sd

    def fit_tuning(self, e: torch.Tensor, x: torch.Tensor):
        if not len(e) or not len(x):
            raise ValueError("Reliability tuning needs nonempty records")
        se, sd = self.raw_scores(e.detach().float().cpu(), x.detach().float().cpu())
        self.q_e = max(float(torch.quantile(se, 0.95)), 1e-6)
        self.q_d = max(float(torch.quantile(sd, 0.95)), 1e-6)
        s = torch.maximum(se / self.q_e, sd / self.q_d)
        s, _ = s.sort()
        n = s.numel()
        ranks = (torch.arange(n, dtype=torch.float) + 1.0) / n
        knots, inv = torch.unique(s, return_inverse=True)
        cdf = torch.zeros(knots.numel()).index_reduce_(0, inv, ranks, "amax", include_self=False)  # tied values pooled
        self.knots, self.cdf = knots, cdf
        self.fitted = True

    def score(self, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        se, sd = self.raw_scores(e, x)
        s = torch.maximum(se / self.q_e, sd / self.q_d)
        k, c = self.knots.to(s), self.cdf.to(s)
        if k.numel() < 2:
            return torch.ones_like(s).to(e.device)
        idx = torch.searchsorted(k, s).clamp(1, k.numel() - 1)
        k0, k1, c0, c1 = k[idx - 1], k[idx], c[idx - 1], c[idx]
        w = ((s - k0) / (k1 - k0).clamp_min(1e-12)).clamp(0, 1)
        r = (c0 + w * (c1 - c0)).clamp(0, 1)
        r = torch.where(s <= k[0], torch.full_like(r, float(c[0])), r)
        r = torch.where(s >= k[-1], torch.ones_like(r), r)
        return r.to(e.device)

    def state_dict(self):
        return {k: getattr(self, k) for k in ("e_scale", "x_mu", "x_L", "q_e", "q_d", "knots", "cdf", "fitted")}

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)


# ============================================================================================
# base policy βₖ (GRU) and students
# ============================================================================================


class BasePolicyGRU(nn.Module):
    """GRU of width w_A on (x_ℓ, A_{ℓ−1}, t_ℓ) proposing a feasible role-wise action."""

    def __init__(self, geom: ActionGeometry, dx: int, wA: int, init_raw: Optional[torch.Tensor] = None):
        super().__init__()
        self.geom, self.dx, self.wA = geom, dx, wA
        self.cell = nn.GRUCell(dx + geom.Kc * geom.p + 1, wA)
        self.head = nn.Linear(wA, geom.Kc * geom.p)
        nn.init.zeros_(self.head.weight)
        if init_raw is not None:
            with torch.no_grad():
                self.head.bias.copy_(init_raw.reshape(-1))

    def forward(self, x: torch.Tensor, A_prev: torch.Tensor, eta: torch.Tensor, t: torch.Tensor):
        inp = torch.cat([x, A_prev.flatten(1), t.reshape(-1, 1).expand(x.shape[0], 1)], 1)
        eta2 = self.cell(inp, eta)
        raw = self.head(eta2).reshape(-1, self.geom.Kc, self.geom.p)
        return self.geom.parameterize(raw), eta2


# ============================================================================================
# interventional designs (Lemma 12)
# ============================================================================================


def hadamard(n: int) -> torch.Tensor:
    assert n & (n - 1) == 0 and n > 0
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


class RotatingDesign:
    """Rows of a Hadamard matrix of order N_H = 2^⌈log₂ du⌉ restricted to du columns, handed out
    K_b at a time; a block of N_H/K_b consecutive states completes the design."""

    def __init__(self, du: int, Kb: int):
        if du < 1 or Kb < 1:
            raise ValueError("Hadamard action dimension and pairs per state must be positive")
        self.du, self.Kb = du, Kb
        self.NH = 1 << max(1, math.ceil(math.log2(max(2, du))))
        if self.NH % Kb:
            raise ValueError("Pairs per state must divide the Hadamard order")
        self.H = hadamard(self.NH)[:, :du]
        self.ptr = 0
        self.states_per_block = self.NH // Kb

    def next_rows(self) -> Tuple[torch.Tensor, int]:
        rows = torch.stack([self.H[(self.ptr + k) % self.NH] for k in range(self.Kb)], 0)
        start = self.ptr
        self.ptr = (self.ptr + self.Kb) % self.NH
        return rows, start


def tangent_constraints(geom: ActionGeometry, device, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rows (C_u, d_u) of the polytope (8) in the tangent coordinates of one role block:
    C_u u ≤ d_u, with a = 1/p + T u."""
    p, T = geom.p, geom.T.to(device=device, dtype=dtype)
    rows, rhs = [], []
    for k in range(p):                       # a_k ≥ lo_k  ⇔  −T_k u ≤ 1/p − lo_k
        rows.append(-T[k])
        rhs.append(1.0 / p - float(geom.lo[k]))
    rows.append(T[0])                        # a_s ≤ α_max
    rhs.append(float(geom.alpha_max) - 1.0 / p)
    nT = geom.n_vec.to(device=device, dtype=dtype) @ T           # nᵀa ≤ 1
    rows.append(nT)
    rhs.append(1.0 - float(geom.n_vec.sum()) / p)
    return torch.stack(rows, 0), torch.tensor(rhs, device=device, dtype=dtype)


def slack_radii(geom: ActionGeometry, u_center: torch.Tensor, delta: float, eta_s: float, floor_div: float = 8.0) -> torch.Tensor:
    """δ_j = max{min(δ, η_s min_s slack_{s,j}), δ/8} per tangent coordinate (states stacked in u_center
    (S, du); the minimum over the block's states is taken)."""
    S, du = u_center.shape
    Cu, du_ = tangent_constraints(geom, u_center.device, u_center.dtype)
    Uc = u_center.reshape(S, geom.Kc, geom.M + 1)
    slack = du_[None, None, :] - Uc @ Cu.t()                       # (S, Kc, n_cons)
    coef = Cu.abs().t()                                             # (M+1, n_cons)
    ratio = slack.unsqueeze(2) / coef[None, None, :, :].clamp_min(1e-12)   # (S, Kc, M+1, n_cons)
    ratio = torch.where(coef[None, None, :, :] > 1e-9, ratio, torch.full_like(ratio, 1e9))
    s_j = ratio.amin(dim=3).amin(dim=0).reshape(-1)                 # min over constraints and states: (du,)
    dj = torch.minimum(torch.full_like(s_j, delta), eta_s * s_j).clamp_min(delta / floor_div)
    return dj


def eroded_bounds(geom: ActionGeometry, dj: torch.Tensor):
    """Exact box erosion of the implemented inequalities: C_u u ≤ d_u−|C_u|δ.
    In particular the floor erosion is |nᵀT|δ, not nᵀ|T|δ.
    """
    D = dj.reshape(geom.Kc, geom.M + 1)
    e = geom.T.to(dj).abs() @ D.t()                               # (p, Kc)
    e = e.t()                                                        # (Kc, p)
    lo = geom.lo.to(dj)[None, :] + e
    hi = geom.hi.to(dj)[None, :] - e
    beta = 1.0 - D @ (geom.n_vec.to(dj) @ geom.T.to(dj)).abs()
    return lo, hi, beta


def pooled_response(zp: torch.Tensor, zm: torch.Tensor, xi: torch.Tensor, dj: torch.Tensor) -> torch.Tensor:
    """Ĉ = (1/2N_H) Σ_k (z⁺ − z⁻) ξᵀ D_δ⁻¹ for one completed block (zp, zm: (N_H, dz); xi: (N_H, du))."""
    NH = zp.shape[0]
    return ((zp - zm).t() @ xi) / (2.0 * NH) / dj[None, :]


def excitation_profile(Chat: torch.Tensor) -> torch.Tensor:
    """ε_j = du ‖Ĉ e_j‖² / ‖Ĉ‖_F² (mean one over j) (21)."""
    du = Chat.shape[1]
    col = (Chat ** 2).sum(0)
    tot = col.sum().clamp_min(1e-30)
    return du * col / tot


def ridge_weights(eps: torch.Tensor, lam0: float, tau_exc: float, kappa_max: float) -> torch.Tensor:
    return lam0 * torch.minimum(torch.maximum(torch.ones_like(eps), tau_exc / eps.clamp_min(1e-12)), torch.full_like(eps, kappa_max))


# ============================================================================================
# planner (Section V-D; Appendix D)
# ============================================================================================


@dataclass
class PlannerConfig:
    H: int = 3
    No: int = 3
    Npd: int = 8
    delta_tr: float = 0.025
    beta_w: float = 0.5
    lam_delta: float = 0.02
    lam_v: float = 0.05
    lam_V: float = 0.3
    lam_sup: float = 0.5
    L_V: float = 4.0
    eps_P: float = 1e-4
    hazard: float = 1.0 / 32
    ell0: float = 8.0
    x_max_factor: float = 1.25


def depth_t(ell, ell0: float) -> torch.Tensor:
    ell = torch.as_tensor(ell, dtype=torch.float)
    return ell / (ell + ell0)


class Planner:
    """Certified rollout planner: identified H-step rollout under the base, convex majorant M_o with
    certified remainder ρ_o (Lemma 10), N_pd primal–dual steps, certificate c_o and continuous blend
    ω_o = clip(c_o/ε_P, 0, 1) (13); never increases the learned objective (Lemma 5)."""

    def __init__(self, cfg: PlannerConfig, geom: ActionGeometry, model: TransitionModel, value: ConvexHead,
                 stage: ConvexHead, health: HealthRegion, support: SupportDistance, x_max: float):
        self.cfg, self.geom, self.model, self.value, self.stage, self.health, self.support = cfg, geom, model, value, stage, health, support
        self.x_max = x_max
        if cfg.H < 1 or cfg.No < 1 or cfg.Npd < 1 or cfg.eps_P <= 0 or cfg.delta_tr < 0:
            raise ValueError("Planner budgets/ramp must be positive and trust radius nonnegative")
        if min(cfg.lam_delta, cfg.lam_v, cfg.lam_V, cfg.lam_sup, cfg.L_V) < 0 or not 0 <= cfg.beta_w <= 1 or not 0 < cfg.hazard < 1:
            raise ValueError("Planner weights must be nonnegative, damping in [0,1], and hazard in (0,1)")
        H, Kc, p = cfg.H, geom.Kc, geom.p
        self.dim = H * Kc * p
        dev = model.C.device
        TT = geom.T.to(model.C).t().contiguous()                                         # (M+1, p)
        self.Tfull = torch.kron(torch.eye(H * Kc, device=dev, dtype=model.C.dtype), TT)   # (H·du, dim)
        self.last_info: Dict = {}

    # ---------------- rollout with Jacobians
    def rollout(self, x0: torch.Tensor, Useq: torch.Tensor, t_seq: torch.Tensor, c: torch.Tensor, depth_zero0: torch.Tensor,
                with_jac: bool = True):
        """x0 (B,dx); Useq (B,H,du); t_seq (H,); returns x̂ (B,H,dx), ẑ (B,H,dz), Dx (B,H,dx,H·du)."""
        m, H = self.model, self.cfg.H
        Bn, du = Useq.shape[0], Useq.shape[2]
        xs, Dxs = [], []
        x = x0
        Dx = torch.zeros(Bn, m.dx, H * du, device=x0.device, dtype=x0.dtype)
        dz0 = depth_zero0
        for j in range(H):
            t = t_seq[j].expand(Bn)
            u = Useq[:, j]
            dzflag = dz0 if j == 0 else torch.zeros_like(dz0)
            Bm = m.Bmat(x, t, dzflag)
            x_next = torch.bmm(m.A_of(t), x.unsqueeze(2)).squeeze(2) + torch.bmm(Bm, u.unsqueeze(2)).squeeze(2) + c @ m.E.t() + m.b_of(t)
            if with_jac:
                Mj = m.M_of(x, t, u, dzflag)
                Dx_next = torch.bmm(Mj, Dx)
                Dx_next[:, :, j * du:(j + 1) * du] = Dx_next[:, :, j * du:(j + 1) * du] + Bm
                Dx = Dx_next
                Dxs.append(Dx)
            xs.append(x_next)
            x = x_next
        X = torch.stack(xs, 1)
        Z = X @ m.C.t()
        D = torch.stack(Dxs, 1) if with_jac else None
        return X, Z, D

    def curvature_constants(self, X: torch.Tensor, Useq: torch.Tensor, t_seq: torch.Tensor,
                            *, x0: torch.Tensor, context: torch.Tensor) -> List[float]:
        """β_j such that ‖ẑ_j(V) − z̄_j − G_j ΔV‖ ≤ (β_j/2)‖ΔV‖² on the trust region (recursion
        on the bilinear rollout; β₁ = 0 because the first transition is affine).

        Bounds cover every feasible action sequence, hence the whole trust box.
        They start at the actual corrected state and propagate context/bias and
        simplex action-norm bounds. Training maxima and nominal rollouts are
        diagnostics only and are never used as domain certificates.
        """
        m, H = self.model, self.cfg.H
        nrm = m.norms()
        C2 = nrm["C"]
        ru_bound = float((m.Ru.detach().abs() * m.tau).sum(1).max())
        a_coef = nrm["A"] + nrm["U"] * nrm["Gam"] * ru_bound
        c_coef = nrm["U"] * min(nrm["Gam"] * nrm["Ru_row"], nrm["Gam_row"] * nrm["Ru"])
        xmax = float(x0.detach().norm(dim=-1).max())
        u_max = math.sqrt(self.geom.Kc * (1.0 - 1.0 / self.geom.p))
        betas, state_bounds = [], []
        L = 0.0
        K = 0.0
        for j in range(H):
            tB = float(t_seq[j])
            B0n = float(torch.linalg.matrix_norm((1 - tB) * m.Ba + tB * m.Bb, 2))
            Bn_bound = B0n + nrm["U"] * nrm["Gam_row"] * xmax * nrm["Ru"]
            K = a_coef * K + 2.0 * c_coef * L
            L = a_coef * L + Bn_bound
            betas.append(C2 * K)
            forcing = context @ m.E.t() + (1 - tB) * m.ba + tB * m.bb
            xmax = a_coef * xmax + B0n * u_max + float(forcing.detach().norm(dim=-1).max())
            state_bounds.append(xmax)
        if not all(math.isfinite(v) for v in betas + state_bounds):
            raise RuntimeError("Nonfinite rollout bounds; check transition model parameters")
        self.last_info["state_norm_bounds"] = state_bounds
        return betas

    # ---------------- objective pieces
    def _weights(self):
        h, H, lamV = self.cfg.hazard, self.cfg.H, self.cfg.lam_V
        w = [lamV * h * (1 - h) ** j for j in range(1, H)] + [lamV * (1 - h) ** H]
        return w

    def action_terms(self, V: torch.Tensor, b_seq: torch.Tensor, A_prev: torch.Tensor) -> torch.Tensor:
        """λ_Δ‖v_j − v_{j−1}‖² + λ_v‖v_j − b_{ℓ+j}‖² + λ_sup[m(v_j) − χ_q]²₊ summed over stages (B,)."""
        cfg = self.cfg
        Vm1 = torch.cat([A_prev.unsqueeze(1), V[:, :-1]], 1)
        sm = cfg.lam_delta * ((V - Vm1) ** 2).sum(dim=(1, 2, 3)) + cfg.lam_v * ((V - b_seq) ** 2).sum(dim=(1, 2, 3))
        U = self.geom.to_tangent(V)                                                        # (B,H,du)
        sup = self.support.penalty(U).sum(1) if self.support.fitted and cfg.lam_sup > 0 else 0.0
        return sm + sup

    def action_grad(self, V: torch.Tensor, b_seq: torch.Tensor, A_prev: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        Vm1 = torch.cat([A_prev.unsqueeze(1), V[:, :-1]], 1)
        diff = V - Vm1
        diff_next = torch.cat([diff[:, 1:], torch.zeros_like(diff[:, :1])], 1)
        g = 2 * cfg.lam_delta * (diff - diff_next) + 2 * cfg.lam_v * (V - b_seq)
        if self.support.fitted and cfg.lam_sup > 0:
            U = self.geom.to_tangent(V)
            gu = self.support.grad(U)                                                       # (B,H,du)
            g = g + (gu.reshape(*gu.shape[:-1], self.geom.Kc, self.geom.M + 1) @ self.geom.T.to(gu).t())
        return g

    def objective(self, V: torch.Tensor, x: torch.Tensor, t_seq: torch.Tensor, c: torch.Tensor, depth_zero: torch.Tensor,
                  b_seq: torch.Tensor, A_prev: torch.Tensor, ctxH: torch.Tensor) -> torch.Tensor:
        """Ĵ_ℓ(V) of (11), (B,)."""
        H = self.cfg.H
        U = self.geom.to_tangent(V)
        _, Z, _ = self.rollout(x, U, t_seq, c, depth_zero, with_jac=False)
        w = self._weights()
        J = torch.zeros(V.shape[0], device=V.device, dtype=V.dtype)
        for j in range(1, H):
            J = J + w[j - 1] * self.stage(Z[:, j - 1])
        J = J + w[H - 1] * self.value(Z[:, H - 1], ctxH)
        for j in range(1, H + 1):
            J = J + self.health.psi(Z[:, j - 1], t_seq[j].expand(V.shape[0]))
        return J + self.action_terms(V, b_seq, A_prev)

    # ---------------- one planning call
    def plan(self, x: torch.Tensor, ell: int, c: torch.Tensor, b_seq: torch.Tensor, V_prev: Optional[torch.Tensor],
             A_prev: torch.Tensor, ctxH: torch.Tensor, unit_test_points: int = 0,
             role_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """x: corrected state (B,dx); b_seq: nominal base proposals (B,H,Kc,p); V_prev: previous plan or None.
        Returns V* (B,H,Kc,p) and certificates/diagnostics."""
        cfg, geom, m = self.cfg, self.geom, self.model
        Bn, H, Kc, p = x.shape[0], cfg.H, geom.Kc, geom.p
        dev, dt = x.device, x.dtype
        self.last_info = {}
        t_seq = torch.stack([depth_t(ell + j, cfg.ell0) for j in range(H + 1)]).to(x)
        depth_zero = torch.full((Bn,), ell == 0, dtype=torch.bool, device=dev)
        # trust region and warm start
        lo = torch.maximum(geom.lo.to(b_seq).reshape(1, 1, 1, p).expand_as(b_seq), b_seq - cfg.delta_tr)
        hi = torch.minimum(geom.hi.to(b_seq).reshape(1, 1, 1, p).expand_as(b_seq), b_seq + cfg.delta_tr)
        if role_mask is not None:
            active = role_mask.to(device=dev, dtype=torch.bool)[:, None, :, None]
            lo, hi = torch.where(active, lo, b_seq), torch.where(active, hi, b_seq)
        if V_prev is None:
            Vw = b_seq.clone()
        else:
            shifted = torch.cat([V_prev[:, 1:], V_prev[:, -1:]], 1)
            Vw = (1 - cfg.beta_w) * b_seq + cfg.beta_w * shifted
        Vbar, nfb = geom.project(Vw, lo, hi)
        J0 = self.objective(Vbar, x, t_seq, c, depth_zero, b_seq, A_prev, ctxH)
        w = self._weights()
        Tfull = self.Tfull.to(x)
        Delta_tr = float((hi - lo).flatten(1).norm(dim=1).max())
        Gf = max(self.stage.grad_bound(), self.value.grad_bound())
        if any(bool((head.W2 < 0).any() | (head.w3 < 0).any()) for head in (self.stage, self.value)):
            raise ValueError("Planning requires projected nonnegative convex-head weights")
        LV = max(cfg.L_V, self.stage.hessian_bound(), self.value.hessian_bound())
        Lpsi = self.health.L_psi()
        cert, omegas, steps_full = [], [], 0
        Lsup = self.support.lipschitz() if (self.support.fitted and cfg.lam_sup > 0) else 0.0
        for o in range(cfg.No):
            # linearise the rollout at the incumbent
            U = geom.to_tangent(Vbar)
            X, Z, D = self.rollout(x, U, t_seq, c, depth_zero, with_jac=True)
            G = torch.matmul(m.C.unsqueeze(0).unsqueeze(0), D) @ Tfull                       # (B,H,dz,dim)
            betas = self.curvature_constants(X, U, t_seq, x0=x, context=c)
            rho_o = sum(w[j] * (Gf * betas[j] + LV * betas[j] ** 2 * Delta_tr ** 2 / 4.0) for j in range(H)) + Lpsi * sum(betas)
            # head values and gradients at z̄_j
            fvals, grads = [], []
            for j in range(H):
                if j < H - 1:
                    fv, gz = self.stage.grad_z(Z[:, j])
                else:
                    fv, gz = self.value.grad_z(Z[:, j], ctxH)
                fvals.append(fv)
                grads.append(gz)
            # Displacement coordinates: K_j ΔV + c_j, c_j = Σ^{-1/2}(Ξz̄_j−µ_j).
            Vflat = Vbar.reshape(Bn, -1)
            Ks, cs, mus, Ls = [], [], [], []
            use_hinge = self.health.lam_psi > 0 and self.health.fitted
            if use_hinge:
                sel = self.health.sel.to(dev)
                for j in range(H):
                    mu_j, L_j = self.health.inv_sqrt(t_seq[j + 1].expand(Bn), dev)
                    mu_j, L_j = mu_j.to(x), L_j.to(x)
                    Gsel = G[:, j][:, sel, :]                                               # (B,dy,dim)
                    Kj = torch.bmm(L_j, Gsel)
                    cj = torch.bmm(L_j, (Z[:, j][:, sel] - mu_j).unsqueeze(2)).squeeze(2)
                    Ks.append(Kj)
                    cs.append(cj)
                K = torch.cat(Ks, 1)                                                          # (B, H·dy, dim)
                cvec = torch.cat(cs, 1)
                Knorm = torch.linalg.matrix_norm(K, ord=2)                                   # (B,)
            else:
                K = None
                Knorm = torch.zeros(Bn, device=dev, dtype=dt)
            # quadratic data of the smooth part: Q ΔV + q
            Q = rho_o * torch.eye(self.dim, device=dev, dtype=dt).unsqueeze(0).expand(Bn, -1, -1).clone()
            q = torch.zeros(Bn, self.dim, device=dev, dtype=dt)
            for j in range(H):
                Gj = G[:, j]
                Q = Q + w[j] * LV * torch.bmm(Gj.transpose(1, 2), Gj)
                q = q + w[j] * torch.bmm(Gj.transpose(1, 2), grads[j].unsqueeze(2)).squeeze(2)
            Gnorm2 = sum(w[j] * float(torch.linalg.matrix_norm(G[:, j], ord=2).max()) ** 2 for j in range(H))
            Lf = LV * Gnorm2 + rho_o + 2 * cfg.lam_v + 8 * cfg.lam_delta + Lsup
            sigma = torch.where(Knorm > 1e-9, 1.0 / Knorm.clamp_min(1e-9), torch.ones_like(Knorm))
            tau = 0.9 / (Lf / 2.0 + sigma * Knorm ** 2).clamp_min(1e-12)                       # (B,)

            def M_o(Vq: torch.Tensor) -> torch.Tensor:
                dV = Vq.reshape(Bn, -1) - Vflat
                val = torch.zeros(Bn, device=dev, dtype=dt)
                for j in range(H):
                    val = val + w[j] * fvals[j]
                val = val + (q * dV).sum(1) + 0.5 * torch.bmm(dV.unsqueeze(1), torch.bmm(Q, dV.unsqueeze(2))).reshape(Bn)
                if use_hinge:
                    wv = torch.bmm(K, dV.unsqueeze(2)).squeeze(2) + cvec
                    wv = wv.reshape(Bn, H, -1).norm(dim=2)
                    val = val + self.health.lam_psi * (wv - self.health.rho_c).clamp_min(0).sum(1)
                return val + self.action_terms(Vq, b_seq, A_prev)

            def smooth_grad(Vq: torch.Tensor) -> torch.Tensor:
                dV = Vq.reshape(Bn, -1) - Vflat
                g = q + torch.bmm(Q, dV.unsqueeze(2)).squeeze(2)
                return g.reshape_as(Vq) + self.action_grad(Vq, b_seq, A_prev)

            # primal–dual (Chambolle–Pock / Condat–Vũ) on the majorant over V_ℓ
            Vk = Vbar.clone()
            y = torch.zeros(Bn, H * self.health.dy, device=dev, dtype=dt) if use_hinge else None
            for _ in range(cfg.Npd):
                g = smooth_grad(Vk)
                if use_hinge:
                    g = g + torch.bmm(K.transpose(1, 2), y.unsqueeze(2)).squeeze(2).reshape_as(Vk)
                Vn, nf = geom.project(Vk - tau.reshape(-1, 1, 1, 1) * g, lo, hi)
                nfb += nf
                if use_hinge:
                    yt = y + sigma.reshape(-1, 1) * (torch.bmm(K, (2 * Vn - Vk).reshape(Bn, -1, 1).sub(Vflat.unsqueeze(2))).squeeze(2) + cvec)
                    yb = yt.reshape(Bn, H, -1)
                    nrm = yb.norm(dim=2, keepdim=True)
                    yb = yb * (1 - (sigma.reshape(-1, 1, 1) * self.health.rho_c) / nrm.clamp_min(1e-12)).clamp_min(0)
                    nrm2 = yb.norm(dim=2, keepdim=True)
                    yb = yb * torch.clamp(self.health.lam_psi / nrm2.clamp_min(1e-12), max=1.0)
                    y = yb.reshape(Bn, -1)
                Vk = Vn
            if unit_test_points > 0:
                tangent_error = float((M_o(Vbar) - self.objective(Vbar, x, t_seq, c, depth_zero, b_seq, A_prev, ctxH)).abs().max())
                self.last_info["majorant_tangent_error"] = max(self.last_info.get("majorant_tangent_error", 0.0), tangent_error)
            Vtil = Vk
            c_o = M_o(Vbar) - M_o(Vtil)
            omega = (c_o / cfg.eps_P).clamp(0.0, 1.0)
            Vbar = Vbar + omega.reshape(-1, 1, 1, 1) * (Vtil - Vbar)
            cert.append(c_o.detach())
            omegas.append(omega.detach())
            if unit_test_points > 0:
                # released unit test: M_o ≥ Ĵ_ℓ at random trust-region points
                worst = float("inf")
                for _ in range(unit_test_points):
                    Vr, _ = geom.project(lo + torch.rand_like(lo) * (hi - lo), lo, hi)
                    slack = (M_o(Vr) - self.objective(Vr, x, t_seq, c, depth_zero, b_seq, A_prev, ctxH)).min()
                    worst = min(worst, float(slack))
                self.last_info["majorant_min_slack"] = min(self.last_info.get("majorant_min_slack", float("inf")), worst)
        J1 = self.objective(Vbar, x, t_seq, c, depth_zero, b_seq, A_prev, ctxH)
        # stationarity diagnostic: projected-gradient residual of Ĵ at the returned plan
        with torch.enable_grad():
            Vg = Vbar.detach().clone().requires_grad_(True)
            Jg = self.objective(Vg, x, t_seq, c, depth_zero, b_seq, A_prev, ctxH).sum()
            gJ, = torch.autograd.grad(Jg, Vg)
        Vp, _ = geom.project(Vbar - gJ, lo, hi)
        resid = (Vbar - Vp).reshape(Bn, -1).norm(dim=1)
        info = {"J_warm": J0.detach(), "J_plan": J1.detach(), "cert": torch.stack(cert, 1), "omega": torch.stack(omegas, 1),
                "rho_o": float(rho_o), "betas": betas, "head_smoothness_bound": LV,
                "state_norm_bounds": self.last_info["state_norm_bounds"],
                "pg_residual": resid.detach(), "fallbacks": nfb, "t_seq": t_seq}
        self.last_info.update({k: v for k, v in info.items() if k != "t_seq"})
        return Vbar.detach(), info

    def lookahead(self, x: torch.Tensor, ell: int, c: torch.Tensor, a: torch.Tensor, z_now: torch.Tensor, ctx1: torch.Tensor) -> torch.Tensor:
        """Q̂_ℓ(a) = h ĝ(z_ℓ) + (1−h) V(C x̂⁻_{ℓ+1}(a); t_{ℓ+1}, ζ̄_{ℓ+1}), convex in a."""
        cfg, m = self.cfg, self.model
        Bn = x.shape[0]
        t = depth_t(ell, cfg.ell0).to(x.device).expand(Bn)
        dz0 = torch.full((Bn,), ell == 0, dtype=torch.bool, device=x.device)
        xn = m.predict(x, t, self.geom.to_tangent(a), c, dz0)
        zn = xn @ m.C.t()
        return cfg.hazard * self.stage(z_now) + (1 - cfg.hazard) * self.value(zn, ctx1)


# ============================================================================================
# controller (Algorithm 1): sense → observe → propose → plan → execute → update memory
# ============================================================================================


class Controller:
    """Deployment policies: 'ropc' (planner over the base, margin-scaled execution), 'base' (βₖ alone),
    'amortized_raw' (β_{K+1} executed directly), 'amortized_gated' (ROPC-A: β_{K+1} through (14) anchored at
    the reference with the round-1 value), 'student_tw' (a student of the plan in ROPC's own gate),
    'schedule' (open loop: a schedule in t), 'replay' (compiled trace, Corollary 1)."""

    def __init__(self, mode: str, geom: ActionGeometry, cfg: PlannerConfig, model: Optional[TransitionModel],
                 value: Optional[ConvexHead], stage: Optional[ConvexHead], health: Optional[HealthRegion],
                 support: Optional[SupportDistance], envelope: Envelope, eps_A: float, scorer: Optional[ReliabilityScorer],
                 normalizer: Optional[Normalizer], ref_schedule: SplineSchedule, base: Optional[BasePolicyGRU] = None,
                 student: Optional[BasePolicyGRU] = None, schedule: Optional[nn.Module] = None, trace: Optional[torch.Tensor] = None,
                 wA: int = 48, x_max: float = 10.0, gate_value: Optional[ConvexHead] = None, planner_over_reference: bool = False):
        self.mode, self.geom, self.cfg = mode, geom, cfg
        self.model, self.value, self.stage, self.health, self.support = model, value, stage, health, support
        self.envelope, self.eps_A, self.scorer, self.normalizer = envelope, eps_A, scorer, normalizer
        self.ref_schedule, self.base, self.student, self.schedule, self.trace = ref_schedule, base, student, schedule, trace
        self.wA = wA
        self.role_mask: Optional[torch.Tensor] = None
        self.gate_value = gate_value if gate_value is not None else value
        self.planner_over_reference = planner_over_reference
        self.planner = None
        if mode in ("ropc",) and model is not None:
            self.planner = Planner(cfg, geom, model, value, stage, health, support, x_max)
        self.gate_planner = None
        if mode in ("amortized_gated", "student_tw") and model is not None:
            self.gate_planner = Planner(cfg, geom, model, self.gate_value, stage, health, support, x_max)
        self.needs_sensing = mode not in ("schedule", "replay")
        self.reset(1, None)

    # ---- memory
    def reset(self, Bn: int, c: Optional[torch.Tensor]):
        dev = self.ref_schedule.raw.device
        self.Bn = Bn
        self.c = c
        self.x_prior = self.model.x_init(c) if (self.model is not None and c is not None) else None
        self.eta = torch.zeros(Bn, self.wA, device=dev, dtype=self.ref_schedule.raw.dtype)
        self.eta_student = torch.zeros_like(self.eta)
        t0 = depth_t(0, self.cfg.ell0).to(self.ref_schedule.raw)
        self.A_prev = self.ref_schedule(t0).expand(Bn, -1, -1).detach()
        self.V_prev = None
        self.records: List[Dict] = []

    def _ctx(self, t: torch.Tensor, x_prior: torch.Tensor, eta: torch.Tensor, A_prev: torch.Tensor) -> torch.Tensor:
        return torch.cat([t.reshape(-1, 1).expand(x_prior.shape[0], 1), x_prior, eta, A_prev.flatten(1)], 1)

    def propose(self, x: torch.Tensor, t: torch.Tensor, A_prev: torch.Tensor, eta: torch.Tensor, policy: Optional[BasePolicyGRU]):
        if policy is None:
            b = self.ref_schedule(t).expand(x.shape[0], -1, -1)
            return b, eta
        return policy(x, A_prev, eta, t)

    def nominal_rollout(self, x: torch.Tensor, ell: int, c: torch.Tensor, policy: Optional[BasePolicyGRU]):
        """Base proposals b_{ℓ:ℓ+H−1} along the identified rollout and the head contexts ζ̄_{ℓ+1}, ζ̄_{ℓ+H}."""
        cfg, m = self.cfg, self.model
        Bn = x.shape[0]
        dev = x.device
        A_prev, eta = self.A_prev, self.eta
        xs, bs = x, []
        ctx1 = None
        x_prior = self.x_prior
        for j in range(cfg.H):
            t = depth_t(ell + j, cfg.ell0).to(dev)
            b, eta_n = self.propose(xs, t, A_prev, eta, policy)
            b = b.detach()
            bs.append(b)
            dz0 = torch.full((Bn,), (ell + j) == 0, dtype=torch.bool, device=dev)
            x_prior = m.predict(xs, t.expand(Bn), self.geom.to_tangent(b), c, dz0)
            if j == 0:
                eta_next = eta_n
                ctx1 = self._ctx(depth_t(ell + 1, cfg.ell0).to(dev), x_prior, eta_n, b)
            xs = x_prior                                                  # zero innovation along the model rollout
            A_prev, eta = b, eta_n
        ctxH = self._ctx(depth_t(ell + cfg.H, cfg.ell0).to(dev), x_prior, eta, A_prev)
        return torch.stack(bs, 1), ctx1, ctxH, eta_next

    @torch.no_grad()
    def step(self, ell: int, z_raw: Optional[torch.Tensor], c: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """One decision.  z_raw: raw observation (B,dz) or None for sensing-free modes.  Returns the
        executed role-wise action A (B,Kc,p) and a record; call commit(A) afterwards."""
        cfg, geom, dev = self.cfg, self.geom, self.ref_schedule.raw.device
        t = depth_t(ell, cfg.ell0).to(self.ref_schedule.raw)
        rec: Dict = {"ell": ell}
        if self.mode == "schedule":
            A = self.schedule(t).expand(self.Bn, -1, -1) if self.schedule is not None else self.ref_schedule(t).expand(self.Bn, -1, -1)
            return A, rec
        if self.mode == "replay":
            if self.trace is None or not 0 <= ell < self.trace.shape[0]:
                raise ValueError("Replay requires a recorded action for every requested layer")
            action = self.trace[ell].to(self.ref_schedule.raw)
            if action.ndim == 2:
                action = action.unsqueeze(0).expand(self.Bn, -1, -1)
            if action.shape != (self.Bn, geom.Kc, geom.p):
                raise ValueError("Replay trace graph count/action dimensions do not match this batch")
            return action, rec
        z = self.normalizer(z_raw) if self.normalizer is not None else z_raw
        x, e = self.model.correct(self.x_prior, z)
        rec.update({"z": z, "x": x, "e": e, "x_prior": self.x_prior.clone(), "A_prev": self.A_prev.clone()})
        # A decision carries its own future memory. Offline branch evaluations
        # may execute other steps before this decision is committed.
        rec["eta_student_next"] = self.eta_student.clone()
        Bn = self.Bn
        if self.mode == "base":
            b, eta_n = self.propose(x, t, self.A_prev, self.eta, self.base)
            self._pending = (eta_n, self.eta_student, None)
            rec["b"] = b
            rec["eta_next"] = eta_n
            return b.detach(), rec
        if self.mode == "amortized_raw":
            A, eta_n = self.propose(x, t, self.A_prev, self.eta, self.student)
            self._pending = (eta_n, self.eta_student, None)
            rec["b"] = A
            rec["eta_next"] = eta_n
            return A.detach(), rec
        if self.mode == "ropc":
            policy = None if self.planner_over_reference else self.base
            b_seq, ctx1, ctxH, eta_n = self.nominal_rollout(x, ell, c, policy)
            Vstar, info = self.planner.plan(x, ell, c, b_seq, self.V_prev, self.A_prev, ctxH, role_mask=self.role_mask)
            b = b_seq[:, 0]
            d = Vstar[:, 0] - b
            q_b = self.planner.lookahead(x, ell, c, b, z, ctx1)
            q_v = self.planner.lookahead(x, ell, c, Vstar[:, 0], z, ctx1)
            delta_hat = q_b - q_v
            r = self.scorer.score(e, x) if (self.scorer is not None and self.scorer.fitted) else torch.zeros(Bn, device=dev)
            A, rho, dc, Lc, dn, net = margin_scaled_execution(b, d, delta_hat, r, self.envelope, self.eps_A, geom)
            self._pending = (eta_n, self.eta_student, Vstar)
            rec.update({"b": b, "d": d, "delta_hat": delta_hat, "r": r, "rho": rho, "delta_c": dc, "Lc": Lc, "dnorm": dn, "net": net,
                        "cert": info["cert"], "omega": info["omega"], "J_warm": info["J_warm"], "J_plan": info["J_plan"],
                        "pg_residual": info["pg_residual"], "rho_o": info["rho_o"], "q_b": q_b, "q_v": q_v, "ctx1": ctx1,
                        "bin": self.envelope.bin_of(r), "ramp": self.envelope.in_ramp(r), "eta_next": eta_n, "Vstar": Vstar,
                        "health_dist": self.health.distance(z[:, self.health.sel.to(dev)], t.expand(Bn)) if self.health.fitted else torch.zeros(Bn, device=dev)})
            return A.detach(), rec
        if self.mode in ("amortized_gated", "student_tw"):
            if self.mode == "amortized_gated":
                b = self.ref_schedule(t).expand(Bn, -1, -1)
                v, eta_s = self.propose(x, t, self.A_prev, self.eta_student, self.student)
                eta_n = self.eta
                x_prior1 = self.model.predict(x, t.expand(Bn), geom.to_tangent(b), c, torch.full((Bn,), ell == 0, dtype=torch.bool, device=dev))
                ctx1 = self._ctx(depth_t(ell + 1, cfg.ell0).to(dev), x_prior1, torch.zeros(Bn, self.wA, device=dev, dtype=x.dtype), b)
            else:
                b, eta_n = self.propose(x, t, self.A_prev, self.eta, self.base)
                v, eta_s = self.propose(x, t, self.A_prev, self.eta_student, self.student)
                x_prior1 = self.model.predict(x, t.expand(Bn), geom.to_tangent(b), c, torch.full((Bn,), ell == 0, dtype=torch.bool, device=dev))
                ctx1 = self._ctx(depth_t(ell + 1, cfg.ell0).to(dev), x_prior1, eta_n, b)
            if self.role_mask is not None:
                v = torch.where(self.role_mask.to(device=dev, dtype=torch.bool)[:, :, None], v, b)
            d = v - b
            q_b = self.gate_planner.lookahead(x, ell, c, b, z, ctx1)
            q_v = self.gate_planner.lookahead(x, ell, c, v, z, ctx1)
            delta_hat = q_b - q_v
            r = self.scorer.score(e, x) if (self.scorer is not None and self.scorer.fitted) else torch.zeros(Bn, device=dev)
            A, rho, dc, Lc, dn, net = margin_scaled_execution(b, d, delta_hat, r, self.envelope, self.eps_A, geom)
            self._pending = (eta_n, eta_s, None)
            rec.update({"b": b, "d": d, "delta_hat": delta_hat, "r": r, "rho": rho, "delta_c": dc, "Lc": Lc, "dnorm": dn, "net": net,
                        "q_b": q_b, "q_v": q_v, "ctx1": ctx1, "bin": self.envelope.bin_of(r), "ramp": self.envelope.in_ramp(r), "eta_next": eta_n, "eta_student_next": eta_s,
                        "health_dist": self.health.distance(z[:, self.health.sel.to(dev)], t.expand(Bn)) if self.health.fitted else torch.zeros(Bn, device=dev)})
            return A.detach(), rec
        raise ValueError(self.mode)

    @torch.no_grad()
    def commit(self, ell: int, A: torch.Tensor, rec: Dict, c: Optional[torch.Tensor], record: bool = False):
        """Update the memory after executing A at layer ℓ."""
        if self.mode in ("schedule", "replay"):
            self.A_prev = A.detach()
            if record:
                self.records.append({"ell": ell, "A": A.detach().cpu()})
            return
        dev = A.device
        t = depth_t(ell, self.cfg.ell0).to(dev).expand(A.shape[0])
        x = rec["x"]
        dz0 = torch.full((A.shape[0],), ell == 0, dtype=torch.bool, device=dev)
        self.x_prior = self.model.predict(x, t, self.geom.to_tangent(A), c, dz0)
        if "eta_next" not in rec or "eta_student_next" not in rec:
            raise ValueError("A commit needs the memory stored in its own decision record")
        self.eta, self.eta_student, self.V_prev = rec["eta_next"].detach(), rec["eta_student_next"].detach(), rec.get("Vstar")
        self.A_prev = A.detach()
        if record:
            self.records.append({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in rec.items()} | {"A": A.detach().cpu()})

    def lookahead_from_record(self, ell: int, rec: Dict, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Q̂_ℓ(a) for an arbitrary action at the recorded decision (planner or gate value)."""
        pl = self.planner if self.planner is not None else self.gate_planner
        return pl.lookahead(rec["x"], ell, c, a, rec["z"], rec["ctx1"])

    # ---- authority of the executed action relative to the reference
    def authority(self, ell: int, A: torch.Tensor) -> torch.Tensor:
        t = depth_t(ell, self.cfg.ell0).to(A.device)
        return self.geom.norm_1sigma(A, self.ref_schedule(t).expand_as(A))

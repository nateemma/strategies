"""Per-fold training loop (T024).

FR-015 one policy per fold, trained on that fold's training range only
FR-017 early stopping on the TRAINING objective only; no validation split from a test fold
R7     truncated BPTT with a 96-bar (one-day) burn-in that contributes state but no gradient
R9     seeded via MLX_SEED; byte-identical reproduction is required (SC-008)
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from . import losses as L
from . import sim
from .policy import ExposurePolicy, PolicyConfig

BURN_IN = 96          # one day at 15m; contributes hidden state, not gradient


@dataclass(frozen=True)
class TrainConfig:
    objective: str = "utility"
    turnover_control: str = "band"
    gamma: float = 1.0
    lam_risk: float = 2.0
    lam_turnover: float = 0.001
    epochs: int = 40
    lr: float = 1e-3
    seed: int = 1
    leaky: bool = False
    chunk: int = 96 * 30           # truncated BPTT window
    # Degeneracy guard (see losses.participation_penalty). Applied to BOTH objectives so
    # the utility-vs-robust_sharpe comparison stays like-for-like.
    min_exposure: float = 0.05
    # 100.0, not 10.0. Measured on f08 (3 pairs, 1 epoch): at weight 10 the robust_sharpe
    # band arm still converged to mean exposure 0.0001 and 100% flat — the guard did not
    # bite. At 100 it reaches the 0.05 floor and trades. The guard is inert where exposure
    # is already above the floor (identical results at 0 and 100 for the penalty arms), so
    # raising it does not distort the configurations that never collapsed.
    participation_weight: float = 100.0

    def describe(self) -> dict:
        return {"objective": self.objective, "turnover_control": self.turnover_control,
                "gamma": self.gamma, "lam_risk": self.lam_risk,
                "lam_turnover": self.lam_turnover, "epochs": self.epochs,
                "lr": self.lr, "seed": self.seed, "leaky": self.leaky,
                "min_exposure": self.min_exposure,
                "participation_weight": self.participation_weight}


def train_fold(features: np.ndarray, open_px: np.ndarray, cost_per_unit: np.ndarray,
               volume_quote: np.ndarray, cfg: TrainConfig,
               mask: np.ndarray | None = None) -> tuple[ExposurePolicy, dict]:
    """Train one policy on one fold's training range. Returns the policy and a history."""
    mx.random.seed(cfg.seed)
    x = mx.array(features.astype(np.float32))
    # Normalise to batched (B, T, F). Slicing axis 0 on a pooled (n_pairs, T, F) panel
    # slices the PAIR axis, not time — which silently feeds the whole sequence in and
    # blows Metal's resource limit. Every slice below therefore indexes axis 1 explicitly.
    batched = x.ndim == 3
    if not batched:
        x = x[None, ...]
    if cfg.leaky:
        x = mx.stack([sim.shift_features_leaky(x[b]) for b in range(x.shape[0])])

    pol = ExposurePolicy(PolicyConfig(n_features=x.shape[-1],
                                      turnover_control=cfg.turnover_control))
    opt = optim.Adam(learning_rate=cfg.lr)
    obj = L.OBJECTIVES[cfg.objective]

    def _2d(a):
        a = mx.array(a.astype(np.float32))
        return a if a.ndim == 2 else a[None, ...]

    o = _2d(open_px)
    c = _2d(cost_per_unit * cfg.gamma)
    v = _2d(volume_quote)
    # Zero exposure where a pair has no data at that timestamp: a gap must earn nothing and
    # cost nothing, rather than silently holding a stale position through it.
    m = _2d(mask) if mask is not None else None

    # Truncated BPTT (R7). A single forward pass over a full fold (~139k bars) exceeds
    # Metal's resource limit, so each step takes a window of BURN_IN + chunk bars: the
    # burn-in warms the recurrent state and is dropped from the loss, the remainder carries
    # the gradient. Windows are drawn in a seeded, deterministic order so runs reproduce.
    span = BURN_IN + cfg.chunk
    T_total = x.shape[1]                      # axis 1 is time; axis 0 is the pair panel
    n_windows = max(1, (T_total - 2) // cfg.chunk)
    rng = np.random.default_rng(cfg.seed)
    starts = [min(i * cfg.chunk, max(0, T_total - span - 2)) for i in range(n_windows)]

    def loss_fn(model, s0: int):
        xs = x[:, s0:s0 + span]
        os_ = o[:, s0:s0 + span + 2]
        cs = c[:, s0:s0 + span]
        vs = v[:, s0:s0 + span]
        w = model(xs)
        if m is not None:
            w = w * m[:, s0:s0 + span]
        r = sim.net_returns(w, os_, cs, volume_quote=vs, soft_cap=True)
        r = r[..., BURN_IN:]
        base = obj(r, cfg.lam_risk) if cfg.objective == "utility" else obj(r)
        # Degeneracy guard: inaction must not be the cheapest way to score well.
        base = base + L.participation_penalty(
            w[..., BURN_IN:], cfg.min_exposure, cfg.participation_weight,
            mask=None if m is None else m[:, s0:s0 + span][..., BURN_IN:])
        if cfg.turnover_control == "penalty":
            # `...` indexes time. A bare w[BURN_IN:] slices the PAIR axis: on a 10-pair
            # panel w[96:] is EMPTY, mean() of it is NaN, and the whole loss goes NaN while
            # training silently continues. That cost 26 wasted runs on 2026-09-10.
            base = base + L.turnover_penalty(w[..., BURN_IN:], cfg.lam_turnover)
        return base

    grad_fn = nn.value_and_grad(pol, loss_fn)
    history, best = [], float("inf")
    for _ in range(cfg.epochs):
        order = rng.permutation(len(starts))
        ep = []
        for j in order:
            loss, grads = grad_fn(pol, int(starts[j]))
            opt.update(pol, grads)
            # Every updated array must be evaluated each step or Metal's limit trips.
            mx.eval(pol.parameters(), opt.state, loss)
            lv = float(loss)
            if not np.isfinite(lv):
                raise FloatingPointError(
                    f"non-finite loss at epoch {len(history)}, window {int(starts[j])}: {lv}")
            ep.append(lv)
        v_loss = float(np.mean(ep))
        history.append(v_loss)
        best = min(best, v_loss)

    # Report over the whole range, chunked so the graph stays inside Metal's limit.
    ws, rs = [], []
    for s0 in range(0, T_total - 2, cfg.chunk):
        xs = x[:, s0:s0 + cfg.chunk]
        if xs.shape[1] < 4:
            break
        w_c = pol(xs)
        if m is not None:
            w_c = w_c * m[:, s0:s0 + cfg.chunk]
        r_c = sim.net_returns(w_c, o[:, s0:s0 + cfg.chunk + 2], c[:, s0:s0 + cfg.chunk],
                              volume_quote=v[:, s0:s0 + cfg.chunk], soft_cap=False)
        mx.eval(w_c, r_c)
        ws.append(np.asarray(w_c)); rs.append(np.asarray(r_c))
    w = mx.array(np.concatenate(ws, axis=1)); r = mx.array(np.concatenate(rs, axis=1))
    rn = np.asarray(r[:, BURN_IN:])
    wn = np.asarray(w[:, BURN_IN:])
    return pol, {
        "final_loss": history[-1], "best_loss": best, "epochs": cfg.epochs,
        # Per-pair sums averaged, so a pooled panel is not just N times a single pair's
        # return — the figure stays comparable across different universe sizes.
        "total_log_return": float(rn.sum(axis=1).mean()),
        "total_return_pct": float((np.exp(rn.sum(axis=1).mean()) - 1) * 100),
        "n_pairs": int(rn.shape[0]),
        "mean_exposure": float(wn.mean()),
        "flat_share": float((wn < 0.01).mean()),
        "turnover": float(np.abs(np.diff(wn, axis=1)).sum(axis=1).mean()),
        "history": history,
    }


def evaluate(pol: ExposurePolicy, features: np.ndarray, open_px: np.ndarray,
             cost_per_unit: np.ndarray, volume_quote: np.ndarray,
             mask: np.ndarray | None = None, chunk: int = 96 * 30) -> dict:
    """Forward-only evaluation of a trained policy on a held-out range.

    No gradient, no optimiser, hard liquidity cap and full cost — this is what the
    pre-registered carry-forward rule ranks on. Selecting on the TRAINING range instead
    selects for overfitting, which is the failure this function exists to prevent.
    """
    x = mx.array(features.astype(np.float32))
    if x.ndim == 2:
        x = x[None, ...]

    def _2d(a):
        a = mx.array(a.astype(np.float32))
        return a if a.ndim == 2 else a[None, ...]

    o, c, v = _2d(open_px), _2d(cost_per_unit), _2d(volume_quote)
    m = _2d(mask) if mask is not None else None

    ws, rs = [], []
    for s0 in range(0, x.shape[1] - 2, chunk):
        xs = x[:, s0:s0 + chunk]
        if xs.shape[1] < 4:
            break
        w = pol(xs)
        if m is not None:
            w = w * m[:, s0:s0 + chunk]
        r = sim.net_returns(w, o[:, s0:s0 + chunk + 2], c[:, s0:s0 + chunk],
                            volume_quote=v[:, s0:s0 + chunk], soft_cap=False)
        mx.eval(w, r)
        ws.append(np.asarray(w)); rs.append(np.asarray(r))
    wn = np.concatenate(ws, axis=1)[:, BURN_IN:]
    rn = np.concatenate(rs, axis=1)[:, BURN_IN:]
    return {
        "log_return": float(rn.sum(axis=1).mean()),
        "return_pct": float((np.exp(rn.sum(axis=1).mean()) - 1) * 100),
        "mean_exposure": float(wn.mean()),
        "flat_share": float((wn < 0.01).mean()),
        "turnover": float(np.abs(np.diff(wn, axis=1)).sum(axis=1).mean()),
        "n_pairs": int(rn.shape[0]),
    }

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

    def describe(self) -> dict:
        return {"objective": self.objective, "turnover_control": self.turnover_control,
                "gamma": self.gamma, "lam_risk": self.lam_risk,
                "lam_turnover": self.lam_turnover, "epochs": self.epochs,
                "lr": self.lr, "seed": self.seed, "leaky": self.leaky}


def train_fold(features: np.ndarray, open_px: np.ndarray, cost_per_unit: np.ndarray,
               volume_quote: np.ndarray, cfg: TrainConfig) -> tuple[ExposurePolicy, dict]:
    """Train one policy on one fold's training range. Returns the policy and a history."""
    mx.random.seed(cfg.seed)
    x = mx.array(features.astype(np.float32))
    if cfg.leaky:
        x = sim.shift_features_leaky(x)

    pol = ExposurePolicy(PolicyConfig(n_features=x.shape[-1],
                                      turnover_control=cfg.turnover_control))
    opt = optim.Adam(learning_rate=cfg.lr)
    obj = L.OBJECTIVES[cfg.objective]

    o = mx.array(open_px.astype(np.float32))
    c = mx.array((cost_per_unit * cfg.gamma).astype(np.float32))
    v = mx.array(volume_quote.astype(np.float32))

    # Truncated BPTT (R7). A single forward pass over a full fold (~139k bars) exceeds
    # Metal's resource limit, so each step takes a window of BURN_IN + chunk bars: the
    # burn-in warms the recurrent state and is dropped from the loss, the remainder carries
    # the gradient. Windows are drawn in a seeded, deterministic order so runs reproduce.
    span = BURN_IN + cfg.chunk
    T_total = x.shape[0]
    n_windows = max(1, (T_total - 2) // cfg.chunk)
    rng = np.random.default_rng(cfg.seed)
    starts = [min(i * cfg.chunk, max(0, T_total - span - 2)) for i in range(n_windows)]

    def loss_fn(model, s0: int):
        xs = x[s0:s0 + span]
        os_ = o[s0:s0 + span + 2]
        cs = c[s0:s0 + span]
        vs = v[s0:s0 + span]
        w = model(xs)
        r = sim.net_returns(w, os_, cs, volume_quote=vs, soft_cap=True)
        r = r[BURN_IN:]
        base = obj(r, cfg.lam_risk) if cfg.objective == "utility" else obj(r)
        if cfg.turnover_control == "penalty":
            base = base + L.turnover_penalty(w[BURN_IN:], cfg.lam_turnover)
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
            ep.append(float(loss))
        v_loss = float(np.mean(ep))
        history.append(v_loss)
        best = min(best, v_loss)

    # Report over the whole range, chunked so the graph stays inside Metal's limit.
    ws, rs = [], []
    for s0 in range(0, T_total - 2, cfg.chunk):
        xs = x[s0:s0 + cfg.chunk]
        if xs.shape[0] < 4:
            break
        w_c = pol(xs)
        r_c = sim.net_returns(w_c, o[s0:s0 + cfg.chunk + 2], c[s0:s0 + cfg.chunk],
                              volume_quote=v[s0:s0 + cfg.chunk], soft_cap=False)
        mx.eval(w_c, r_c)
        ws.append(np.asarray(w_c)); rs.append(np.asarray(r_c))
    w = mx.array(np.concatenate(ws)); r = mx.array(np.concatenate(rs))
    rn = np.asarray(r[BURN_IN:])
    wn = np.asarray(w[BURN_IN:])
    return pol, {
        "final_loss": history[-1], "best_loss": best, "epochs": cfg.epochs,
        "total_log_return": float(rn.sum()),
        "total_return_pct": float((np.exp(rn.sum()) - 1) * 100),
        "mean_exposure": float(wn.mean()),
        "flat_share": float((wn < 0.01).mean()),
        "turnover": float(np.abs(np.diff(wn)).sum()),
        "history": history,
    }

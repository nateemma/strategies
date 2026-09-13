"""Arms A/B/C as configurations of brief 01's trainer (T014).

Arms differ ONLY in initialisation and objective. Everything else — simulator, costs,
features, folds, participation floor, held-out ranking, trial log, seeds — is brief 01's,
unchanged (FR-002).

    A warm_start   classifier encoder, per fold   utility loss, encoder LR <= head_LR/10
    B blended      random                         alpha*CE(label) + (1-alpha)*utility
    C auxiliary    random                         utility + forward-volatility head
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.labelfree import features as FT          # noqa: E402
from research.labelfree import folds as F              # noqa: E402
from research.labelfree import labels as LB            # noqa: E402
from research.labelfree import losses as L             # noqa: E402
from research.labelfree import sim                     # noqa: E402
from research.labelfree import stats as ST             # noqa: E402
from research.labelfree import trials                  # noqa: E402
from research.labelfree import warmstart as W          # noqa: E402
from research.labelfree.costs import CostModel         # noqa: E402
from research.labelfree.policy import ClassifierEncoderPolicy   # noqa: E402

BURN_IN = 96
SEQ_LEN = 16
CHUNK = 96 * 30


@dataclass(frozen=True)
class ArmConfig:
    arm: str                       # warm_start | blended | auxiliary | pnl_control
    seed: int = 1
    fold_id: str = "f08"
    turnover_control: str = "band"
    gamma: float = 1.0
    lam_risk: float = 2.0
    epochs: int = 2
    head_lr: float = 1e-3
    encoder_lr: float = 1e-4       # Arm A only; must be <= head_lr/10 (FR-005)
    blend_alpha: float | None = None      # Arm B
    label_threshold: float | None = None  # Arm B
    min_exposure: float = 0.05
    participation_weight: float = 100.0
    lam_turnover: float = 0.0      # explicit |dw| penalty; only read when control=="penalty"
    band_width: float = 0.10       # no-trade half-width scale; the turnover lever for "band"
    warm_start_ref: str | None = None

    def describe(self) -> dict:
        d = {"arm": self.arm, "seed": self.seed, "fold_id": self.fold_id,
             "turnover_control": self.turnover_control, "gamma": self.gamma,
             "epochs": self.epochs, "head_lr": self.head_lr,
             "min_exposure": self.min_exposure,
             "participation_weight": self.participation_weight}
        if self.turnover_control == "penalty":
            d["lam_turnover"] = self.lam_turnover
        else:
            d["band_width"] = self.band_width
        if self.arm == "warm_start":
            d.update(encoder_lr=self.encoder_lr, warm_start_ref=self.warm_start_ref)
        if self.arm == "blended":
            d.update(blend_alpha=self.blend_alpha, label_threshold=self.label_threshold)
        return d


def windows(feat: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """(T, F) -> (T-seq+1, seq, F). Built PER CHUNK by the caller: panel-wide is 2.14 GB."""
    n, k = feat.shape
    idx = np.arange(seq_len)[None, :] + np.arange(n - seq_len + 1)[:, None]
    return feat[idx]


def load_panel(fold, pairs: list[str], train: bool, label_threshold: float | None):
    """Date-aligned panel plus optional label/vol targets, with leakage control applied.

    Alignment is on DATE, never position — inherited from feature 009, where stacking by
    position put 2022-06-22 and 2024-12-20 in the same cross-section.
    """
    from regime import regimedata as rd
    sys.path.insert(0, str(ROOT / "utils"))
    from DataframePopulator import DataframePopulator

    start, end = (fold.train_start, fold.train_end) if train else (fold.test_start, fold.test_end)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    index = pd.date_range(lo, hi, freq="15min", tz="UTC")

    X, P, C, V, M, Y, Vol = [], [], [], [], [], [], []
    for p in pairs:
        feat, px, vq, _ = FT.build_features(p, start, end)
        df = rd.load_ohlcv(p.split("/")[0])
        df = df[(df.index >= lo) & (df.index <= hi)]
        n = min(len(df), len(px), len(feat))
        own = df.index[:n]
        cost = CostModel().per_unit(df, evaluation=True).to_numpy()[:n]

        fx = pd.DataFrame(feat[:n], index=own).reindex(index)
        valid = fx.notna().all(axis=1).to_numpy()
        fx = fx.ffill().fillna(0.0)
        X.append(fx.to_numpy(dtype=np.float32))
        P.append(pd.Series(px[:n], index=own).reindex(index).ffill().bfill().to_numpy())
        C.append(pd.Series(cost, index=own).reindex(index).ffill().bfill().to_numpy())
        V.append(pd.Series(vq[:n], index=own).reindex(index).fillna(0.0).to_numpy())
        M.append(valid.astype(np.float32))

        pop = DataframePopulator().add_minimal_indicators(df.reset_index().copy())
        if label_threshold is not None:
            t = LB.directional_label(pop, min_gain=label_threshold)
            Y.append(pd.Series(t.values[:n], index=own).reindex(index).fillna(LB.HOLD)
                     .to_numpy().astype(int))
        v = LB.forward_realised_vol(pop)
        Vol.append(pd.Series(v.values[:n], index=own).reindex(index).fillna(0.0).to_numpy())

    Mst = np.stack(M)

    # C1 (analyze 2026-09-12) -- PURGE AND EMBARGO AT THE FOLD BOUNDARY (FR-013, FR-014).
    #
    # These were computed by labels.apply_leakage_control and never applied: every Fold
    # carried purge_bars=0, and load_panel kept LabelTarget.values while discarding
    # LabelTarget.valid. Both arms with a forward-looking term (B's label CE, C's forward
    # realised vol) therefore trained on boundary bars whose target reaches past the split.
    #
    # Train and test panels are loaded separately, so the boundary is an EDGE of each
    # panel, not an interior index: purge the last `horizon` bars of train, embargo the
    # first `horizon` of test. Folding it into the validity mask M is what makes it bind --
    # M already gates the loss everywhere downstream.
    h = LB.DEFAULT_HORIZON
    if train:
        Mst[:, -h:] = 0.0          # purge: their forward window crosses into the test range
    else:
        Mst[:, :h] = 0.0           # embargo: they overlap the training range's own windows
    leakage_record = {"purge_bars": h if train else 0,
                      "embargo_bars": 0 if train else h,
                      "side": "train" if train else "test",
                      "n_bars": int(Mst.shape[1]),
                      "usable_share": float(Mst.mean())}

    return (np.stack(X), np.stack(P), np.stack(C), np.stack(V), Mst,
            np.stack(Y) if Y else None, np.stack(Vol), index, leakage_record)


def build_policy(cfg: ArmConfig, n_features: int):
    """Arm A inherits the encoder; B and C initialise the same architecture randomly."""
    mx.random.seed(cfg.seed)
    pol = ClassifierEncoderPolicy(n_features=n_features,
                                  turnover_control=cfg.turnover_control,
                                  band_width=cfg.band_width)
    if cfg.arm == "warm_start":
        if not cfg.warm_start_ref:
            raise ValueError("warm_start requires warm_start_ref")
        W.check_learning_rates(cfg.encoder_lr, cfg.head_lr)
        before = W.differs_from_random(pol, n_features, seed=cfg.seed)
        W.load_encoder(pol, mx.load(cfg.warm_start_ref))
        if not W.differs_from_random(pol, n_features, seed=cfg.seed):
            raise ValueError(
                "warm-start loaded weights identical to random init — the arm would be "
                "indistinguishable from Arm B and reported as a warm-start result")
        assert not before
    return pol


def train_arm(cfg: ArmConfig, panel_tr, panel_te) -> dict:
    """Train one arm on the fold's training panel, evaluate on its held-out panel."""
    Xtr, Ptr, Ctr, Vtr, Mtr, Ytr, Voltr, _, _leak_tr = panel_tr
    n_pairs, T, n_feat = Xtr.shape
    pol = build_policy(cfg, n_feat)

    # Arm A trains the inherited encoder an order of magnitude slower than the new head.
    enc_names = set(W.ENCODER_PREFIXES)
    opt_head = optim.Adam(learning_rate=cfg.head_lr)
    opt_enc = optim.Adam(learning_rate=cfg.encoder_lr) if cfg.arm == "warm_start" else None

    obj = L.utility
    rng = np.random.default_rng(cfg.seed)
    starts = [s for s in range(0, T - CHUNK - 2, CHUNK)] or [0]
    history = []

    def loss_fn(model, s0):
        total = mx.array(0.0)
        for pi in range(n_pairs):
            xw = mx.array(windows(Xtr[pi, s0:s0 + CHUNK + SEQ_LEN - 1]))
            w = model(xw)
            m = mx.array(Mtr[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1])
            k = min(w.shape[0], m.shape[0])
            w = w[:k] * m[:k]
            o = mx.array(Ptr[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN + 1])
            c = mx.array(Ctr[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1])[:k]
            v = mx.array(Vtr[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1])[:k]
            r = sim.net_returns(w, o, c * cfg.gamma, volume_quote=v, soft_cap=True)
            total = total + obj(r[BURN_IN:], cfg.lam_risk)
            total = total + L.participation_penalty(
                w[BURN_IN:], cfg.min_exposure, cfg.participation_weight,
                mask=m[:k][BURN_IN:])
            if cfg.turnover_control == "penalty":
                # MISSING from arms.py while present in train.py:111 -- so
                # turnover_control="penalty" gave the policy-side learned band with no
                # penalty attached. Half-wired, and silent. `w` is 1-D here (the loop is
                # per pair), so [BURN_IN:] is the time axis; on a pooled (n_pairs, T)
                # array it would slice the PAIR axis and go NaN, which cost 26 runs on
                # 2026-09-10 -- see train.py's comment.
                total = total + L.turnover_penalty(w[BURN_IN:], cfg.lam_turnover)
            if cfg.arm == "blended" and Ytr is not None:
                y = mx.array(np.eye(3, dtype=np.float32)[
                    Ytr[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1][:k]])
                total = total + cfg.blend_alpha * L.exposure_ce(w[:k], y)
            if cfg.arm == "auxiliary":
                tgt = mx.array(Voltr[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1][:k])
                total = total + L.aux_vol_loss(w[:k], tgt)
        return total / n_pairs

    grad_fn = nn.value_and_grad(pol, loss_fn)
    for _ in range(cfg.epochs):
        ep = []
        for j in rng.permutation(len(starts)):
            loss, grads = grad_fn(pol, int(starts[j]))
            if opt_enc is None:
                opt_head.update(pol, grads)
            else:
                enc_g = {k: v for k, v in grads.items() if k in enc_names}
                head_g = {k: v for k, v in grads.items() if k not in enc_names}
                opt_enc.update(pol, enc_g)
                opt_head.update(pol, head_g)
            mx.eval(pol.parameters(), opt_head.state, loss)
            lv = float(loss)
            if not np.isfinite(lv):
                raise FloatingPointError(f"non-finite loss at window {int(starts[j])}")
            ep.append(lv)
        history.append(float(np.mean(ep)))

    ev = evaluate_arm(pol, panel_te, cfg)
    ev["final_loss"] = history[-1]
    return ev


def evaluate_arm(pol, panel_te, cfg: ArmConfig) -> dict:
    """Forward-only on the held-out panel: hard cap, full cost (FR-017)."""
    Xte, Pte, Cte, Vte, Mte, Yte, _Vol, _idx, _leak_te = panel_te
    n_pairs, T, _ = Xte.shape
    rs, ws = [], []
    for pi in range(n_pairs):
        pr, pw = [], []
        for s0 in range(0, T - CHUNK - 2, CHUNK) or [0]:
            xw = mx.array(windows(Xte[pi, s0:s0 + CHUNK + SEQ_LEN - 1]))
            w = pol(xw)
            m = mx.array(Mte[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1])
            k = min(w.shape[0], m.shape[0])
            w = w[:k] * m[:k]
            o = mx.array(Pte[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN + 1])
            c = mx.array(Cte[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1])[:k]
            v = mx.array(Vte[pi, s0 + SEQ_LEN - 1: s0 + CHUNK + SEQ_LEN - 1])[:k]
            r = sim.net_returns(w, o, c, volume_quote=v, soft_cap=False)
            mx.eval(w, r)
            pw.append(np.asarray(w)); pr.append(np.asarray(r))
        ws.append(np.concatenate(pw)); rs.append(np.concatenate(pr))
    wn = np.stack(ws)[:, BURN_IN:]; rn = np.stack(rs)[:, BURN_IN:]
    out = {"test_log_return": float(rn.sum(axis=1).mean()),
           "test_mean_exposure": float(wn.mean()),
           "test_flat_share": float((wn < 0.01).mean()),
           "test_turnover": float(np.abs(np.diff(wn, axis=1)).sum(axis=1).mean()),
           "n_pairs": int(rn.shape[0])}
    # Activity travels WITH the return (FR-019, SC-014): a return quoted alone cannot be
    # told apart from an arm that scored well by declining to trade.
    act = ST.activity(wn)
    out.update(test_trades_per_year=act["trades_per_year"],
               test_mean_hold_hours=act["mean_hold_hours"],
               test_turnover_per_year=act["turnover_per_year"])
    out["degenerate"] = bool(out["test_mean_exposure"] < cfg.min_exposure)
    if cfg.arm == "warm_start" and Yte is not None:
        y = Yte[:, BURN_IN + SEQ_LEN - 1:][:, :wn.shape[1]]
        k = min(y.shape[1], wn.shape[1])
        out["exposure_label_corr"] = float(
            np.corrcoef(wn[:, :k].ravel(), (y[:, :k] == LB.BUY).astype(float).ravel())[0, 1])
    return out


# ---------------------------------------------------------------------------
# Screening runner (T015) — pre-registers every configuration before it runs.
# ---------------------------------------------------------------------------

def run_screen(arm_specs: list[dict], fold_id: str = "f08",
               pairs: list[str] | None = None, epochs: int = 2,
               seeds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    fold = {f.fold_id: f for f in F.folds()}[fold_id]
    pairs = pairs or list(fold.pair_universe)
    run_id = trials.new_run_id()
    print(f"screen fold {fold_id} [run {run_id}] {len(pairs)} pairs, "
          f"{len(arm_specs)} arms x {len(seeds)} seeds", flush=True)

    # ONE PANEL PER DISTINCT LABEL THRESHOLD. Loading a single panel from the first
    # spec's threshold makes `label_threshold` silently INERT: every arm then trains
    # against the same label and the configurations come back byte-identical, which reads
    # as "the two labels behave the same" when it is really "the parameter did nothing".
    # Caught 2026-09-12 after 15 runs were wasted that way.
    thresholds = sorted({a.get("label_threshold") for a in arm_specs
                         if a.get("label_threshold")} or {LB.PRODUCTION_THRESHOLD})
    panels = {}
    for thr in thresholds:
        panels[thr] = (load_panel(fold, pairs, True, thr),
                       load_panel(fold, pairs, False, thr))
        lk_tr, lk_te = panels[thr][0][8], panels[thr][1][8]
        print(f"  panel@thr={thr}: train {panels[thr][0][0].shape} "
              f"({100*panels[thr][0][4].mean():.0f}% valid)  "
              f"purge={lk_tr['purge_bars']} embargo={lk_te['embargo_bars']}", flush=True)
        # FR-014: a run whose purge silently fell to zero is VOID, not caveated -- so the
        # values ACTUALLY applied are asserted here rather than assumed from the config.
        if lk_tr["purge_bars"] <= 0 or lk_te["embargo_bars"] <= 0:
            raise ValueError(f"leakage control not applied: {lk_tr} / {lk_te} (FR-013/FR-014)")

    # Liveness proof (Principle IV): the label must actually differ between thresholds.
    if len(thresholds) > 1:
        import numpy as _np
        a, b = (panels[t][0][5] for t in thresholds[:2])
        if a is not None and b is not None and _np.array_equal(a, b):
            raise ValueError(
                f"label_threshold is INERT: thresholds {thresholds[:2]} produced identical "
                "labels. Every downstream comparison between them would be vacuous.")
        diff = float((a != b).mean()) if a is not None and b is not None else 0.0
        print(f"  label liveness: thresholds differ on {100*diff:.1f}% of bars", flush=True)

    results: dict[str, list[dict]] = {}
    for spec in arm_specs:
        key = spec.get("tag") or spec["arm"]
        results[key] = []
        for seed in seeds:
            cfg = ArmConfig(seed=seed, fold_id=fold_id, epochs=epochs,
                            **{k: v for k, v in spec.items() if k != "tag"})
            tid = f"{key}-s{seed}-{fold_id}@{run_id}"
            trials.pre_register({"trial_id": tid, "run_id": run_id, "feature": "010",
                                 "n_pairs": len(pairs), **cfg.describe()})
            t0 = time.time()
            tr, te = panels[cfg.label_threshold or thresholds[0]]
            try:
                r = train_arm(cfg, tr, te)
            except Exception as e:
                trials.abandon(tid, f"{type(e).__name__}: {str(e)[:200]}")
                print(f"    {tid}: ABANDONED {type(e).__name__}: {e}", flush=True)
                continue
            r["seconds"] = round(time.time() - t0, 1)
            r["status"] = "ok"
            trials.complete(tid, r)
            results[key].append(r)
            corr = r.get("exposure_label_corr")
            print(f"    {tid}: log_ret={r['test_log_return']:+.4f} "
                  f"mean_w={r['test_mean_exposure']:.3f} "
                  f"trades/yr={r['test_trades_per_year']:.1f} "
                  f"hold={r['test_mean_hold_hours']:.1f}h "
                  f"flat={100*r['test_flat_share']:.0f}% "
                  f"degen={r['degenerate']}"
                  + (f" corr={corr:+.3f}" if corr is not None else "")
                  + f" ({r['seconds']}s)", flush=True)
    return {"run_id": run_id, "fold": fold_id, "pairs": pairs, "results": results}


# ---------------------------------------------------------------------------
# T027 / T028 — degeneracy screen, then the pre-registered carry-forward rule.
# ---------------------------------------------------------------------------

def screen_degenerate(results: dict[str, list[dict]],
                      min_exposure: float = 0.05) -> dict:
    """FR-020 / SC-015: an arm below the participation floor is INCONCLUSIVE.

    Its return is deliberately NOT carried into the output. An arm that declines to
    trade can post an excellent number simply by avoiding the control's losses, and
    reporting that number invites it to be read as an edge. Judging degeneracy on the
    IQM across seeds, not per seed, keeps one flat seed from voiding an arm that trades.
    """
    from research.labelfree.admissibility import Arm

    eligible, inconclusive = {}, {}
    for key, runs in results.items():
        ok = [r for r in runs if r.get("status") == "ok"]
        if not ok:
            inconclusive[key] = {"reason": "no completed run", "n_ok": 0}
            continue
        w = ST.iqm([r["test_mean_exposure"] for r in ok])
        # Delegated to the SHARED gate so this site cannot drift from the others. Three
        # separate implementations of this floor existed before it was centralised, and the
        # comparison site that mattered checked only ONE SIDE.
        probe = Arm(key, ST.iqm([r["test_log_return"] for r in ok]), w,
                    ST.iqm([r.get("test_trades_per_year", float("nan")) for r in ok])
                    if all("test_trades_per_year" in r for r in ok) else None)
        if probe.degenerate:
            inconclusive[key] = {"reason": probe.reason,
                                 "mean_exposure": w, "floor": min_exposure,
                                 "n_ok": len(ok), "return": "WITHHELD (SC-015)"}
        else:
            eligible[key] = ok
    return {"eligible": eligible, "inconclusive": inconclusive,
            "min_exposure": min_exposure}


def select_arms(results: dict[str, list[dict]], min_exposure: float = 0.05) -> dict:
    """T028: rank on the HELD-OUT range only, via sweep.py's pre-registered rule.

    Ranking on the training range is the defect that voided 60 of brief 01's runs; the
    rule is imported rather than restated so the two cannot drift apart.
    """
    from research.labelfree.sweep import CARRY_FORWARD_RULE, select

    screened = screen_degenerate(results, min_exposure)
    held_out = {k: [r["test_log_return"] for r in runs]      # TEST range, never train
                for k, runs in screened["eligible"].items()}
    out = {"rule": CARRY_FORWARD_RULE, "screen": screened}
    if not held_out:
        out.update(resolved=False, carried=[],
                   reason="every arm was screened out as degenerate or incomplete")
        return out
    out.update(select(held_out))
    out["activity"] = {
        k: {"mean_exposure": ST.iqm([r["test_mean_exposure"] for r in runs]),
            "trades_per_year": ST.iqm([r.get("test_trades_per_year", float("nan"))
                                       for r in runs]),
            "mean_hold_hours": ST.iqm([r.get("test_mean_hold_hours", float("nan"))
                                       for r in runs])}
        for k, runs in screened["eligible"].items()}
    return out

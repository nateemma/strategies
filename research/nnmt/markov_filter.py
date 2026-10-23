"""040 DESCRIPTIVE: a causal Markov forward filter on NNMT's per-bar head probabilities (no retraining).

Not pre-registered (written after the head and regime-state screens): results are descriptive only and can
only motivate a pre-registered P&L arm, never adopt anything.

Per model, fold, head and pair, along time over the fold's test bars:

    f_0 = pi,   f_t ∝ (p_t / pi) ⊙ (f_{t-1} @ T)

p_t = the head's class probabilities (a posterior), p_t / pi = its likelihood up to a constant (dividing by
the prior stops the filter counting the base rate twice); T = the one-bar label transition matrix and pi =
the label class shares, both counted per pair on that fold's TRAINING range only (dates before test_start
minus the label's look-ahead), pooled over pairs. No tuned parameter. The trading head is filtered as two
states (buy vs not buy) against label:buy (037 cache), as head_screen scores it.

Reported, raw vs filtered, per head and half: one-vs-rest AUC per class (trading: also on gbb candidate bars),
and flicker = changes per pair-day of the argmax class (trading: of p_buy >= the operating threshold 0.80),
plus the flagged share / precision / recall at 0.80 and 0.90.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.nnmt.markov_filter --models mt_lr_ri,mt_lr_ri_ah
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNMT")]

from research.nnmt.head_screen import NEED, SIG, TRADING, _auc, _strategy, head_labels, precision_recall  # noqa: E402

HERE = Path(__file__).resolve().parent
HEADS = ["regime", "risk", "momentum", "flow", "profit"]
BAR = pd.Timedelta(minutes=15)
LOOKAHEAD_BARS = 192  # >= every auxiliary/trading label's future reach (regime: centred 200 + 5 shift; gbb h48)
OP_THRESHOLD, THRESHOLDS = 0.80, [0.8, 0.9]
EPS = 1e-6


def transition_matrix(seqs: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, pi) from per-pair class sequences (ints, negative = unknown); transitions never cross a pair or
    an unknown label. Rows with no counts fall back to pi."""
    counts, occ = np.zeros((k, k)), np.zeros(k)
    for s in seqs:
        s = np.asarray(s, int)
        ok = s >= 0
        occ += np.bincount(s[ok], minlength=k)[:k]
        a, b = s[:-1], s[1:]
        m = (a >= 0) & (b >= 0)
        np.add.at(counts, (a[m], b[m]), 1.0)
    pi = occ / occ.sum()
    rs = counts.sum(axis=1, keepdims=True)
    T = np.where(rs > 0, counts / np.where(rs > 0, rs, 1.0), pi[None, :])
    return T, pi


def forward_filter(p: np.ndarray, T: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Causal forward filter of an (n, k) posterior sequence; NaN rows pass the prediction step only."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    f = pi.copy()
    for t in range(len(p)):
        prior = f @ T
        if np.isfinite(p[t]).all():
            f = np.clip(p[t], EPS, None) / pi * prior
            f /= f.sum()
            out[t] = f
        else:
            f = prior
    return out


def _flicker(state: pd.Series, g: pd.DataFrame) -> float:
    """State changes per pair-day."""
    ch = state.groupby(g["pair"]).transform(lambda s: s.ne(s.shift()) & s.shift().notna())
    days = g.groupby("pair")["date"].agg(lambda d: (d.max() - d.min()) / pd.Timedelta(days=1)).sum()
    return round(float(ch.sum() / days), 4) if days > 0 else None


def _labels(tag: str, folds: list[dict]) -> pd.DataFrame:
    """Per pair: the head labels (and label:buy, gbb candidate) over the full cache."""
    from research.indicator_screen import harvest as H

    s = _strategy()
    rows = []
    pairs = sorted({p for f in folds for p in pd.read_parquet(SIG / f"{tag.upper()}_{f['fold_id']}.parquet",
                                                                columns=["pair"])["pair"].unique()})
    for pair in pairs:
        c = H.read_cache(pair, NEED + TRADING)
        c["date"] = pd.to_datetime(c["date"], utc=True)
        lab = head_labels(c, s)
        lab["trading"] = c["label:buy"].to_numpy()
        lab = lab.add_prefix("lab_")
        rows.append(pd.concat([c[["date", "volume", "base:gbb_candidate"]].reset_index(drop=True), lab], axis=1)
                    .assign(pair=pair))
    return pd.concat(rows, ignore_index=True)


def _seq(x: pd.Series) -> np.ndarray:
    return np.where(x.notna(), x.fillna(-1), -1).astype(int)


def run_model(tag: str, halves: dict) -> dict:
    folds = json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}.json").read_text())["folds"]
    lab = _labels(tag, folds)
    scored, Ts = [], {}
    for f in folds:
        t = pd.read_parquet(SIG / f"{tag.upper()}_{f['fold_id']}.parquet")
        t["date"] = pd.to_datetime(t["date"], utc=True)
        t = t.sort_values(["pair", "date"]).merge(lab, on=["date", "pair"], how="left", suffixes=("", "_c"))
        ts = pd.Timestamp(f["test_start"], tz="UTC")
        tr = lab[(lab["date"] >= pd.Timestamp(f["train_start"], tz="UTC"))
                 & (lab["date"] < ts - LOOKAHEAD_BARS * BAR)]
        Ts[f["fold_id"]] = {}
        for head in HEADS + ["trading"]:
            if head == "trading":
                cols, k = ["p_buy"], 2
                seqs = [_seq(g["lab_trading"]) for _, g in tr.groupby("pair")]
            else:
                cols = [c for c in t.columns if c.startswith(f"p_{head}_")]
                k = len(cols)
                seqs = [_seq(g[f"lab_{head}"]) for _, g in tr.groupby("pair")]
            T, pi = transition_matrix(seqs, k)
            Ts[f["fold_id"]][head] = {"diag": np.round(np.diag(T), 4).tolist(), "pi": np.round(pi, 4).tolist()}
            for pair, g in t.groupby("pair", sort=False):
                p = g[cols].to_numpy(float)
                if head == "trading":
                    p = np.column_stack([1.0 - p[:, 0], p[:, 0]])
                    t.loc[g.index, "f_buy"] = forward_filter(p, T, pi)[:, 1]
                else:
                    ff = forward_filter(p, T, pi)
                    for j, c in enumerate(cols):
                        t.loc[g.index, "f" + c[1:]] = ff[:, j]
        scored.append(t.assign(fold=f["fold_id"]))
    t = pd.concat(scored, ignore_index=True)
    t = t[t["volume"] > 0]
    out = {"heads": {}, "trading": {}, "transition": Ts}
    for head in HEADS:
        cols = [c for c in t.columns if c.startswith(f"p_{head}_")]
        r = {}
        for half, fl in halves.items():
            h = t[t["fold"].isin(fl) & t[f"lab_{head}"].notna()]
            auc = {}
            for j, c in enumerate(cols):
                y = (h[f"lab_{head}"] == j).astype(float)
                a, b = _auc(h[c], y), _auc(h["f" + c[1:]], y)
                auc[c[len(f"p_{head}_"):]] = {"raw": a, "filtered": b, "delta": None if a is None or b is None else b - a}
            raw_s = pd.Series(np.argmax(h[cols].to_numpy(float), axis=1), index=h.index)
            fil_s = pd.Series(np.argmax(np.nan_to_num(h[["f" + c[1:] for c in cols]].to_numpy(float), nan=-1), axis=1),
                              index=h.index)
            r[half] = {"auc": auc, "flicker_per_pair_day": {"raw": _flicker(raw_s, h), "filtered": _flicker(fil_s, h)}}
        out["heads"][head] = r
    for half, fl in halves.items():
        h = t[t["fold"].isin(fl) & t["lab_trading"].notna()]
        for pop, g in (("all", h), ("candidate", h[h["base:gbb_candidate"] > 0.5])):
            y = g["lab_trading"]
            out["trading"][f"{half}/{pop}"] = {
                "auc": {"raw": _auc(g["p_buy"], y), "filtered": _auc(g["f_buy"], y)},
                "flicker_per_pair_day": {k: _flicker((g[c] >= OP_THRESHOLD).astype(int), g)
                                         for k, c in (("raw", "p_buy"), ("filtered", "f_buy"))},
                "by_threshold": {"raw": precision_recall(g["p_buy"], y, THRESHOLDS),
                                 "filtered": precision_recall(g["f_buy"], y, THRESHOLDS)}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated plan tags, e.g. mt_lr_ri,mt_lr_ri_ah")
    args = ap.parse_args(argv)
    pr = json.loads((HERE / "preregistration_screens.json").read_text())
    halves = {"early": pr["early"], "late": pr["late"]}
    p = HERE / "markov_filter_descriptive.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    res["note"] = "DESCRIPTIVE, not pre-registered (written after the head and regime-state screens)"
    for tag in args.models.split(","):
        res[tag] = run_model(tag, halves)
        p.write_text(json.dumps(res, indent=1, default=str))
        print(f"{tag} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

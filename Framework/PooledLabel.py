"""Pooled, reference-relative training labels (feature 033; strategy-agnostic).

A strategy may declare ``POOLED_LABEL`` (a dict, see ``validate``). When it does,
``TrainingEngine._train_keyed`` replaces the labels of the declared eligible bars, pooled over
every pair of the training run, before training:

  * each eligible bar gets a barrier-trade outcome (``TrainingSignals.path_outcome``);
  * per side, bars are bucketed by equal-count quantiles of a reference column, fitted on the
    training rows only (``fit_ends``) and applied unchanged to the validation rows;
  * within a bucket, bars are ranked by "goodness" = (outcome, tie-break); the top ``share`` of
    the entry side is BUY, the bottom ``share`` of the exit side is SELL, the rest HOLD.

So within every bucket about ``share`` of the bars are positive, and the reference column alone
carries ~no label information. Bars outside both eligibility rules keep their label; ``%wr_keep``
(1.0 / NaN) marks the eligible bars with a valid outcome, for the training row filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Framework.RowFilter import row_filter_mask


SELL, HOLD, BUY = 0, 1, 2  # TradingAction's values (Framework/BaseStrategy.py)
KEEP_COL = "%wr_keep"
REQUIRED = ("target", "stop", "horizon", "reference", "eligible", "exit_eligible", "exit_loss",
            "exit_take_profit", "buckets", "share", "min_bucket")


def validate(spec) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("POOLED_LABEL must be a dict")
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError(f"POOLED_LABEL missing {missing}")
    if not spec["target"] > 0 > spec["stop"]:
        raise ValueError("POOLED_LABEL needs target > 0 > stop")
    if int(spec["horizon"]) < 1 or int(spec["buckets"]) < 2 or int(spec["min_bucket"]) < 1:
        raise ValueError("POOLED_LABEL horizon >= 1, buckets >= 2, min_bucket >= 1")
    if not 0 < float(spec["share"]) < 1:
        raise ValueError("POOLED_LABEL share must be in (0, 1)")
    if spec["exit_eligible"] is not None and not (spec["exit_loss"] > 0 and
                                                  spec["exit_take_profit"] > 0):
        raise ValueError("POOLED_LABEL exit_loss and exit_take_profit must be > 0")
    if not isinstance(spec["reference"], str):
        raise ValueError("POOLED_LABEL reference must be a column name")
    return spec


def goodness_tiebreak(outcome, bars, top, bottom, side: str) -> np.ndarray:
    """Secondary key, higher = better, among equal outcomes. Buy: a target hit sooner is better,
    a stop hit later is better. Sell: a fall reached sooner is worse, a take-profit reached later
    ranks higher (spec US1 scenario 3b). Other outcomes: 0."""
    o, b = np.asarray(outcome, float), np.asarray(bars, float)
    s2 = np.zeros(len(o))
    if side == "buy":
        s2[o == top] = -b[o == top]
        s2[o == bottom] = b[o == bottom]
    else:
        s2[o == top] = b[o == top]
        s2[o == bottom] = b[o == bottom]
    return s2


def _edges(ref: np.ndarray, buckets: int, min_bucket: int) -> tuple[np.ndarray, int]:
    """Inner edges of equal-count buckets over `ref`, merging any bucket under `min_bucket` into
    its smaller neighbour. Returns (inner edges, merges)."""
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, buckets + 1))[1:-1])
    merges = 0
    while len(inner):
        counts = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1)
        small = int(np.argmin(counts))
        if counts[small] >= min_bucket:
            break
        if small == 0:
            drop = 0
        elif small == len(counts) - 1:
            drop = small - 1
        else:
            drop = small - 1 if counts[small - 1] <= counts[small + 1] else small
        inner = np.delete(inner, drop)
        merges += 1
    return inner, merges


def assign(o, s2, ref, pos, fit, share: float, buckets: int, min_bucket: int, side: str):
    """Pure label assignment for ONE side over pooled eligible valid bars.

    o, s2: goodness keys (higher better); ref: reference values; pos: a stable position rank
    (lower = earlier); fit: bool, the rows the buckets and cuts are fitted on. Returns
    (selected: bool array, bucket report list). Buy side selects the top `share` of each bucket,
    sell side the bottom `share`; the cut fitted on `fit` rows is applied to the rest.
    """
    o, s2, ref, pos, fit = (np.asarray(x) for x in (o, s2, ref, pos, fit))
    if not fit.any():
        raise ValueError(f"no eligible valid {side} bars in the fit rows")
    inner, merges = _edges(ref[fit], buckets, min_bucket)
    b = np.searchsorted(inner, ref, side="right")
    sel = np.zeros(len(o), dtype=bool)
    report = []
    lo_edges = np.r_[-np.inf, inner]
    hi_edges = np.r_[inner, np.inf]
    for k in range(len(inner) + 1):
        idx_f = np.flatnonzero((b == k) & fit)
        idx_v = np.flatnonzero((b == k) & ~fit)
        n = len(idx_f)
        if n == 0:
            continue
        # order fit rows best-first: goodness desc, then position asc
        order = idx_f[np.lexsort((pos[idx_f], -s2[idx_f], -o[idx_f]))]
        m = int(round(share * n))
        if side == "buy":
            chosen = order[:m]
            cut = (o[order[m - 1]], s2[order[m - 1]]) if m else (np.inf, np.inf)
            val = (o[idx_v] > cut[0]) | ((o[idx_v] == cut[0]) & (s2[idx_v] >= cut[1]))
        else:
            chosen = order[n - m:]
            cut = (o[order[n - m]], s2[order[n - m]]) if m else (-np.inf, -np.inf)
            val = (o[idx_v] < cut[0]) | ((o[idx_v] == cut[0]) & (s2[idx_v] <= cut[1]))
        sel[chosen] = True
        sel[idx_v[val]] = True
        report.append({"side": side, "lo": float(lo_edges[k]), "hi": float(hi_edges[k]), "n": n,
                       "pos_share": round(m / n, 4),
                       "median_outcome": float(np.median(o[idx_f])),
                       "cut": [float(cut[0]), float(cut[1])],
                       "val_n": int(len(idx_v)),
                       "val_pos_share": round(float(val.mean()), 4) if len(idx_v) else None,
                       "merged": merges > 0})
    return sel, report


def _spearman(x, y) -> float | None:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return round(float(spearmanr(x, y).statistic), 4)


def relabel(frames: list, labels: list, spec: dict, fit_ends: list | None = None):
    """Return (labels_out, report); writes KEEP_COL into each frame (the only mutation)."""
    from Framework.TrainingSignals import path_outcome

    validate(spec)
    if len(frames) != len(labels):
        raise ValueError("frames and labels differ in length")
    fit_ends = [len(f) for f in frames] if fit_ends is None else list(fit_ends)
    H = int(spec["horizon"])
    sides = {"buy": (spec["eligible"], spec["target"], spec["stop"])}
    if spec["exit_eligible"] is not None:
        sides["sell"] = (spec["exit_eligible"], -spec["exit_loss"], spec["exit_take_profit"])
    pooled = {s: {"o": [], "s2": [], "ref": [], "fit": [], "where": []} for s in sides}
    keep, eligible_any = [], 0
    for i, df in enumerate(frames):
        if spec["reference"] not in df.columns:
            raise KeyError(spec["reference"])
        ref = df[spec["reference"]].to_numpy(float)
        fit_row = np.arange(len(df)) < fit_ends[i]
        k = np.zeros(len(df), dtype=bool)
        masks = {}
        for s, (rule, top, bottom) in sides.items():
            elig = row_filter_mask(df, rule)
            o, bars = path_outcome(df, top, bottom, H, side=s)
            ok = elig & np.isfinite(o) & np.isfinite(ref)
            masks[s] = elig
            rows = np.flatnonzero(ok)
            P = pooled[s]
            P["o"].append(o[rows])
            P["s2"].append(goodness_tiebreak(o[rows], bars[rows], top, bottom, s))
            P["ref"].append(ref[rows])
            P["fit"].append(fit_row[rows])
            P["where"].append(np.c_[np.full(len(rows), i), rows])
            k |= ok
        if "sell" in masks and np.any(masks["buy"] & masks["sell"]):
            raise ValueError("entry and exit eligibility overlap")
        eligible_any += int(np.logical_or.reduce(list(masks.values())).sum())
        keep.append(k)

    out = [np.asarray(lab).copy() for lab in labels]
    report = {"spec": spec, "buckets": [], "pairs": len(frames)}
    for s in sides:
        P = {k: np.concatenate(v) if v else np.empty(0) for k, v in pooled[s].items()}
        where = P["where"].reshape(-1, 2).astype(int)
        pos = np.lexsort((where[:, 1], where[:, 0])).argsort()
        sel, brep = assign(P["o"], P["s2"], P["ref"], pos, P["fit"].astype(bool),
                           float(spec["share"]), int(spec["buckets"]), int(spec["min_bucket"]), s)
        report["buckets"] += brep
        cls = (BUY if s == "buy" else SELL)
        for (i, r), chosen in zip(where, sel):
            out[i][r] = cls if chosen else HOLD
        f = P["fit"].astype(bool)
        report[f"spearman_{s}"] = _spearman(sel[f].astype(float), P["ref"][f])
        report[f"eligible_valid_{s}"] = int(len(sel))
        report[f"labelled_{s}"] = int(sel.sum())
    for df, k in zip(frames, keep):
        df[KEEP_COL] = np.where(k, 1.0, np.nan)
    report["kept"] = int(sum(k.sum() for k in keep))
    report["invalid"] = eligible_any - report["kept"]  # eligible bars with no valid outcome
    return out, report

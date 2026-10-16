"""037 US3/US4 pairwise AND screen (research R9, FR-016, FR-019).

Every unordered pair of raw signals that must BOTH fire: raw buys for every target except
label:sell, which uses raw sells. Lift per window = P(target | both fire) / P(target).
Populations: candidate bars for the labels; all bars (primary) and candidate bars (secondary)
for regime targets.

--select: early windows + E0 only. Pre-registered rule: computable in >= min_windows early windows;
median early lift >= uplift x max(component median early lifts) and >= min_lift; top_n by median
early lift. --confirm (needs the lock): survivors scored once on the late windows. A survivor is
"yes" if its late median lift >= min_lift and above both components' late medians AND its E0 lift
>= max(component E0 lifts) (Principle II); "early+late only" if E0 fails or is not computable;
otherwise "no".

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.screen_pairs --target label:buy --select
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.indicator_screen import harvest as H
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel, scored_columns
from research.indicator_screen.screen_single import HERE, LabelTargets, prereg, rows

LABELS = ("label:buy", "label:sell")


def populations(target: str) -> tuple[str, ...]:
    return ("candidate",) if target in LABELS else ("all", "candidate")


def signal_columns(target: str, cols: list[str]) -> list[str]:
    side = "raw_sell:" if target == "label:sell" else "raw_buy:"
    return sorted(c for c in cols if c.startswith(side))


def out_path(target: str, out_dir: Path = HERE) -> Path:
    return out_dir / f"pairs_{target.replace(':', '_')}.json"


def window_lifts(S: np.ndarray, y: np.ndarray, min_fires: int, max_rate: float):
    """(single lifts [k], pair lifts [k, k]) for 0/1 signal matrix S [n, k] and target y [n] (NaN
    rows dropped). NaN where not computable."""
    ok = np.isfinite(y)
    S = S[ok].astype(np.float64)
    y = (y[ok] > 0.5).astype(np.float64)
    n = len(y)
    if n == 0 or y.sum() == 0:
        k = S.shape[1]
        return np.full(k, np.nan), np.full((k, k), np.nan)
    base = y.mean()
    fire = S.T @ S                      # diagonal: single fires; off-diagonal: both fire
    pos = S.T @ (S * y[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (pos / fire) / base
    bad = (fire < min_fires) | (fire / n > max_rate)
    lift[bad] = np.nan
    return np.diag(lift).copy(), lift


def _median(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def score(panel, targets, target, pop, wins, sig_cols, pr):
    """{window_id: (singles, pairs)} for the given windows."""
    data = panel.columns(sig_cols)
    out = {}
    for w in wins:
        idx = rows(panel, w, pop)
        S = np.column_stack([np.nan_to_num(data[c][idx], nan=0.0) for c in sig_cols])
        out[w.id] = window_lifts(S, targets.y(target, w)[idx], pr["min_fires"], pr["max_fire_rate"])
    return out


def select(panel, targets, target, wins_fn, cols, pr) -> dict:
    sig = signal_columns(target, cols)
    rule = pr["pairs"]
    early = wins_fn(("early",))
    e0 = wins_fn(("e0",))
    res = {"target": target, "signals": len(sig), "rule": rule, "windows_read": [], "populations": {}}
    iu = np.triu_indices(len(sig), 1)
    for pop in populations(target):
        sc = score(panel, targets, target, pop, early + e0, sig, pr)
        res["windows_read"] = sorted(sc)
        single = np.array([sc[w.id][0] for w in early])          # [windows, k]
        pair = np.array([sc[w.id][1][iu] for w in early])        # [windows, pairs]
        n_ok = np.isfinite(pair).sum(0)
        with np.errstate(all="ignore"):
            med_pair = np.nanmedian(np.where(np.isfinite(pair), pair, np.nan), axis=0)
            med_single = np.nanmedian(single, axis=0)
        comp = np.fmax(med_single[iu[0]], med_single[iu[1]])
        keep = (n_ok >= rule["min_windows"]) & (med_pair >= rule["uplift"] * comp) & \
            (med_pair >= rule["min_lift"])
        order = [i for i in np.argsort(-np.nan_to_num(med_pair, nan=-1)) if keep[i]][: rule["top_n"]]
        e0s, e0p = (sc[e0[0].id] if e0 else (None, None))
        surv = []
        for i in order:
            a, b = iu[0][i], iu[1][i]
            surv.append({"a": sig[a], "b": sig[b], "side": "sell" if target == "label:sell" else "buy",
                         "target": target, "population": pop,
                         "early": {"median_lift": float(med_pair[i]), "n_windows": int(n_ok[i])},
                         "lift_a_early": _f(med_single[a]), "lift_b_early": _f(med_single[b]),
                         "selected": True,
                         "e0_lift": _f(e0p[a, b]) if e0p is not None else None,
                         "e0_lift_a": _f(e0s[a]) if e0s is not None else None,
                         "e0_lift_b": _f(e0s[b]) if e0s is not None else None})
        res["populations"][pop] = {"n_pairs": int(len(iu[0])), "n_passing_rule": int(keep.sum()),
                                   "survivors": surv}
    return res


def confirm(panel, targets, sel: dict, wins_fn, pr) -> dict:
    late = wins_fn(("late",))
    W.require_lock_for(late)
    rule = pr["pairs"]
    for pop, block in sel["populations"].items():
        surv = block["survivors"]
        if not surv:
            continue
        sig = sorted({s["a"] for s in surv} | {s["b"] for s in surv})
        pos = {c: i for i, c in enumerate(sig)}
        sc = score(panel, targets, sel["target"], pop, late, sig, pr)
        for s in surv:
            a, b = pos[s["a"]], pos[s["b"]]
            lp = _median([sc[w.id][1][a, b] for w in late])
            la = _median([sc[w.id][0][a] for w in late])
            lb = _median([sc[w.id][0][b] for w in late])
            n = int(np.isfinite([sc[w.id][1][a, b] for w in late]).sum())
            s["late"] = {"median_lift": lp, "n_windows": n, "lift_a": la, "lift_b": lb}
            late_ok = lp is not None and lp >= rule["min_lift"] and lp > (la or 0) and lp > (lb or 0)
            comp_e0 = max(v for v in (s["e0_lift_a"], s["e0_lift_b"], 0.0) if v is not None)
            e0_ok = s["e0_lift"] is not None and s["e0_lift"] >= comp_e0
            s["confirmed"] = "no" if not late_ok else ("yes" if e0_ok else "early+late only")
    sel["windows_read"] = sorted(set(sel["windows_read"]) | {w.id for w in late})
    sel["confirmed_at_lock"] = W.late_lock() if late else None
    return sel


def _f(v):
    return float(v) if v is not None and np.isfinite(v) else None


def provider(panel, target):
    if target in LABELS:
        return LabelTargets(panel)
    from research.indicator_screen.targets import RegimeTargets

    return RegimeTargets(panel, only=[target])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--confirm", action="store_true")
    a = ap.parse_args(argv)
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    pr = prereg()
    panel = Panel()
    t = provider(panel, a.target)
    path = out_path(a.target)
    try:
        if a.select:
            res = select(panel, t, a.target, W.all_windows, scored_columns(), pr)
        else:
            res = confirm(panel, t, json.loads(path.read_text()), W.all_windows, pr)
    except W.LateLockError as e:
        print(f"REFUSED: {e}")
        return 1
    path.write_text(json.dumps(res, indent=1))
    for pop, b in res["populations"].items():
        print(f"{a.target} [{pop}]: {b['n_passing_rule']} pass the rule, {len(b['survivors'])} survivors"
              + (f", confirmed {sum(s.get('confirmed') == 'yes' for s in b['survivors'])}" if a.confirm else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

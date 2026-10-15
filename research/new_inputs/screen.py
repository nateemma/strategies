"""035 T019: S1 -- does an input rank the early width-book trades beyond bb_width and the
control's p_buy? No training, no backtest, early half only (FR-008..FR-011; research R8).

Per input, on the 412 early trades (research.new_inputs.trades, which first reproduces the trade
count and the control-at-share gap):
  coverage        share of trades with a defined value at the signal bar
  rho_width       Spearman(input, bb_width)        rho_pbuy  Spearman(input, control p_buy)
  partial         Spearman partial correlation with trade pp given bb_width and p_buy: ranks,
                  residuals of rank(input) and rank(pp) on [1, rank(bb_width), rank(p_buy)],
                  Pearson of the residuals -- pooled, and per fold with ranks within the fold
  sign_folds      folds whose partial has the pooled sign
  split_gap       pp/trade above minus below the input's per-fold median (reported)

Refuses unless the pre-registration is committed and clean.

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.screen
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.new_inputs import inputs as I  # noqa: E402


HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration.json"
CACHE = STRAT.parent / "data" / "new_inputs"


def partial_spearman(x, y, controls) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5:
        return None
    X = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    cx = rx - rx.mean()
    if ex @ ex <= 1e-12 * (cx @ cx):
        return 0.0  # the input is (rank-)collinear with the controls: it adds nothing
    den = np.sqrt((ex @ ex) * (ey @ ey))
    return None if den == 0 else float(ex @ ey / den)


def _sp(a, b) -> float | None:
    r = spearmanr(a, b).statistic
    return None if r is None or np.isnan(r) else float(r)


def score(t: pd.DataFrame, col: str) -> dict:
    ok = t[col].notna().to_numpy()
    s = t[ok]
    out = {"trades": len(t), "covered": int(ok.sum()), "coverage": round(float(ok.mean()), 4)}
    if len(s) < 20:
        return out | {"partial": None, "per_fold": {}, "sign_folds": 0}
    out["rho_width"] = _sp(s[col], s["bb_width"])
    out["rho_pbuy"] = _sp(s[col], s["p_ctl"])
    out["partial"] = partial_spearman(s[col], s["pp"], [s["bb_width"], s["p_ctl"]])
    per = {}
    for f, g in s.groupby("fold"):
        per[f] = partial_spearman(g[col], g["pp"], [g["bb_width"], g["p_ctl"]])
    out["per_fold"] = per
    sign = np.sign(out["partial"] or 0.0)
    out["sign_folds"] = int(sum(1 for v in per.values() if v is not None and np.sign(v) == sign
                                and sign != 0))
    hi = s.groupby("fold")[col].transform("median") < s[col]
    out["split_gap"] = round(float(s.loc[hi, "pp"].mean() - s.loc[~hi, "pp"].mean()), 4) \
        if hi.any() and (~hi).any() else None
    return out


def attach(t: pd.DataFrame, cache_dir: Path = CACHE) -> pd.DataFrame:
    """Every ni_* value at each trade's signal bar, from the per-pair cache."""
    parts = []
    for pair, g in t.groupby("pair"):
        c = pd.read_parquet(cache_dir / f"{I.file_stem(pair)}.parquet")
        c["date"] = pd.to_datetime(c["date"], utc=True)
        m = g.merge(c, left_on="signal_date", right_on="date", how="left").drop(columns="date")
        m.index = g.index
        parts.append(m)
    return pd.concat(parts).loc[t.index]


def _committed_and_clean(path: Path) -> bool:
    rel = path.relative_to(STRAT)
    st = subprocess.run(["git", "-C", str(STRAT), "status", "--porcelain", "--", str(rel)],
                        capture_output=True, text=True, check=False).stdout.strip()
    log = subprocess.run(["git", "-C", str(STRAT), "log", "-1", "--format=%h", "--", str(rel)],
                         capture_output=True, text=True, check=False).stdout.strip()
    return bool(log) and not st


def main(argv=None) -> int:
    from research.new_inputs import trades as TR

    if not _committed_and_clean(PREREG):
        print(f"REFUSED: {PREREG.relative_to(STRAT)} is uncommitted or dirty")
        return 2
    pr = json.loads(PREREG.read_text())
    if pr["table_sha"] != I.TABLE_SHA:
        print(f"REFUSED: pre-registered table {pr['table_sha']} != current {I.TABLE_SHA}")
        return 2
    meta = json.loads((CACHE / "build_meta.json").read_text())
    if meta["table_sha"] != I.TABLE_SHA:
        print(f"REFUSED: cache built from table {meta['table_sha']}")
        return 2
    t = TR.trades()
    print("instrument:", t.attrs["check"], flush=True)
    t = attach(t)
    inconclusive = set(pr.get("s0_inconclusive", {}))
    out = {"label": "035 S1 (pre-registered; early half only, no training)",
           "instrument": t.attrs.get("check") or TR.check(t), "inputs": {}}
    for inp in I.TABLE:
        if inp.tag in inconclusive:
            out["inputs"][inp.tag] = {"status": "INCONCLUSIVE",
                                      "reason": pr["s0_inconclusive"][inp.tag]}
            continue
        r = score(t, I.column(inp.tag))
        out["inputs"][inp.tag] = r
        print(inp.tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in r.items() if k != "per_fold"}), flush=True)
    (HERE / "screen.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

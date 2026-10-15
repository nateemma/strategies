"""035 S1b: the S1 screen on a larger sample -- every gbb candidate run start, early half only
(follow-up 1 of report.md; pre-registered in preregistration_s1b.json before any score).

Sample: the FIRST bar of each run of the width book's gates WITHOUT the width cut (gbb rule and
the control's guards, research/model_calibration/width_twin._gates) in folds f00-f07, that is
tradeable (quote volume >= the simulator's MIN_QV) and whose entry bar lies in the fold's test
window. Run starts, not every bar, limit overlap between neighbouring trades.

Outcome: the net profit ratio of a trade entered at the next bar's open and exited by the 24 h
grace control's exits (research/exit_coupling/paths.replay, as the width-book simulator), free of
occupancy, wallet and max-open limits -- the trade this bar would have produced on its own.

Statistic, per input: partial Spearman with the outcome given bb_width and the control's p_buy
(screen.partial_spearman), pooled and per fold; sign folds; Spearman with bb_width and p_buy.

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.screen_s1b
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research.new_inputs import inputs as I  # noqa: E402
from research.new_inputs import screen as SC  # noqa: E402


HERE = Path(__file__).resolve().parent
PREREG = HERE / "preregistration_s1b.json"
PREREGS = {"ARO": PREREG, "GBR": HERE / "preregistration_gbr.json"}
SUFFIX = {"ARO": "", "GBR": "_gbr"}
EARLY = tuple(f"f{i:02d}" for i in range(8))


def run_starts(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return m & ~np.r_[False, m[:-1]]


def sample(folds=EARLY, model: str = "ARO") -> pd.DataFrame:
    """Candidate run starts with their replayed outcome, bb_width and control p_buy."""
    from research.exit_coupling import feasibility as F
    from research.exit_coupling import paths as P
    from research.model_calibration.width_twin import _gates
    from research.nnnc_framework import entries as E

    base = E.Model(model)
    base.params = dataclasses.replace(base.params, stoploss_grace_hours=24.0)
    rows = []
    for fold in folds:
        if fold not in EARLY:
            raise RuntimeError(f"S1b reads the early half only; got {fold}")
        t = base.tables[fold].reset_index(drop=True)
        f = base.folds[fold]
        g = np.asarray(_gates(t, E.fold_params(base.prefix, fold)), dtype=bool)
        for pair, idx in t.groupby("pair").groups.items():
            idx = np.asarray(sorted(idx))
            rs = idx[run_starts(g[idx])]
            if not len(rs):
                continue
            b, dates = F.bars_for(pair)
            start = int(dates.searchsorted(pd.Timestamp(f["test_start"], tz="UTC")))
            end = int(dates.searchsorted(pd.Timestamp(f["test_end"], tz="UTC"))) - 1
            di = pd.DatetimeIndex(t.loc[rs, "date"])
            di = di.tz_localize("UTC") if di.tz is None else di
            sig = dates.get_indexer(di)
            ent = sig[sig >= 0] + 1
            suppressed = frozenset(int(e) for e in ent)
            tick = F._tick(pair, fold, b, start, end)
            vol = E._vol(pair)
            for r, s in zip(rs[sig >= 0], sig[sig >= 0], strict=True):
                e = int(s) + 1
                if e > end:
                    continue
                if float(b.close[e - 1]) * float(vol[e - 1]) < E.MIN_QV:
                    continue
                x = P.replay(b, e, float(b.open[e]), base.params, roi=base.roi,
                             roi_suppressed=suppressed, end=end, tick=tick)
                rows.append({"fold": fold, "pair": pair, "signal_date": dates[s],
                             "outcome": float(x.profit),
                             "bb_width": float(t.at[r, "bb_width"]),
                             "p_ctl": float(t.at[r, "p_buy"])})
    out = pd.DataFrame(rows)
    out["signal_date"] = pd.to_datetime(out["signal_date"], utc=True)
    return out


def score(s: pd.DataFrame, col: str) -> dict:
    ok = s[col].notna().to_numpy()
    d = s[ok]
    out = {"n": len(s), "covered": int(ok.sum()), "coverage": round(float(ok.mean()), 4)}
    if len(d) < 100:
        return out | {"partial": None, "per_fold": {}, "sign_folds": 0}
    out["rho_width"] = SC._sp(d[col], d["bb_width"])
    out["rho_pbuy"] = SC._sp(d[col], d["p_ctl"])
    out["partial"] = SC.partial_spearman(d[col], d["outcome"], [d["bb_width"], d["p_ctl"]])
    per = {f: SC.partial_spearman(g[col], g["outcome"], [g["bb_width"], g["p_ctl"]])
           for f, g in d.groupby("fold")}
    out["per_fold"] = per
    sign = np.sign(out["partial"] or 0.0)
    out["sign_folds"] = int(sum(1 for v in per.values()
                                if v is not None and sign != 0 and np.sign(v) == sign))
    return out


def main(argv=None) -> int:
    import argparse

    from research.new_inputs import survivors as SV

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PREREGS), default="ARO")
    model = ap.parse_args(argv).model
    prereg = PREREGS[model]
    if not SC._committed_and_clean(prereg):
        print(f"REFUSED: {prereg.relative_to(STRAT)} is uncommitted or dirty")
        return 2
    pr = json.loads(prereg.read_text())
    pr = pr.get("s1b", pr)  # the GBR pre-registration nests its S1b block
    if pr["table_sha"] != I.TABLE_SHA:
        print(f"REFUSED: pre-registered table {pr['table_sha']} != current {I.TABLE_SHA}")
        return 2
    s = sample(model=model)
    print("sample:", len(s), "run starts;", s.groupby("fold").size().to_dict(), flush=True)
    s = SC.attach(s)
    out = {"label": f"035 S1b {model} (pre-registered; early half only, no training)",
           "sample": {"n": len(s), "per_fold": s.groupby("fold").size().to_dict(),
                      "outcome_mean": round(float(s["outcome"].mean()), 5)},
           "inputs": {}}
    for inp in I.TABLE:
        r = score(s, I.column(inp.tag))
        out["inputs"][inp.tag] = r
        print(inp.tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in r.items() if k != "per_fold"}), flush=True)
    (HERE / f"screen_s1b{SUFFIX[model]}.json").write_text(json.dumps(out, indent=1, default=str))
    res = SV.select(out, pr["rule"])
    res["label"] = f"035 survivors (S1b, {model})"
    (HERE / f"survivors_s1b{SUFFIX[model]}.json").write_text(json.dumps(res, indent=1))
    print("S1b survivors:", res["survivors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

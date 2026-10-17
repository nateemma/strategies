"""Mixed sizer study: the per-fold percentile maps behind `cs_mix` (preregistration.json). ONE fold per process.

Score = mean of two percentiles: ARO's p_buy and the 037 'dip' composite (research/cmp_sizer), each
ranked against the width-book run starts (first bar of each entry run) in the fold's VALIDATION TAIL --
the rows cs_aro's threshold rule already reads; nothing from the test window. The tail frames come
from the cs_cmp fold strategy (same entries, ARO weights and composite table as cs_mix).

Output, user_data/data/mix_sizer/ (not committed; sha256 in the plan's threshold_picks):
- MIX_fNN.json: sorted run-start values {"aro": [...], "cmp": [...]} (the strategy reads it);
- HGB_fNN.json: meta for research/hgb_sizer/pick.py (fit_last empty: nothing is fitted);
- signals_fNN.parquet: the control's ARO_fNN table with p_buy replaced by the mixed score.

    PYTHONPATH=. ../../.venv/bin/python -m research.mix_sizer.knots --fold f00
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.cmp_sizer.signals import OUT as CMP  # noqa: E402
from research.hgb_sizer.signals import ARO_SIGNALS  # noqa: E402

OUT = STRAT.parents[1] / "user_data" / "data" / "mix_sizer"
CMP_PLAN = STRAT / "evaluation" / "folds" / "plan_cs_cmp.json"


def pct(x, knots) -> np.ndarray:
    """Share of knots <= x (NaN stays NaN)."""
    k = np.asarray(knots, float)
    x = np.asarray(x, float)
    out = np.searchsorted(k, x, side="right") / len(k)
    return np.where(np.isfinite(x), out, np.nan)


def mixed(aro, cmp, knots: dict) -> np.ndarray:
    return 0.5 * (pct(aro, knots["aro"]) + pct(cmp, knots["cmp"]))


def run(fid: str) -> dict:
    from research.width_relative import pick as WP

    frames, _ = WP._frames(json.loads(CMP_PLAN.read_text()), fid, None)
    a, c = [], []
    for s in frames.values():
        rs = WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)
        a.append(s["%predict_prob_buy"].to_numpy(float)[rs])
        c.append(s["sizer_score"].to_numpy(float)[rs])
    a, c = np.concatenate(a), np.concatenate(c)
    knots = {"aro": np.sort(a[np.isfinite(a)]).tolist(), "cmp": np.sort(c[np.isfinite(c)]).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"MIX_{fid}.json").write_text(json.dumps(knots))
    sha = hashlib.sha256((OUT / f"MIX_{fid}.json").read_bytes()).hexdigest()

    aro = pd.read_parquet(ARO_SIGNALS / f"ARO_{fid}.parquet")
    d = pd.to_datetime(aro["date"])
    aro["date"] = d.dt.tz_localize("UTC") if d.dt.tz is None else d.dt.tz_convert("UTC")
    tab = pd.read_parquet(CMP / f"HGB_{fid}.parquet").rename(columns={"p_buy": "cmp"})
    tab["date"] = pd.to_datetime(tab["date"], utc=True)
    sig = aro.merge(tab, on=["date", "pair"], how="left")
    sig["p_buy"] = mixed(sig["p_buy"], sig["cmp"], knots)
    sig.drop(columns="cmp").to_parquet(OUT / f"signals_{fid}.parquet", index=False)
    meta = {"fold": fid, "run_starts": int(len(a)), "aro_missing": int((~np.isfinite(a)).sum()),
            "cmp_missing": int((~np.isfinite(c)).sum()), "fit_last": {}, "sha256": sha}
    (OUT / f"HGB_{fid}.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    a = ap.parse_args(argv)
    m = run(a.fold)
    print(f"{a.fold}: knots on {m['run_starts']} tail run starts (missing aro {m['aro_missing']}, cmp {m['cmp_missing']})")
    return 0 if m["run_starts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

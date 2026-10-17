"""Cross-model sizer study: per-fold percentile knots behind `cs_<m>_mix` (preregistration.json). ONE fold per process.

As research/mix_sizer/knots.py, for GBR / MLX: the model's p_buy and the 037 composite, each ranked
against the width-book run starts in the fold's VALIDATION TAIL, read from the plain sizer's fold
strategy (`cs_<m>`: same entries and weights) with the composite looked up in the model-free
user_data/data/cmp_sizer tables.

Output, user_data/data/mix_sizer_<m>/:
- MIX_fNN.json: sorted run-start values {"model": [...], "cmp": [...]} (the strategy reads it);
- HGB_fNN.json: meta for research/hgb_sizer/pick.py (fit_last empty: nothing is fitted);
- signals_fNN.parquet: the model's own test-window table (nnnc_framework/signals/CS_<M>_fNN) with
  p_buy replaced by the mixed score, for the instrument.

    PYTHONPATH=. ../../.venv/bin/python -m research.xmodel_sizer.knots --model MLX --fold f00
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
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "NNNC")]

from research.cmp_sizer.signals import OUT as CMP  # noqa: E402

DATA = STRAT.parents[1] / "user_data" / "data"
SIG = STRAT / "research" / "nnnc_framework" / "signals"
FOLDS = STRAT / "evaluation" / "folds"


def out_dir(model: str) -> Path:
    return DATA / f"mix_sizer_{model.lower()}"


def _cmp_lookup(fid: str) -> pd.Series:
    t = pd.read_parquet(CMP / f"HGB_{fid}.parquet")
    t["date"] = pd.to_datetime(t["date"], utc=True)
    return t.set_index(["pair", "date"])["p_buy"].sort_index()


def run(model: str, fid: str) -> dict:
    from _mix_sizer import pct
    from research.width_relative import pick as WP

    m = model.lower()
    frames, _ = WP._frames(json.loads((FOLDS / f"plan_cs_{m}.json").read_text()), fid, None)
    cl = _cmp_lookup(fid)
    a, c = [], []
    for pair, s in frames.items():
        rs = WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)
        a.append(s["%predict_prob_buy"].to_numpy(float)[rs])
        ser = cl.loc[pair] if pair in cl.index.get_level_values(0) else pd.Series(dtype=float)
        c.append(pd.to_datetime(s["date"], utc=True).map(ser).to_numpy(float)[rs])
    a, c = np.concatenate(a), np.concatenate(c)
    knots = {"model": np.sort(a[np.isfinite(a)]).tolist(), "cmp": np.sort(c[np.isfinite(c)]).tolist()}
    out = out_dir(model)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"MIX_{fid}.json").write_text(json.dumps(knots))
    sha = hashlib.sha256((out / f"MIX_{fid}.json").read_bytes()).hexdigest()

    sig = pd.read_parquet(SIG / f"CS_{model.upper()}_{fid}.parquet")
    d = pd.to_datetime(sig["date"])
    sig["date"] = d.dt.tz_localize("UTC") if d.dt.tz is None else d.dt.tz_convert("UTC")
    key = pd.MultiIndex.from_arrays([sig["pair"], sig["date"]])
    comp = cl.reindex(key).to_numpy(float)
    sig["p_buy"] = 0.5 * (pct(sig["p_buy"].to_numpy(float), knots["model"]) + pct(comp, knots["cmp"]))
    sig.to_parquet(out / f"signals_{fid}.parquet", index=False)
    meta = {"fold": fid, "model": model, "run_starts": int(len(a)), "model_missing": int((~np.isfinite(a)).sum()),
            "cmp_missing": int((~np.isfinite(c)).sum()), "fit_last": {}, "sha256": sha}
    (out / f"HGB_{fid}.json").write_text(json.dumps(meta, indent=1))
    return meta


def link_plain(model: str) -> Path:
    """user_data/data/xsizer_<m>/signals_fNN.parquet -> nnnc_framework/signals/CS_<M>_fNN.parquet
    (the plain sizer's instrument reads its own model's table)."""
    out = DATA / f"xsizer_{model.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    for src in sorted(SIG.glob(f"CS_{model.upper()}_f??.parquet")):
        dst = out / f"signals_{src.stem.rsplit('_', 1)[1]}.parquet"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("GBR", "MLX"))
    ap.add_argument("--fold", required=True)
    a = ap.parse_args(argv)
    m = run(a.model, a.fold)
    print(f"{a.model} {a.fold}: knots on {m['run_starts']} tail run starts "
          f"(missing model {m['model_missing']}, cmp {m['cmp_missing']})")
    return 0 if m["run_starts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

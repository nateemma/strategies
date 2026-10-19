"""038 instrument checks (SC-001, SC-002a/b, SC-003, R12). Nothing after a failed check is read.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.instrument
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from research.forecast import common as C
from research.forecast import frames as F
from research.forecast import score as S

PATH = C.HERE / "instrument.json"


def sc001() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings",
                        "utils/tests/test_forecaster_horizons.py"], cwd=C.STRAT, capture_output=True, text=True)
    return {"pass": r.returncode == 0, "detail": r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:]}


def sc002a(frames: dict) -> dict:
    """Zero forecast: r2_oos == 0 exactly, on every horizon."""
    z = S.load_pred("zero")
    vals = {}
    for n in C.ALL_HORIZONS:
        f, r = [], []
        for pair, fr in frames.items():
            p = z[z["pair"] == pair]
            t = fr.set_index("date")[f"target_{n}"].reindex(p["date"]).to_numpy()
            ok = np.isfinite(t)
            f.append(p[f"f_{n}"].to_numpy()[ok])
            r.append(t[ok])
        vals[str(n)] = S.r2_oos(np.concatenate(f), np.concatenate(r))
    return {"pass": all(v == 0.0 for v in vals.values()), "r2_oos": vals}


def sc002b() -> dict:
    """lgb_null is not PASS at any N (filled in once the null has been scored and judged)."""
    v = C.HERE / "verdicts.json"
    judged = json.loads(v.read_text()).get("cells", {}) if v.exists() else {}
    nulls = [m for m in judged if m.startswith("lgb_null")]  # lgb_null and lgb_null_long
    if not nulls:
        return {"pass": None, "detail": "lgb_null not judged yet"}
    passed = [f"{m}:{n}" for m in nulls for n, c in judged[m].items() if c["verdict"].startswith("PASS")]
    return {"pass": not passed, "null_pass_n": passed}


def sc003(frames: dict) -> dict:
    """Share of eligible scored candles with no stride-1 baseline forecast, excluding gaps."""
    z = S.load_pred("zero")
    miss = tot = 0
    for pair, fr in frames.items():
        s = fr[(fr["fold"] != "") & fr["has_data"] & fr["eligible_1"]]
        have = set(z.loc[z["pair"] == pair, "date"])
        tot += len(s)
        miss += int((~s["date"].isin(have)).sum())
    share = miss / tot if tot else 1.0
    return {"pass": share < 0.01, "unscored_share": share, "eligible": tot}


def liveness() -> dict:
    """R12: every method's forecast moves with its input (not constant across candles within a fold).
    Whether it also differs across N is reported, not required: level models (SES, ExpSmoothing, ETS,
    state-space) forecast a flat path by construction."""
    out, across_n = {}, {}
    for p in sorted(C.PRED.glob("*.feather")):
        if p.stem == "zero":
            continue
        d = pd.read_feather(p)
        f = d[[c for c in d.columns if c.startswith("f_")]].dropna()
        across_n[p.stem] = bool(len(f) and not np.allclose(f["f_1"].to_numpy()[:, None], f.to_numpy()))
        if p.stem == "hist_mean":  # constant within a fold by definition; it moves across folds
            out[p.stem] = bool(d["f_1"].nunique() > 1)
            continue
        out[p.stem] = bool(len(f) and f["f_16"].std() > 0)
    return {"pass": all(out.values()) if out else None, "methods": out, "differs_across_n": across_n}


def main(argv=None) -> int:
    cols = ["date", "fold", "has_data", "eligible_1", *[f"target_{n}" for n in C.ALL_HORIZONS]]
    frames = {p: F.read(p, cols) for p in C.pairs()}
    res = {"SC-001": sc001(), "SC-002a": sc002a(frames), "SC-002b": sc002b(), "SC-003": sc003(frames),
           "R12": liveness()}
    res["pass"] = all(v["pass"] is not False for v in res.values())
    C.write_json(PATH, res)
    for k, v in res.items():
        print(k, v if k == "pass" else json.dumps(v)[:300])
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

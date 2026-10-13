"""Why does 026's volreg reversion gate hurt NNNC when it helped SimpleTrigger? (diagnosis, no backtests)

Reads the switch experiment's records (`sw_<m>_gate`, `sw_<m>_gateoff`) and, for contrast, the
SimpleTrigger gbb arms (`st_gbb_b029` ungated; `st_book_rev2_b029` gated reversion).

  1. OVERLAP: the volreg off-state share of all bars, of the gbb trigger's rows and of each
     model's signal rows (from the gate arms' entry_condition_counts: `signal` counts rows the
     gate let through, `regime_gate_refused` the rows it dropped).
  2. BUCKETS: the ungated arm's trades split by the state at their signal bar
     (open_date - 15 min, as-of), P&L per half. 026's gate was chosen because gbb's off-state
     trades lost in both halves.
  3. DECOMPOSITION of the measured gate effect (gate arm - gateoff arm):
       removed   = - P&L of the gateoff arm's off-state trades
       displaced = P&L of the gateoff arm's ON-state trades the gate arm did not take
                   (slots and pair occupancy change once off-state trades are gone)
       added     = P&L of gate-arm trades the gateoff arm did not take
     gate - gateoff = removed - displaced + added  (exact up to rounding).

P&L unit = per-trade pp contribution (profit_abs x scale_pp_per_abs), summed. PROVISIONAL.

    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_switches.gate_overlap
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

HERE = Path(__file__).resolve().parent
THRESHOLD = 1.3658023280926226          # off when volreg < threshold (st_book_rev2's cut)
EARLY = frozenset(f"f{i:02d}" for i in range(8))
FOLDS = [f"f{i:02d}" for i in range(15)]
MODELS = {"ARO": "arm D", "GBR": "GbbRaw", "GFM": "GbbFwdMae", "MLX": "NNNC_MLX"}


def _plan(tag: str) -> dict:
    return json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}.json").read_text())


def _counts(tag: str) -> dict:
    tot = {"signal": 0, "refused": 0, "off_bars": 0, "bars": 0}
    for f in FOLDS:
        ec = json.loads((STRAT / "regime" / "arms" / f"{tag}_{f}_{f.upper()}.json").read_text())[
            "entry_condition_counts"]
        tot["signal"] += ec["signal"]["pass"]
        tot["refused"] += ec["regime_gate_refused"]["pass"]
        tot["off_bars"] += ec["regime_gate_off_bars"]["pass"]
        tot["bars"] += ec["regime_gate_off_bars"]["total"]
    return tot


def _trades(tag: str, volreg: pd.Series) -> pd.DataFrame:
    from evaluation.pool import pool

    p = pool(_plan(tag))
    if p["folds_missing"]:
        raise SystemExit(f"{tag}: missing folds {p['folds_missing']}")
    tr = pd.DataFrame(p["trades"])
    sig = pd.to_datetime(tr["open_date"], utc=True) - pd.Timedelta(minutes=15)
    v = volreg.reindex(pd.DatetimeIndex(sig), method="ffill").to_numpy()
    return pd.DataFrame({
        "key": tr["pair"] + "|" + tr["open_date"].astype(str),
        "pnl": (tr["profit_abs"].fillna(0) * tr["scale_pp_per_abs"].fillna(0)).to_numpy(),
        "half": np.where(tr["fold_id"].isin(EARLY), "early", "late"),
        "off": (v < THRESHOLD) & ~np.isnan(v),
    })


def _bucket(t: pd.DataFrame) -> dict:
    out = {}
    for h in ("early", "late"):
        x = t[t["half"] == h]
        for s, m in (("on", ~x["off"]), ("off", x["off"])):
            y = x[m]
            out[f"{h}_{s}"] = {"n": int(len(y)), "pp": float(y["pnl"].sum()),
                               "mean_bp": float(y["pnl"].mean() * 100) if len(y) else None}
        out[f"{h}_off_share"] = float(x["off"].mean()) if len(x) else None
    return out


def _decompose(gate: pd.DataFrame, off: pd.DataFrame) -> dict:
    out = {}
    for h in ("early", "late", "all"):
        g = gate if h == "all" else gate[gate["half"] == h]
        o = off if h == "all" else off[off["half"] == h]
        gk, ok = set(g["key"]), set(o["key"])
        removed = -float(o.loc[o["off"], "pnl"].sum())
        displaced = float(o.loc[~o["off"] & ~o["key"].isin(gk), "pnl"].sum())
        added = float(g.loc[~g["key"].isin(ok), "pnl"].sum())
        out[h] = {"gate_minus_gateoff": float(g["pnl"].sum() - o["pnl"].sum()),
                  "removed": removed, "displaced": displaced, "added": added,
                  "n_removed": int(o["off"].sum()),
                  "n_displaced": int((~o["off"] & ~o["key"].isin(gk)).sum()),
                  "n_added": int((~g["key"].isin(ok)).sum()),
                  "n_gate_off": int(g["off"].sum())}
    return out


def main() -> int:
    from research.clfstudy.data import load_pair
    from research.regime_gate.feasibility import PAIRS
    from utils import RegimeState as RS

    volreg = RS.volreg({p: load_pair(f"{p}/USDT") for p in PAIRS}).sort_index()
    res = {"threshold": THRESHOLD, "overlap": {}, "buckets": {}, "decomposition": {}}
    st = _counts("st_book_rev2_b029")
    res["overlap"]["bars"] = st["off_bars"] / st["bars"]
    res["overlap"]["gbb trigger"] = st["refused"] / (st["signal"] + st["refused"])
    for m in MODELS:
        c = _counts(f"sw_{m.lower()}_gate")
        res["overlap"][m] = c["refused"] / (c["signal"] + c["refused"])
    res["buckets"]["gbb trigger (st_gbb_b029)"] = _bucket(_trades("st_gbb_b029", volreg))
    for m in MODELS:
        off = _trades(f"sw_{m.lower()}_gateoff", volreg)
        gate = _trades(f"sw_{m.lower()}_gate", volreg)
        res["buckets"][m] = _bucket(off)
        res["decomposition"][m] = _decompose(gate, off)
    (HERE / "gate_overlap.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

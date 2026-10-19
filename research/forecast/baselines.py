"""038 baselines and the cost line per horizon (Story 2; research R8, data-model § Horizon).

- zero: f = 0 ("no change", the random walk);
- hist_mean: f[n] = n x the pair's mean 1-candle log return over all data before the fold start
  (anchored, causal, recomputed per fold; the forward window uses all data before it).
The horizon table gives, per pair and pooled, the median absolute N-candle trade return on eligible
scored candles against the round-trip cost: move_to_cost < 1 at the verdict slippage = untradeable.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.baselines
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.forecast import common as C
from research.forecast import frames as F


def scored(fr: pd.DataFrame) -> pd.DataFrame:
    """Rows a stride-1 method forecasts: in a fold or the forward window, with real data at t."""
    return fr[(fr["fold"] != "") & fr["has_data"]]


def zero_pred(pair: str, fr: pd.DataFrame) -> pd.DataFrame:
    s = scored(fr)
    out = pd.DataFrame({"pair": pair, "date": s["date"].to_numpy()})
    for n in C.HORIZONS:
        out[f"f_{n}"] = 0.0
    return out.assign(failed=False, skipped_gap=False)


def hist_mean_pred(pair: str, fr: pd.DataFrame) -> pd.DataFrame:
    r1 = fr["lc"].diff()
    real = fr["has_data"] & fr["has_data"].shift(1, fill_value=False)
    s = scored(fr)
    starts = {fid: a for fid, (a, _) in C.folds().items()} | {"fwd": C.FWD_START}
    mu = {fid: float(r1[real & (fr["date"] < a)].mean()) for fid, a in starts.items()}
    m = s["fold"].map(mu).to_numpy(float)
    out = pd.DataFrame({"pair": pair, "date": s["date"].to_numpy()})
    for n in C.HORIZONS:
        out[f"f_{n}"] = n * m
    return out.assign(failed=False, skipped_gap=False)


def horizon_table(frames: dict[str, pd.DataFrame]) -> dict:
    out = {}
    pooled = {n: [] for n in C.HORIZONS}
    for pair, fr in frames.items():
        s = fr[fr["half"].isin(["early", "late"])]
        out[pair] = {}
        for n in C.HORIZONS:
            mv = np.abs(s.loc[s[f"eligible_{n}"], f"trade_ret_{n}"].to_numpy())
            pooled[n].append(mv)
            out[pair][str(n)] = _row(mv, n)
    out["pooled"] = {str(n): _row(np.concatenate(pooled[n]), n) for n in C.HORIZONS}
    return out


def _row(mv: np.ndarray, n: int) -> dict:
    typical = float(np.median(mv)) if len(mv) else float("nan")
    row = {"typical_move": typical, "n_eligible": int(len(mv)), "hours": n / 4}
    for s in C.SLIPPAGE_BPS:
        row[f"cost_rt_{s}"] = C.cost_rt(s)
        row[f"move_to_cost_{s}"] = typical / C.cost_rt(s)
    row["untradeable"] = bool(row[f"move_to_cost_{C.VERDICT_BPS}"] < 1.0)
    return row


def main(argv=None) -> int:
    cols = ["date", "fold", "half", "has_data", "lc",
            *[c for n in C.HORIZONS for c in (f"trade_ret_{n}", f"eligible_{n}")]]
    frames = {p: F.read(p, cols) for p in C.pairs()}
    C.write_feather(pd.concat([zero_pred(p, f) for p, f in frames.items()], ignore_index=True), C.PRED / "zero.feather")
    C.write_feather(pd.concat([hist_mean_pred(p, f) for p, f in frames.items()], ignore_index=True),
                    C.PRED / "hist_mean.feather")
    table = horizon_table(frames)
    C.write_json(C.HERE / "baselines.json", {"horizons": table})
    print(f"{'N':>4s} {'hours':>6s} {'median |move|':>14s} {'cost@5bp':>9s} {'move/cost':>9s}")
    for n in C.HORIZONS:
        r = table["pooled"][str(n)]
        print(f"{n:4d} {n / 4:6.2f} {r['typical_move'] * 1e4:11.1f} bp {C.cost_rt(5.0) * 1e4:6.1f} bp "
              f"{r['move_to_cost_5.0']:9.2f}{'  UNTRADEABLE' if r['untradeable'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

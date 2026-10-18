"""ATR-barrier study, exit REPLAY (descriptive; replays choose what to test, backtests judge).

Question (user, 2026-10-17): today's exits (fixed 2.2% take-profit, -5% stop with 24 h grace)
reward volatility, which is what the width book harvests and what cs_aro's gbb_path label predicts.
Would exits that match the ATR-barrier label let the tbatr direction score (cs_tba_d) pay?

On the width book's own trades (w_aro, 15 folds; cs_aro and cs_tba_d take the same trades, their
stakes are joined by pair + open date), each trade is re-exited from its recorded entry
(open[e], signal bar e-1) under:
  null  -- today's exit (research/exit_coupling/paths.replay; fidelity vs the record reported);
  c2    -- C2 ATR take-profit 2.897 x atr_pct_roll (clamped 1-5%), today's stop + grace (028);
  bar_close_K -- the label exactly: barriers entry x (1 +- K x ATR14/close at e-1) on CLOSES,
                 exit at the next bar's open after the first close beyond a barrier; at 48 bars
                 exit at that bar's close (the label's timeout);
  bar_intra_K -- the same barriers as resting take-profit / stop orders (high / low; a bar
                 spanning both counts as the stop; a gap through fills at the open); 48-bar timeout;
  tp_grace_K  -- take-profit at +K x ATR14 (intrabar), today's stop + grace, no timeout.
K = 4 is the label's; 2 and 3 are reported beside it. Fees as ExitPolicy.net_from_price_ratio.

For each exit x sizer (equal = w_aro's stakes, cs_aro's, cs_tba_d's): sum of profit x stake x scale
(pp of the wallet) per half. Replays ignore re-sequencing and wallet effects (023 Addendum 7).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.replay
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.exit_coupling import feasibility as F
from research.exit_coupling import paths as P

HERE = Path(__file__).resolve().parent
HORIZON = 48
KS = (2.0, 3.0, 4.0)
C2_K = 2.8970711921574543
SIZERS = ("cs_aro", "cs_tba_d")
_ATR: dict = {}


def atr14(pair: str) -> np.ndarray:
    """Framework.TrainingSignals._atr / close: the label's ATR fraction."""
    if pair not in _ATR:
        b, _ = F.bars_for(pair)
        pc = np.roll(b.close, 1)
        pc[0] = b.close[0]
        tr = np.maximum(b.high - b.low, np.maximum(np.abs(b.high - pc), np.abs(b.low - pc)))
        a = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
        _ATR[pair] = np.maximum(a, 1e-12) / b.close
    return _ATR[pair]


def exit_setup_json(class_name: str):
    """feasibility.exit_setup with the fold's own params JSON (the backtest's values: grace 24 h /
    -0.27, max hold 10 d), not the class defaults it falls back to for NNNC fold classes."""
    from evaluation.fingerprint import load_strategy_class
    from utils.FamilyExits import class_level_exit_params

    cls = load_strategy_class(class_name)
    js = json.loads((F.STRAT / "NNNC" / f"{class_name}.json").read_text())["params"]
    sell = js["sell"]
    params = dataclasses.replace(class_level_exit_params(cls, sell),
                                 stoploss_grace_hours=float(sell["stoploss_grace_hours"]),
                                 stoploss_grace_level=float(sell["stoploss_grace_level"]))
    hold = getattr(cls, "HOLD_WINDOW_HOURS", None)
    if hold is not None:
        params = dataclasses.replace(params, unclog_12h_hours=max(params.unclog_12h_hours, float(hold)))
    roi = js.get("roi", {}).get("0")
    return cls, params, None if roi is None else float(roi)


F.exit_setup = exit_setup_json


def barrier_close(b, e: int, open_rate: float, frac: float, end: int, fee: float) -> tuple[float, str]:
    up, dn = open_rate * (1 + frac), open_rate * (1 - frac)
    last = min(e + HORIZON - 1, end)
    for k in range(e, last + 1):
        c = b.close[k]
        if c >= up or c <= dn:
            px = b.open[k + 1] if k + 1 <= end else c
            return P.net_from_price_ratio(px / open_rate, fee), "up" if c >= up else "down"
    return P.net_from_price_ratio(b.close[last] / open_rate, fee), "timeout"


def barrier_intra(b, e: int, open_rate: float, frac: float, end: int, fee: float) -> tuple[float, str]:
    up, dn = open_rate * (1 + frac), open_rate * (1 - frac)
    last = min(e + HORIZON - 1, end)
    for k in range(e, last + 1):
        if b.low[k] <= dn:
            px = min(b.open[k], dn)
            return P.net_from_price_ratio(px / open_rate, fee), "down"
        if b.high[k] >= up:
            px = max(b.open[k], up)
            return P.net_from_price_ratio(px / open_rate, fee), "up"
    return P.net_from_price_ratio(b.close[last] / open_rate, fee), "timeout"


def _stakes(tag: str) -> dict:
    return {(t["pair"], str(pd.Timestamp(t["open_date"]))): float(t["stake_amount"]) * t["scale"]
            for t in F.archive_trades(tag)}


def main() -> int:
    df, setup = F.trade_rows("w_aro", "NNNC_W_ARO_f00", "reversion")
    fid = F.fidelity(df)
    cls, params, roi = F.exit_setup("NNNC_W_ARO_f00")
    params = F.family_params(cls, params, "reversion")
    fee = params.fee_per_side
    w = {(r.pair, str(F.bars_for(r.pair)[1][r.e])): r.stake * r.scale for r in df.itertuples()}
    stakes = {"equal": w} | {s: _stakes(s) for s in SIZERS}
    exits = {"null": [], "c2": []} | {f"{v}_{k:g}": [] for v in ("bar_close", "bar_intra", "tp_grace") for k in KS}
    outcome = {f"bar_close_{k:g}": [] for k in KS}
    keys = []
    for r in df.itertuples():
        b, dates = F.bars_for(r.pair)
        keys.append((r.pair, str(dates[r.e])))
        exits["null"].append(r.null_profit)
        c2 = P.replay(b, r.e, r.open_rate, params, roi=roi, roi_suppressed=r.roi_suppressed, end=r.end,
                      rules=P.Rules(take_profit_atr_mult=C2_K), tick=r.tick)
        exits["c2"].append(c2.profit)
        a = float(atr14(r.pair)[r.e - 1])
        for k in KS:
            p, o = barrier_close(b, r.e, r.open_rate, k * a, r.end, fee)
            exits[f"bar_close_{k:g}"].append(p)
            outcome[f"bar_close_{k:g}"].append(o)
            exits[f"bar_intra_{k:g}"].append(barrier_intra(b, r.e, r.open_rate, k * a, r.end, fee)[0])
            pk = dataclasses.replace(params, take_profit=k * a)
            exits[f"tp_grace_{k:g}"].append(P.replay(b, r.e, r.open_rate, pk, roi=roi,
                                                     roi_suppressed=r.roi_suppressed, end=r.end,
                                                     tick=r.tick).profit)
    halves = df["half"].to_numpy()
    folds = df["fold"].to_numpy()
    out = {"label": "REPLAY (descriptive): replays choose, backtests judge", "fidelity": fid,
           "n_trades": int(len(df)), "recorded_pp": {h: float(df.loc[df.half == h, "rec_pp"].sum())
                                                       for h in ("early", "late")},
           "missing_stakes": {}, "pp": {}, "per_fold": {}, "outcomes": {}}
    for s, st in stakes.items():
        sv = np.array([st.get(k, np.nan) for k in keys])
        out["missing_stakes"][s] = int(np.isnan(sv).sum())
        for x, prof in exits.items():
            v = np.asarray(prof) * np.nan_to_num(sv)
            out["pp"][f"{x}|{s}"] = {h: round(float(v[halves == h].sum()), 2) for h in ("early", "late")}
            out["per_fold"][f"{x}|{s}"] = {f: round(float(v[folds == f].sum()), 3) for f in sorted(set(folds))}
    for x, o in outcome.items():
        o = np.asarray(o)
        out["outcomes"][x] = {h: {c: round(float((o[halves == h] == c).mean()), 3) for c in ("up", "down", "timeout")}
                              for h in ("early", "late")}
    (HERE / "replay.json").write_text(json.dumps(out, indent=1, default=str))
    print("fidelity", json.dumps(fid, default=str)[:300])
    print("recorded", out["recorded_pp"], "missing stakes", out["missing_stakes"])
    for x in exits:
        print(f"{x:14s}", " ".join(f"{s}: {out['pp'][f'{x}|{s}']['early']:7.2f} / {out['pp'][f'{x}|{s}']['late']:7.2f}"
                                    for s in stakes))
    print("outcomes", json.dumps(out["outcomes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""NNNC framework study, stage 1 (entry side): price entry levers on model signals offline.

From the per-bar signal tables (`signals.py`) this rebuilds each model's entry mask exactly as
`BaseStrategy.populate_entry_trend` does (BUY argmax AND p_buy > the fold's threshold, AND
volume > 0, AND rvol > its threshold, AND -- when `entry_enable_guards` -- atr_pct_roll,
guard_metric, close_norm, adx, bb_width) and checks it equals the recorded `enter_long`.

`simulate` then turns a mask into trades the way a backtest does: signal on bar i -> entry at
open[i+1]; one open trade per pair; at most `max_open_trades` open; stake = min(900, last
candle's quote volume / 10), skipped below the 1,000 quote-volume floor; exits by 027's faithful
replay with the .json operating point, ROI suppressed on bars after an entry signal
(`ignore_roi_if_entry_signal`). It must reproduce the recorded entries (FIDELITY_MIN) before any
lever is priced. Not modelled: wallet shrinkage, and the order in which freqtrade fills pairs
that signal on the same bar.

Levers (one at a time; EARLY half only, f00-f07 -- stage 2 pre-registers from this):
  thr_shift    the fold threshold + d
  soft_gate    + d only while volreg is in 026's off-state (< 1.3658); d = 1 is the hard gate
  guard_on     one of MLX's quality guards switched on (open-guard models)
  guard_off    one guard switched off (MLX, guards on)

    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.entries
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import feasibility as F  # noqa: E402
from research.exit_coupling import paths as P  # noqa: E402
from research.nnnc_framework import diagnose as D  # noqa: E402


HERE = Path(__file__).resolve().parent
SIG = HERE / "signals"
FIDELITY_MIN = 0.90
STAKE, MIN_QV, MIN_STAKE, HEADROOM, MAX_OPEN = 900.0, 1000.0, 10.0, 10.0, 10
VOLREG_CUT = 1.3658023280926226
#: MLX's guard values (NNNC_WF_fNN.json), the guards-on operating point.
GUARDS_ON = {"atr_pct": ("atr_pct_roll", "gt", 0.006), "guard": ("guard_metric", "lt", 0.3),
             "close_norm": ("close_norm", "lt", -0.1), "adx": ("adx", "gt", 38.0),
             "bb_width": ("bb_width", "gt", 0.06)}
GUARD_PARAM = {"atr_pct": "entry_atr_pct", "guard": "entry_guard_threshold",
               "close_norm": "entry_close_norm_threshold", "adx": "entry_adx_threshold",
               "bb_width": "entry_bb_width_threshold"}


def _cmp(x: np.ndarray, op: str, v: float) -> np.ndarray:
    return x > v if op == "gt" else x < v


def fold_params(prefix: str, fold: str) -> dict:
    return json.loads((STRAT / "NNNC" / f"{prefix}_{fold}.json").read_text())["params"]["buy"]


def base_mask(t: pd.DataFrame, buy: dict, thr: np.ndarray | float | None = None,
              guards: dict | None = None) -> np.ndarray:
    """populate_entry_trend's mask. `thr` per row overrides the fold threshold; `guards` maps a
    guard name -> (column, op, value) or None (off), overriding the .json for that guard."""
    t_fold = round(float(buy["prediction_threshold"]), 2)  # DecimalParameter(decimals=2)
    thr = t_fold if thr is None else thr
    m = t["buy_argmax"].to_numpy() & (t["p_buy"].to_numpy() > thr)
    m &= t["volume"].to_numpy() > 0
    m &= t["rvol"].to_numpy() > float(buy["entry_rvol_threshold"])
    g = {k: (col, op, float(buy[GUARD_PARAM[k]])) for k, (col, op, _) in GUARDS_ON.items()}
    g |= guards or {}
    if buy["entry_enable_guards"] or guards:
        for spec in g.values():
            if spec is not None:
                col, op, v = spec
                m &= _cmp(t[col].to_numpy(float), op, v)
    return m


class Model:
    def __init__(self, m: str):
        self.m = m
        tag, self.cls_name, self.prefix = D.MODELS[m]
        D._JSON_FOR[self.cls_name] = STRAT / "NNNC" / f"{self.prefix}_f00.json"
        cls, params, self.roi = D.exit_setup(self.cls_name)
        self.params = F.family_params(cls, params, D.FAMILY)
        plan = json.loads((STRAT / "evaluation/folds" / f"plan_{tag}.json").read_text())
        self.folds = {f["fold_id"]: f for f in plan["folds"]}
        rec = pd.DataFrame(F.archive_trades(tag))
        self.recorded = rec
        self.scale = rec.groupby("fold")["scale"].first().to_dict()
        self.tables = {f: pd.read_parquet(SIG / f"{m}_{f}.parquet") for f in self.folds}

    def simulate(self, fold: str, mask: np.ndarray) -> pd.DataFrame:
        t = self.tables[fold]
        f = self.folds[fold]
        sig = t.loc[mask, ["date", "pair"]]
        # bar index of every signal row in its pair's full bar arrays; entry = next bar
        ev = []
        for pair, g in sig.groupby("pair"):
            b, dates = F.bars_for(pair)
            idx = dates.get_indexer(pd.DatetimeIndex(g["date"]).tz_localize("UTC")
                                    if pd.DatetimeIndex(g["date"]).tz is None
                                    else pd.DatetimeIndex(g["date"]))
            ev += [(int(i) + 1, pair) for i in idx if i >= 0]
        ev.sort()
        suppressed = {}
        for pair, g in t.loc[t["enter_long"].to_numpy() == 1 if mask is None else mask,
                             ["date", "pair"]].groupby("pair"):
            _, dates = F.bars_for(pair)
            di = pd.DatetimeIndex(g["date"])
            di = di.tz_localize("UTC") if di.tz is None else di
            suppressed[pair] = frozenset(int(i) + 1 for i in dates.get_indexer(di) if i >= 0)
        out, busy, open_ = [], {}, []
        for e, pair in ev:
            b, dates = F.bars_for(pair)
            start = int(dates.searchsorted(pd.Timestamp(f["test_start"], tz="UTC")))
            end = int(dates.searchsorted(pd.Timestamp(f["test_end"], tz="UTC"))) - 1
            if e > end:
                continue
            open_ = [(x, p) for x, p in open_ if x >= e]
            if busy.get(pair, -1) >= e or len(open_) >= MAX_OPEN:
                continue
            qv = float(b.close[e - 1]) * float(_vol(pair)[e - 1])
            stake = min(STAKE, qv / HEADROOM)
            if qv < MIN_QV or stake < MIN_STAKE:
                continue
            tick = F._tick(pair, fold, b, start, end)
            x = P.replay(b, e, float(b.open[e]), self.params, roi=self.roi,
                         roi_suppressed=suppressed.get(pair, frozenset()), end=end, tick=tick)
            busy[pair] = x.idx
            open_.append((x.idx, pair))
            out.append({"fold": fold, "pair": pair, "e": e, "x": x.idx, "reason": x.reason,
                        "profit": x.profit, "stake": stake,
                        "pp": x.profit * stake * self.scale.get(fold, 0.01)})
        return pd.DataFrame(out, columns=["fold", "pair", "e", "x", "reason", "profit", "stake",
                                          "pp"])


_VOL: dict = {}


def _vol(pair: str) -> np.ndarray:
    if pair not in _VOL:
        from research.clfstudy.data import load_pair

        _VOL[pair] = load_pair(pair)["volume"].to_numpy(float)
    return _VOL[pair]


def fidelity(model: Model, sims: dict) -> dict:
    rec = model.recorded.copy()
    keys_rec = set()
    for r in rec.itertuples():
        _, dates = F.bars_for(r.pair)
        keys_rec.add((r.fold, r.pair, int(dates.get_loc(pd.Timestamp(r.open_date)))))
    keys_sim = {(s.fold, s.pair, s.e) for d in sims.values() for s in d.itertuples()}
    both = keys_rec & keys_sim
    sim_pp = sum(float(d["pp"].sum()) for d in sims.values())
    return {"recorded": len(keys_rec), "simulated": len(keys_sim),
            "recall": len(both) / max(len(keys_rec), 1),
            "precision": len(both) / max(len(keys_sim), 1),
            "pp_recorded": float((rec["profit_abs"] * rec["scale"]).sum()), "pp_simulated": sim_pp}


def volreg_on_dates() -> pd.Series:
    from research.clfstudy.data import load_pair
    from research.regime_gate.feasibility import PAIRS
    from utils import RegimeState as RS

    return RS.volreg({p: load_pair(f"{p}/USDT") for p in PAIRS}).sort_index()


def levers(model: Model, vr: pd.Series) -> dict:
    early = [f for f in model.folds if f in F.EARLY]
    base = {f: model.simulate(f, model.tables[f]["enter_long"].to_numpy() == 1) for f in early}
    base_pp = {f: float(d["pp"].sum()) for f, d in base.items()}

    def price(make):
        tot, up, down, n = 0.0, 0, 0, 0
        for f in early:
            d = model.simulate(f, make(f))
            delta = float(d["pp"].sum()) - base_pp[f]
            tot += delta
            n += len(d)
            up += delta > 1e-9
            down += delta < -1e-9
        return {"early_pp": round(tot, 3), "folds_up": up, "folds_down": down, "trades": n}

    res = {"base": {"early_pp": round(sum(base_pp.values()), 3),
                    "trades": int(sum(len(d) for d in base.values()))}}
    res["thr_shift"] = {}
    for d in (-0.05, 0.03, 0.06, 0.10):
        res["thr_shift"][str(d)] = price(lambda f, d=d: base_mask(
            model.tables[f], fold_params(model.prefix, f),
            thr=round(float(fold_params(model.prefix, f)["prediction_threshold"]), 2) + d))
    res["soft_gate"] = {}
    for d in (0.03, 0.06, 0.10, 1.0):
        def mk(f, d=d):
            t = model.tables[f]
            buy = fold_params(model.prefix, f)
            di = pd.DatetimeIndex(t["date"])
            di = di.tz_localize("UTC") if di.tz is None else di
            v = vr.reindex(di, method="ffill").to_numpy()
            off = (v < VOLREG_CUT) & ~np.isnan(v)
            thr = round(float(buy["prediction_threshold"]), 2) + np.where(off, d, 0.0)
            return base_mask(t, buy, thr=thr)
        res["soft_gate"][str(d)] = price(mk)
    buy0 = fold_params(model.prefix, "f00")
    if buy0["entry_adx_threshold"] < 0:  # guards open
        res["guard_on"] = {g: price(lambda f, g=g: base_mask(
            model.tables[f], fold_params(model.prefix, f), guards={g: GUARDS_ON[g]}))
            for g in GUARDS_ON}
        res["guard_on"]["rvol"] = price(lambda f: base_mask(
            model.tables[f], fold_params(model.prefix, f) | {"entry_rvol_threshold": 2.0}))
    else:
        res["guard_off"] = {g: price(lambda f, g=g: base_mask(
            model.tables[f], fold_params(model.prefix, f), guards={g: None}))
            for g in GUARDS_ON}
        res["guard_off"]["rvol"] = price(lambda f: base_mask(
            model.tables[f], fold_params(model.prefix, f) | {"entry_rvol_threshold": -1.0}))
    return res


def main(argv=None) -> int:
    models = (argv or sys.argv[1:] or ["ARO,GBR,GFM,MLX"])[0].split(",")
    vr = volreg_on_dates()
    out_path = HERE / "entries.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for m in models:
        model = Model(m)
        mism = {f: int((base_mask(t, fold_params(model.prefix, f))
                        != (t["enter_long"].to_numpy() == 1)).sum())
                for f, t in model.tables.items()}
        sims = {f: model.simulate(f, model.tables[f]["enter_long"].to_numpy() == 1)
                for f in model.folds}
        fid = fidelity(model, sims)
        entry = {"mask_mismatch_rows": mism, "fidelity": fid}
        print(m, "mask mismatches", sum(mism.values()), "fidelity", fid, flush=True)
        if fid["recall"] >= FIDELITY_MIN and fid["precision"] >= FIDELITY_MIN:
            entry["levers_early"] = levers(model, vr)
        out[m] = entry
        out_path.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

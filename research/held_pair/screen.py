"""Held-pair policy screen (2026-10-13): what should happen when a fresh entry signal fires on a
pair that already holds a trade?

The 24 h grace study (research/st_grace/report.md) found that 99% of the entries a longer grace
forgoes are blocked by freqtrade's one-trade-per-pair rule, not by the 10 slots, and that the
forgone entries were profitable (SimpleTrigger reversion +37 pp, NNNC arm D +13.6 pp).

On top of `nnnc_framework.entries.Model.simulate` (the faithful book rebuild: next-bar entry, one
trade per pair, 10 slots, 027 exit replay with the .json operating point) this prices, per held
pair and signal bar i (entry bar e = i + 1):

  NONE    freqtrade today: the signal is ignored (must equal `Model.simulate`).
  ROT     close the held trade at open[e] (a custom exit); the pair is free from e + 1, so it
          re-enters only if the signal persists (freqtrade checks entries before exits per
          candle and re-enters on the same candle only for short reversals).
  ANCHOR  no exit: the held trade's clocks (grace, unclog, max hold) and its take-profit/stop
          reference restart at open[e]; its profit stays measured from the original entry.
          ROI keeps its 027 replay semantics but measured from the anchor (approximation).
  STACK   a second position on the pair (ceiling only: freqtrade has no live position stacking).
  DCA     freqtrade position adjustment: add a second stake (same sizing rule) to the held trade
          at open[e], at most once; open rate = amount-weighted average; clocks and the
          take-profit/stop reference restart at e from the average (the 027 replay cannot resume
          a trade mid-life with its old clocks). Bounded by the wallet (10,000 x 0.99 of open
          stake); uses no extra slot.

Each policy can be conditioned: `loss` = only when the held trade is under water at open[e];
`min_age_h` = only when the held trade (or its last anchor) is at least that old.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.screen

Policies are priced on the EARLY half (f00-f07) only; HP_HALVES=both adds the late half, for the
report after the arms are pre-registered.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.screen --dump-adds ARO 24

writes `adds_ARO.json`, the early-half DCA:all:1h adds at 24 h grace (031 fidelity input).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import feasibility as F  # noqa: E402
from research.exit_coupling import paths as P  # noqa: E402
from research.nnnc_framework import diagnose as D  # noqa: E402
from research.nnnc_framework import entries as E  # noqa: E402
from utils.ExitPolicy import net_from_price_ratio  # noqa: E402


HERE = Path(__file__).resolve().parent
WALLET = 10000 * 0.99
POLICIES = [("NONE", None, 0.0)] + [
    (p, cond, age) for p in ("ROT", "ANCHOR", "STACK", "DCA") for cond in ("all", "loss")
    for age in (1.0, 4.0)
]
GRACE = (24.0, 48.0)
HALVES = ("early", "late") if os.environ.get("HP_HALVES") == "both" else ("early",)


def simulate(model: E.Model, fold: str, mask, params, policy: str, cond: str | None,
             min_age_h: float, add_frac: float = 1.0) -> pd.DataFrame:
    t = model.tables[fold]
    f = model.folds[fold]
    sig = t.loc[mask, ["date", "pair"]]
    ev = []
    for pair, g in sig.groupby("pair"):
        _, dates = F.bars_for(pair)
        di = pd.DatetimeIndex(g["date"])
        di = di.tz_localize("UTC") if di.tz is None else di
        ev += [(int(i) + 1, pair) for i in dates.get_indexer(di) if i >= 0]
    ev.sort()
    suppressed = {}
    for pair, g in t.loc[mask, ["date", "pair"]].groupby("pair"):
        _, dates = F.bars_for(pair)
        di = pd.DatetimeIndex(g["date"])
        di = di.tz_localize("UTC") if di.tz is None else di
        suppressed[pair] = frozenset(int(i) + 1 for i in dates.get_indexer(di) if i >= 0)
    hours_per_bar = params.timeframe_minutes / 60.0
    trades: list[dict] = []  # every trade; open ones are those with x >= current e
    held: dict = {}  # pair -> list of indices into trades (open ones)
    fee = params.fee_per_side

    def replay(pair, e, rate, end, tick):
        b, _ = F.bars_for(pair)
        return P.replay(b, e, rate, params, roi=model.roi,
                        roi_suppressed=suppressed.get(pair, frozenset()), end=end, tick=tick)

    for e, pair in ev:
        b, dates = F.bars_for(pair)
        start = int(dates.searchsorted(pd.Timestamp(f["test_start"], tz="UTC")))
        end = int(dates.searchsorted(pd.Timestamp(f["test_end"], tz="UTC"))) - 1
        if e > end:
            continue
        n_open = sum(1 for tr in trades if tr["x"] >= e)
        mine = [k for k in held.get(pair, []) if trades[k]["x"] >= e]
        held[pair] = mine
        tick = F._tick(pair, fold, b, start, end)
        if mine and policy != "STACK":
            tr = trades[mine[0]]
            if policy == "NONE" or tr["x"] <= e:
                continue
            o = float(b.open[e])
            if (e - tr["anchor"]) * hours_per_bar < min_age_h:
                continue
            if cond == "loss" and net_from_price_ratio(o / tr["open_rate"], fee) >= 0:
                continue
            # 031 follow-up: pyramiding -- add only to a held trade that is in profit
            if cond == "profit" and net_from_price_ratio(o / tr["open_rate"], fee) <= 0:
                continue
            if policy == "ROT":
                tr.update(x=e, price=o, reason="rotate")
            elif policy == "DCA":
                if tr["adds"] >= 1:
                    continue
                open_stake = sum(x["stake"] for x in trades if x["x"] >= e)
                qv = float(b.close[e - 1]) * float(E._vol(pair)[e - 1])
                add = min(add_frac * E.STAKE, qv / E.HEADROOM, WALLET - open_stake)
                if qv < E.MIN_QV or add < E.MIN_STAKE:
                    continue
                amount = tr["stake"] / tr["open_rate"] + add / o
                avg = (tr["stake"] + add) / amount
                x = replay(pair, e, avg, end, tick)
                tr.update(x=x.idx, price=x.price, reason=x.reason, anchor=e, open_rate=avg,
                          stake=tr["stake"] + add, adds=tr["adds"] + 1, add_e=e, add_stake=add)
            else:  # ANCHOR
                x = replay(pair, e, o, end, tick)
                tr.update(x=x.idx, price=x.price, reason=x.reason, anchor=e,
                          anchors=tr["anchors"] + 1)
            continue
        if policy == "STACK" and len(mine) >= 2:
            continue
        if policy == "STACK" and mine:
            tr0 = trades[mine[0]]
            if (e - tr0["anchor"]) * hours_per_bar < min_age_h:
                continue
            if cond == "loss" and net_from_price_ratio(float(b.open[e]) / tr0["open_rate"],
                                                       fee) >= 0:
                continue
        if n_open >= E.MAX_OPEN:
            continue
        qv = float(b.close[e - 1]) * float(E._vol(pair)[e - 1])
        stake = min(E.STAKE, qv / E.HEADROOM)
        if qv < E.MIN_QV or stake < E.MIN_STAKE:
            continue
        rate = float(b.open[e])
        x = replay(pair, e, rate, end, tick)
        stacked = bool(mine)
        held.setdefault(pair, []).append(len(trades))
        trades.append({"fold": fold, "pair": pair, "e": e, "open_rate": rate, "stake": stake,
                       "x": x.idx, "price": x.price, "reason": x.reason, "anchor": e,
                       "anchors": 0, "adds": 0, "stacked": stacked, "add_e": -1,
                       "add_stake": 0.0})
    out = pd.DataFrame(trades)
    if out.empty:
        return out
    out["profit"] = [net_from_price_ratio(p / r, fee) for p, r in zip(out["price"],
                                                                      out["open_rate"])]
    out["pp"] = out["profit"] * out["stake"] * model.scale.get(fold, 0.01)
    return out


def dump_adds(m: str, grace: float, halves=("early",), cond: str = "all") -> Path:
    """031 T019: the DCA:all:1h adds (fold, pair, add date, trade open date, add stake) at
    `grace`, for `research.held_pair.fidelity`."""
    model = E.Model(m)
    params = dataclasses.replace(model.params, stoploss_grace_hours=grace)
    rows, delta = [], {}
    for f in model.folds:
        if F.half_of(f) not in halves:
            continue
        mask = model.tables[f]["enter_long"].to_numpy() == 1
        sim = simulate(model, f, mask, params, "DCA", cond, 1.0)
        base = simulate(model, f, mask, params, "NONE", None, 0.0)
        delta[f] = round(float(sim["pp"].sum() - base["pp"].sum()), 4)
        for t in sim.itertuples():
            if t.add_e < 0:
                continue
            _, dates = F.bars_for(t.pair)
            rows.append({"fold": f, "pair": t.pair, "add_date": str(dates[t.add_e]),
                         "open_date": str(dates[t.e]), "add_stake": float(t.add_stake),
                         "exit_idx_eq_add": bool(t.x == t.add_e)})
    out = HERE / (f"adds_{m}.json" if cond == "all" else f"adds_{m}_{cond}.json")
    out.write_text(json.dumps({"model": m, "grace": grace, "policy": f"DCA:{cond}:1h",
                               "halves": list(halves), "fold_delta_pp": delta, "adds": rows},
                              indent=1))
    print(f"{m}: {len(rows)} screen adds -> {out.name}")
    return out


def main(argv=None) -> int:
    args = list(argv or sys.argv[1:])
    if args[:1] == ["--dump-adds"]:
        dump_adds(args[1], float(args[2]), cond=args[3] if len(args) > 3 else "all")
        return 0
    models = (args or ["ARO,GBR,GFM,MLX"])[0].split(",")
    out_path = HERE / f"screen_nnnc{D.SUFFIX}{'_both' if len(HALVES) == 2 else ''}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for m in models:
        model = E.Model(m)
        masks = {f: model.tables[f]["enter_long"].to_numpy() == 1 for f in model.folds}
        # fidelity: the NONE policy must equal the study's simulator
        ref = {f: model.simulate(f, masks[f]) for f in model.folds}
        res[m] = {"fidelity": E.fidelity(model, ref)}
        mine0 = {f: simulate(model, f, masks[f], model.params, "NONE", None, 0.0)
                 for f in model.folds}
        same = all(len(ref[f]) == len(mine0[f]) and
                   abs(ref[f]["pp"].sum() - mine0[f]["pp"].sum()) < 1e-6 for f in model.folds)
        res[m]["none_equals_simulator"] = same
        print(m, res[m], flush=True)
        if not same:
            out_path.write_text(json.dumps(res, indent=1))
            return 2
        for g in GRACE:
            params = dataclasses.replace(model.params, stoploss_grace_hours=g)
            folds = [f for f in model.folds if F.half_of(f) in HALVES]
            base = {f: simulate(model, f, masks[f], params, "NONE", None, 0.0) for f in folds}
            cell = {}
            for pol, cond, age in POLICIES:
                sims = {f: simulate(model, f, masks[f], params, pol, cond, age) for f in folds}
                d = {h: 0.0 for h in HALVES}
                n = {h: 0 for h in HALVES}
                for f in folds:
                    h = F.half_of(f)
                    d[h] += float(sims[f]["pp"].sum() - base[f]["pp"].sum())
                    n[h] += len(sims[f])
                key = pol if cond is None else f"{pol}:{cond}:{age:g}h"
                acts = sum(int((s["reason"] == "rotate").sum()) + int(s["anchors"].sum())
                           + int(s["stacked"].sum()) + int(s["adds"].sum()) for s in sims.values())
                cell[key] = {**{f"{h}_pp": round(d[h], 3) for h in HALVES}, "trades": n,
                             "actions": acts}
            res[m][f"grace_{g:g}"] = {
                "base_pp": {h: round(sum(float(base[f]["pp"].sum()) for f in folds
                                         if F.half_of(f) == h), 3) for h in HALVES},
                "policies": cell}
            print(m, g, json.dumps(cell), flush=True)
        out_path.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

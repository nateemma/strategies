"""Why the offline within-pair tercile did not carry over to the causal signal-rank guard (informal).

Per question (liqsr arm vs its s4k control), in score points (profit_abs x the fold's scale, so the
parts sum to the per-half score gap):
  1. agreement: causal keep (arm's own guard dump) x full-sample within-pair qv tercile of the
     control's trades, with P&L per cell;
  2. static subset: control trades the causal guard keeps vs the offline top tercile;
  3. decomposition of arm - control into shared trades (same pair + open), control-only trades
     (refused by the guard / displaced) and arm-only trades (same signal run as a refused control
     trade = late entry / other).
    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.tb_model.liqsr_decomp <dump root>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.tb_model.liqsr import QUESTIONS, _dumps  # noqa: E402

EARLY = {f"f{i:02d}" for i in range(8)}
BAR = pd.Timedelta("15min")


def _runs(dumps: dict) -> dict:
    """Signal run id per dump row: consecutive 15m mask rows form one run."""
    for frames in dumps.values():
        for d in frames:
            d["run"] = (d.date.diff() != BAR).cumsum()
    return dumps


def _bind(tag: str, dumps: dict) -> pd.DataFrame:
    from research.exit_coupling import feasibility as F

    out = []
    for t in F.archive_trades(tag):
        pair = t["pair"].replace("/", "_")
        sig = pd.Timestamp(t["open_date"]).tz_localize(None) - BAR
        d = min((x for x in dumps.get(pair, []) if x.date.iloc[0] <= sig <= x.date.iloc[-1]),
                key=lambda x: x.date.iloc[0], default=None)
        row = None if d is None else d[d.date == sig]
        r = None if row is None or row.empty else row.iloc[0]
        out.append({"pair": t["pair"].split("/")[0], "open": sig + BAR,
                    "close": pd.Timestamp(t["close_date"]).tz_localize(None), "fold": t["fold"],
                    "pp": float(t["profit_abs"]) * float(t["scale"]),
                    "qv": None if r is None else float(r["qv"]),
                    "keep": bool(r is not None and r["keep"]),
                    "run": None if r is None else (id(d), int(r["run"]))})
    x = pd.DataFrame(out)
    x["early"] = x.fold.isin(EARLY)
    return x


def _pp(x: pd.DataFrame) -> str:
    return f"{len(x):5d} tr  {x.pp.sum():7.2f} pp (e {x[x.early].pp.sum():6.2f} / l {x[~x.early].pp.sum():6.2f})"


def main(argv=None) -> int:
    root = Path((argv or sys.argv[1:])[0])
    for q, (arm, ctl) in QUESTIONS.items():
        dumps = _runs(_dumps(root / arm))
        a, c = _bind(arm, dumps), _bind(ctl, dumps)
        print(f"\n=== {q}: {arm} vs {ctl}   arm {a.pp.sum():.2f} pp, control {c.pp.sum():.2f} pp, "
              f"unbound control {c.qv.isna().mean():.1%}")
        c["terc"] = c.groupby("pair").qv.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)) + 1
        print("-- 1. causal keep x offline within-pair tercile (control trades)")
        for (k, t), g in c.dropna(subset=["terc"]).groupby(["keep", "terc"]):
            print(f"   keep={str(k):5s} tercile {int(t)}: {_pp(g)}")
        print(f"   agreement (keep == top tercile): {((c.terc == 3) == c.keep).mean():.1%}; "
              f"top tercile kept {c[c.terc == 3].keep.mean():.1%}; bottom tercile kept {c[c.terc == 1].keep.mean():.1%}")
        print("-- 2. static subsets of control trades")
        print(f"   all                 {_pp(c)}")
        print(f"   causal keep         {_pp(c[c.keep])}")
        print(f"   causal refuse       {_pp(c[~c.keep])}")
        print(f"   offline top tercile {_pp(c[c.terc == 3])}")
        print("-- 3. arm - control decomposition")
        key = ["pair", "open"]
        m = a.merge(c, on=key, how="outer", suffixes=("_a", "_c"), indicator=True)
        sh = m[m._merge == "both"]
        diff = pd.DataFrame({"pp": sh.pp_a - sh.pp_c, "early": sh.early_a.astype(bool)})
        print(f"   shared (arm - ctl)       {_pp(diff)}   [same exit {(sh.close_a == sh.close_c).mean():.1%}]")
        co = c.merge(a[key], on=key, how="left", indicator=True)
        co = co[co._merge == "left_only"]
        neg = lambda x: x.assign(pp=-x.pp)  # noqa: E731
        print(f"   ctl-only refused  (-)    {_pp(neg(co[~co.keep]))}")
        print(f"   ctl-only displaced (-)   {_pp(neg(co[co.keep]))}")
        ao = a.merge(c[key], on=key, how="left", indicator=True)
        ao = ao[ao._merge == "left_only"]
        refused_runs = set(co[~co.keep].run.dropna())
        late = ao.run.isin(refused_runs)
        print(f"   arm-only, same run as refused ctl {_pp(ao[late])}")
        print(f"   arm-only, other                   {_pp(ao[~late])}")
        if late.any():
            lag = []
            for _, r in ao[late].iterrows():
                o = co[(co.run == r.run) & ~co.keep]
                lag.append((r.open - o.open.min()) / BAR)
            print(f"   late-entry delay (bars): median {pd.Series(lag).median():.0f}")
        tot = diff.pp.sum() - co.pp.sum() + ao.pp.sum()
        print(f"   check: sum {tot:.2f} vs arm - ctl {a.pp.sum() - c.pp.sum():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

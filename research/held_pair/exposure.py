"""031 T026 (research R9): capital at risk, crowd-out and the leverage control for a DCA arm.

From the pooled arm and control trades (`evaluation.pool`, with each add's fill time and stake):

* exposure    time-weighted average and peak committed stake (event sweep over opens, adds,
              closes);
* crowd-out   the 24 h grace study's split of the gain, keyed on (fold, pair, open_date): same
              entries (Δpp), control entries forgone (n, of which pair-held n, pp), new entries
              (n, pp);
* leverage    the gain the adds' extra capital would earn at the control's base return: the
              adds' stake-hours x the control's pp per stake-hour, per fold. A gain no larger
              than this is "explained by extra capital".

    PYTHONPATH=. ../../.venv/bin/python -m research.held_pair.exposure hp_aro_dca wf_aro_g24
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

FOLDS = STRAT / "evaluation" / "folds"


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _pp(t: dict) -> float:
    return float(t["profit_abs"]) * float(t["scale_pp_per_abs"])


def legs(t: dict) -> list[tuple]:
    """(start, end, stake) segments of one trade's committed stake."""
    o, c = _ts(t["open_date"]), _ts(t["close_date"])
    fills = t.get("entry_fills")
    if not fills:
        return [(o, c, float(t.get("stake_amount") or 0.0))]
    return [(_ts(pd.Timestamp(ts, unit="ms", tz="UTC")), c, float(s)) for ts, s in fills]


def exposure(trades: list[dict]) -> dict:
    ev = []
    for t in trades:
        for s, e, k in legs(t):
            ev += [(s, k), (e, -k)]
    if not ev:
        return {"avg_stake": 0.0, "peak_stake": 0.0, "stake_hours": 0.0}
    ev.sort(key=lambda x: (x[0], x[1]))  # closes before opens at the same instant
    cur = peak = area = 0.0
    prev = ev[0][0]
    for ts, d in ev:
        area += cur * (ts - prev).total_seconds() / 3600.0
        cur += d
        peak = max(peak, cur)
        prev = ts
    span = (ev[-1][0] - ev[0][0]).total_seconds() / 3600.0
    return {"avg_stake": area / span if span else 0.0, "peak_stake": peak, "stake_hours": area}


def _key(t: dict) -> tuple:
    return (t["fold_id"], t["pair"], _ts(t["open_date"]))


def decomposition(arm: list[dict], ctl: list[dict]) -> dict:
    a, c = {_key(t): t for t in arm}, {_key(t): t for t in ctl}
    same = set(a) & set(c)
    held = {}
    for t in arm:
        held.setdefault((t["fold_id"], t["pair"]), []).append((_ts(t["open_date"]),
                                                                _ts(t["close_date"])))
    forgone = [c[k] for k in set(c) - set(a)]
    pair_held = sum(any(o <= k[2] < e for o, e in held.get((k[0], k[1]), []))
                    for k in (_key(t) for t in forgone))
    new = [a[k] for k in set(a) - set(c)]
    return {"same_n": len(same), "same_entry_pp": sum(_pp(a[k]) - _pp(c[k]) for k in same),
            "forgone_n": len(forgone), "forgone_pair_held_n": int(pair_held),
            "forgone_pp": sum(_pp(t) for t in forgone),
            "new_n": len(new), "new_pp": sum(_pp(t) for t in new)}


def leverage_control(arm: list[dict], ctl: list[dict]) -> dict:
    """Per fold: adds' stake-hours x the control's pp per stake-hour; summed."""
    out, total = {}, 0.0
    for f in sorted({t["fold_id"] for t in ctl} | {t["fold_id"] for t in arm}):
        c = [t for t in ctl if t["fold_id"] == f]
        sh = sum((e - s).total_seconds() / 3600.0 * k for t in c for s, e, k in legs(t))
        rate = sum(_pp(t) for t in c) / sh if sh else 0.0
        extra = sum((e - s).total_seconds() / 3600.0 * k for t in arm if t["fold_id"] == f
                    for s, e, k in legs(t)[1:])
        out[f] = {"extra_stake_hours": extra, "ctl_pp_per_stake_hour": rate,
                  "leverage_pp": extra * rate}
        total += extra * rate
    return {"leverage_pp": total, "per_fold": out}


def report(arm_tag: str, ctl_tag: str, folds: set | None = None) -> dict:
    from evaluation.pool import pool

    ap = pool(json.loads((FOLDS / f"plan_{arm_tag}.json").read_text()))
    cp = pool(json.loads((FOLDS / f"plan_{ctl_tag}.json").read_text()))
    arm = [t for t in ap["trades"] if folds is None or t["fold_id"] in folds]
    ctl = [t for t in cp["trades"] if folds is None or t["fold_id"] in folds]
    n_adds = sum((t.get("n_entries") or 1) - 1 for t in arm)
    gain = sum(_pp(t) for t in arm) - sum(_pp(t) for t in ctl)
    lev = leverage_control(arm, ctl)
    return {"arm": arm_tag, "control": ctl_tag, "n_adds": n_adds,
            "trade_gain_pp": gain, "exposure_arm": exposure(arm), "exposure_ctl": exposure(ctl),
            "decomposition": decomposition(arm, ctl), "leverage": lev,
            "beats_leverage": gain > lev["leverage_pp"]}


def main(argv=None) -> int:
    args = list(argv or sys.argv[1:])
    out = report(args[0], args[1])
    print(json.dumps({k: v for k, v in out.items() if k != "leverage"}, indent=1, default=str))
    print(f"leverage control {out['leverage']['leverage_pp']:+.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

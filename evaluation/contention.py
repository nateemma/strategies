"""What combining trigger families cost each family (feature 024, contract C6).

A combined arm runs triggers from two families with shared position slots and one position per
pair. Compared with each family's single-family arm (same folds, same exits for that family):

  lost    -- an entry the single-family arm took that the combined arm did not. Its cause is read
             from the combined arm's trades open at that instant, first match wins:
               pair_held_by_other_family  the pair was occupied by the OTHER family's trade
               pair_held_same_family      occupied by a same-family trade (path divergence)
               slots_full                 open trades >= max_open_trades
               divergence                 none of the above (an earlier difference cascading)
  gained  -- a combined-arm entry of that family the single-family arm did not take.

Trades are keyed on (fold_id, pair, open_date), as everywhere in the harness. Every figure is
split early (f00-f07) / late (f08-f14). PROVISIONAL under quality gate 2.

    python -m evaluation.contention --combined pooled_combo.json \\
        --single reversion=pooled_st_gbb.json --single momentum=pooled_st_bgo_mx.json \\
        --config config/config_wf.json --out evaluation/contention_st_fam_combo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.FamilyExits import family_of


CAUSES = ("pair_held_by_other_family", "pair_held_same_family", "slots_full", "divergence")
EARLY = frozenset(f"f{i:02d}" for i in range(8))
LABEL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"


def _frame(pooled: dict) -> pd.DataFrame:
    df = pd.DataFrame(pooled.get("trades", []))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "pair",
                "open_date",
                "close_date",
                "profit_ratio",
                "enter_tag",
                "_open",
                "_close",
                "family",
                "half",
            ]
        )
    df["_open"] = pd.to_datetime(df["open_date"], utc=True)
    df["_close"] = pd.to_datetime(df["close_date"], utc=True)
    df["family"] = df.get("enter_tag", pd.Series([None] * len(df))).map(family_of)
    df["half"] = df["fold_id"].map(lambda f: "early" if f in EARLY else "late")
    return df


def _key(df):
    return list(zip(df["fold_id"], df["pair"], df["_open"], strict=True))


def _cause(row, combo: pd.DataFrame, family: str, max_open: int) -> str:
    t = row["_open"]
    live = combo[
        (combo["fold_id"] == row["fold_id"]) & (combo["_open"] <= t) & (combo["_close"] > t)
    ]
    held = live[live["pair"] == row["pair"]]
    if len(held):
        return (
            "pair_held_same_family"
            if (held["family"] == family).any()
            else "pair_held_by_other_family"
        )
    if len(live) >= max_open:
        return "slots_full"
    return "divergence"


def contention(combined: dict, singles: dict, max_open_trades: int) -> dict:
    combo = _frame(combined)
    out = {"label": LABEL, "max_open_trades": max_open_trades}
    for family, pooled in singles.items():
        single = _frame(pooled)
        mine = combo[combo["family"] == family]
        combo_keys, single_keys = set(_key(mine)), set(_key(single))
        rec = {}
        for half in ("early", "late"):
            s_h, c_h = single[single["half"] == half], mine[mine["half"] == half]
            # a typed boolean mask with .loc: `df[[]]` on an empty half selects zero COLUMNS
            lost = s_h.loc[np.array([k not in combo_keys for k in _key(s_h)], dtype=bool)]
            gained = c_h.loc[np.array([k not in single_keys for k in _key(c_h)], dtype=bool)]
            causes = dict.fromkeys(CAUSES, 0)
            for _, r in lost.iterrows():
                causes[_cause(r, combo, family, max_open_trades)] += 1
            tags = c_h.get("enter_tag", pd.Series(dtype=object)).fillna("")
            rec[half] = {
                "single_trades": len(s_h),
                "combined_trades": len(c_h),
                "lost": len(lost),
                "lost_by_cause": causes,
                "gained": len(gained),
                "lost_pnl_pct": round(float(lost["profit_ratio"].sum() * 100), 4),
                "gained_pnl_pct": round(float(gained["profit_ratio"].sum() * 100), 4),
                "overlap_entries": int(tags.str.endswith("+ov").sum()),
            }
        out[family] = rec
    return out


def _md(rep: dict) -> str:
    lines = [
        "# Contention: combined arm vs single-family arms",
        "",
        f"*{rep['label']}*",
        "",
        (
            "| family | half | single | combined | lost "
            "| lost by cause (other-family / same / slots / divergence) "
            "| gained | lost pnl % | gained pnl % | overlap entries |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam, rec in rep.items():
        if not isinstance(rec, dict) or "early" not in rec:
            continue
        for half in ("early", "late"):
            r = rec[half]
            c = r["lost_by_cause"]
            lines.append(
                f"| {fam} | {half} | {r['single_trades']} | {r['combined_trades']} | {r['lost']} | "
                f"{c['pair_held_by_other_family']} / {c['pair_held_same_family']} / "
                f"{c['slots_full']} / {c['divergence']} | {r['gained']} | "
                f"{r['lost_pnl_pct']:+.2f} | "
                f"{r['gained_pnl_pct']:+.2f} | {r['overlap_entries']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combined", required=True)
    ap.add_argument("--single", action="append", required=True, help="family=pooled.json")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = [Path(a.combined), Path(a.config)] + [Path(x.split("=", 1)[1]) for x in a.single]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print(f"*** missing input(s): {missing}")
        return 2
    try:
        from evaluation.fingerprint import load_config

        cfg = load_config(a.config)
    except Exception:
        cfg = json.loads(Path(a.config).read_text())
    singles = {
        x.split("=", 1)[0]: json.loads(Path(x.split("=", 1)[1]).read_text()) for x in a.single
    }
    rep = contention(json.loads(Path(a.combined).read_text()), singles, int(cfg["max_open_trades"]))
    out = Path(a.out)
    out.write_text(json.dumps(rep, indent=2) + "\n")
    out.with_suffix(".md").write_text(_md(rep))
    print(_md(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())

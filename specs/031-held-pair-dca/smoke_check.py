"""031 T015: check the smoke arm's adds against the backtest archive and the ARO signal table.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python specs/031-held-pair-dca/smoke_check.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
from research.exit_coupling import feasibility as F  # noqa: E402
from research.nnnc_framework.entries import SIG, _vol  # noqa: E402


def archive_trades(rec: dict) -> list:
    z = STRAT.parents[0] / "backtest_results" / rec["result_file"]
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json") and "_config" not in n
                    and "meta" not in n)
        data = json.loads(zf.read(name))
    return next(iter(data["strategy"].values()))["trades"]


def main() -> int:
    rec = json.loads((STRAT / "regime/arms/hp_smoke_f00_F00.json").read_text())
    table = pd.read_parquet(SIG / "ARO_f00.parquet")
    table["date"] = pd.to_datetime(table["date"], utc=True)
    sig = set(zip(table.loc[table["enter_long"] == 1, "pair"], table.loc[table["enter_long"] == 1, "date"]))
    rows, bad = [], []
    for t in archive_trades(rec):
        entries = [o for o in t["orders"] if o["ft_is_entry"] and o.get("order_filled_timestamp")]
        if len(entries) < 2:
            continue
        amt = sum(o["amount"] for o in entries)
        avg = sum(o["amount"] * o["safe_price"] for o in entries) / amt
        add = entries[1]
        fill = pd.Timestamp(add["order_filled_timestamp"], unit="ms", tz="UTC")
        signal = fill - pd.Timedelta(minutes=15)
        b, dates = F.bars_for(t["pair"])
        i = int(dates.get_indexer([signal])[0])
        qv = float(b.close[i]) * float(_vol(t["pair"])[i]) if i >= 0 else float("nan")
        stake = add["amount"] * add["safe_price"]
        row = {"pair": t["pair"], "open": t["open_date"], "add_fill": str(fill),
               "signal_in_table": (t["pair"], signal) in sig,
               "open_rate": t["open_rate"], "avg": avg, "add_stake": round(stake, 2),
               "cap": round(qv / 10.0, 2), "total_stake": t["stake_amount"],
               "hours_after_open": (fill - pd.Timestamp(t["open_date"])).total_seconds() / 3600}
        rows.append(row)
        if not row["signal_in_table"]:
            bad.append(f"{t['pair']} add at {fill}: no enter_long on the signal candle")
        # freqtrade exports open_rate rounded to the pair's price precision
        if abs(avg - t["open_rate"]) > 1e-3 * avg:
            bad.append(f"{t['pair']} open_rate {t['open_rate']} != weighted avg {avg}")
        if stake > 900.0 + 1e-6 or stake > qv / 10.0 + 1e-6:
            bad.append(f"{t['pair']} add stake {stake:.2f} exceeds 900 / cap {qv / 10:.2f}")
        if row["hours_after_open"] < 1.0:
            bad.append(f"{t['pair']} add {row['hours_after_open']:.2f} h after the open")
    out = {"n_trades": rec["n_trades"], "n_adds": rec.get("n_adds"),
           "hp_counters": rec.get("hp_counters"), "checked": len(rows), "failures": bad,
           "adds": rows}
    (Path(__file__).parent / "smoke.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "adds"}, indent=1))
    for r in rows:
        print(r)
    return 1 if bad or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())

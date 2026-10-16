"""036 T016: the recording gate for US2/US3 (FR-019; research R10; clarification Q2 = C).

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.gate

The gate is OPEN when at least 28 days have been recorded AND every whitelist pair has at least 5
recorded orders. At 84 days it opens regardless, with the pairs still short listed as thin; they
are then pooled by the cost model's structural tier rule. Writes research/live_pnl/gate.json.
US2/US3 judging tools call `is_open()` and refuse while it is False.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
MIN_DAYS, MIN_ORDERS, MAX_DAYS = 28, 5, 84


def status(days: float, orders_per_pair: dict) -> dict:
    short = sorted(p for p, n in orders_per_pair.items() if n < MIN_ORDERS)
    is_open = (days >= MIN_DAYS and not short) or days >= MAX_DAYS
    return {"days": round(days, 2), "orders_per_pair": orders_per_pair, "open": bool(is_open),
            "thin_pairs": short if days >= MAX_DAYS else [],
            "rule": f">= {MIN_DAYS} days AND every pair >= {MIN_ORDERS} orders; or >= {MAX_DAYS} days"}


def compute(now: datetime | None = None) -> dict:
    from evaluation.fingerprint import load_config
    from research.live_pnl.parity import read_jsonl

    recs = [r for r in json.loads((HERE / "live_models.json").read_text()) if r.get("started")]
    if not recs:
        return status(0.0, {})
    now = now or datetime.now(timezone.utc)
    first = min(datetime.fromisoformat(r["started"]) for r in recs)
    pairs = load_config(HERE.parents[1] / "config" / "config_live_nnnc.json")["exchange"]["pair_whitelist"]
    counts = {p: 0 for p in pairs}
    for r in recs:
        o = read_jsonl("orders", r["run_id"])
        if not o.empty:
            o = o[~o.get("rejected", False).astype(bool)] if "rejected" in o else o
            for p, n in o.groupby("pair").size().items():
                counts[p] = counts.get(p, 0) + int(n)
    return status((now - first).total_seconds() / 86400, counts)


def is_open() -> bool:
    p = HERE / "gate.json"
    return p.is_file() and bool(json.loads(p.read_text()).get("open"))


def main() -> int:
    s = compute()
    (HERE / "gate.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

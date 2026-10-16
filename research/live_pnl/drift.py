"""036 T015: weekly operating-point drift of the live model (research R11; spec SC-007).

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.drift

The live BUY rate is the share of the signal log's pair-candles with p_buy > the threshold,
compared with the target rate the threshold was picked for (0.014735). A week whose rate falls
outside ×0.5–×2 of target is flagged. Writes research/live_pnl/drift.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
LOW, HIGH = 0.5, 2.0


def weekly(sig: pd.DataFrame, threshold: float, target: float) -> list[dict]:
    if sig.empty:
        return []
    s = sig.dropna(subset=["p_buy"]).copy()
    s["week"] = pd.to_datetime(s.candle, utc=True).dt.tz_localize(None).dt.to_period("W").astype(str)
    out = []
    for wk, g in s.groupby("week"):
        rate = float((g.p_buy > threshold).mean())
        ratio = rate / target if target else None
        out.append({"week": wk, "n": int(len(g)), "buy_rate": round(rate, 6),
                    "ratio_to_target": None if ratio is None else round(ratio, 3),
                    "flag": ratio is None or not (LOW <= ratio <= HIGH)})
    return out


def main() -> int:
    from research.live_pnl.parity import read_jsonl

    recs = [r for r in json.loads((HERE / "live_models.json").read_text()) if r.get("started")]
    out = {"label": "036 drift (PROVISIONAL)", "band": [LOW, HIGH], "runs": {}}
    for r in recs:
        out["runs"][r["run_id"]] = {"threshold": r["threshold"], "target_rate": r["target_rate"],
                                    "weeks": weekly(read_jsonl("signals", r["run_id"]),
                                                    r["threshold"], r["target_rate"])}
    (HERE / "drift.json").write_text(json.dumps(out, indent=1) + "\n")
    for rid, v in out["runs"].items():
        for w in v["weeks"]:
            print(rid, w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

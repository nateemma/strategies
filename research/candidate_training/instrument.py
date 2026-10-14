"""032 T018: instrument checks on the candidate-only arms (entries only; no P&L is read).

Per model (CO_ARO, CO_GBR), over f00-f14:
  (a) simulator fidelity on the arm's OWN recorded entries: the study simulator
      (`research.nnnc_framework.entries`) rebuilds the book from the arm's signal tables'
      `enter_long`; recall / precision of (fold, pair, entry bar) against the records, >= 0.95;
  (b) entry violations: recorded entries whose signal bar (entry bar - 1) fails the gbb predicate
      (guard_metric < 0.3 AND bb_width > 0.035) -- must be 0;
  (c) liveness: each fold's row_filter.json (kept share, class counts; SELL must be 0) and the
      per-fold threshold achieved rate from the pick log.

Needs the CO signal tables first:
    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.signals --models CO_ARO,CO_GBR
    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.instrument
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

HERE = Path(__file__).resolve().parent
FOLDS_DIR = STRAT / "evaluation" / "folds"
ARMS = {"CO_ARO": ("co_aro", "NNNC_CO_ARO", "NNNC_CO_ARO"),
        "CO_GBR": ("co_gbr", "NNNC_CO_GBR", "NNNC_CO_GBR")}
SPEC = (("guard_metric", "lt", 0.3), ("bb_width", "gt", 0.035))
FIDELITY_MIN = 0.95


def violations(recorded: pd.DataFrame, tables: dict, bars_for) -> int:
    """Recorded entries whose signal bar (entry bar - 1) fails SPEC, or has no signal row."""
    from Framework.RowFilter import row_filter_mask

    bad = 0
    for f, g in recorded.groupby("fold"):
        t = tables[f]
        d = pd.to_datetime(t["date"])
        d = d.dt.tz_localize("UTC") if d.dt.tz is None else d.dt.tz_convert("UTC")
        ok = dict(zip(zip(t["pair"], d), row_filter_mask(t, SPEC)))
        for r in g.itertuples():
            _, dates = bars_for(r.pair)
            e = int(dates.get_loc(pd.Timestamp(r.open_date)))
            sig = pd.Timestamp(dates[e - 1])
            sig = sig.tz_localize("UTC") if sig.tz is None else sig.tz_convert("UTC")
            bad += not ok.get((r.pair, sig), False)
    return bad


def achieved_rates(m: str) -> dict:
    p = FOLDS_DIR / f"thresholds_co_{m}_log.txt"
    out = {}
    if p.is_file():
        for line in p.read_text().splitlines():
            g = re.match(r"(f\d\d): threshold=([\d.]+)\s+achieved_rate=([\d.]+)%", line)
            if g:
                out[g[1]] = {"threshold": float(g[2]), "achieved_rate": float(g[3]) / 100}
    return out


def liveness(tag: str) -> dict:
    from research.candidate_training.train import _location

    plan = json.loads((FOLDS_DIR / f"plan_{tag}.json").read_text())
    out = {}
    for f in plan["folds"]:
        p = Path(_location(f, plan["base_strategy"], plan["config"], 1).path).parent / "row_filter.json"
        r = json.loads(p.read_text()) if p.is_file() else None
        out[f["fold_id"]] = None if r is None else {
            "train_kept_share": r["train"]["kept_share"], "train_class_counts": r["train"]["class_counts"],
            "val_kept_share": r["validation"]["kept_share"],
            "val_class_counts": r["validation"]["class_counts"]}
    return out


def main() -> int:
    from research.exit_coupling import feasibility as F
    from research.nnnc_framework import diagnose as D
    from research.nnnc_framework import entries as E

    D.MODELS.update(ARMS)  # runtime registration; the module's study lists are untouched
    out = {"label": "032 instrument checks (entries only; no P&L read)", "models": {}}
    for m in ARMS:
        model = E.Model(m)
        sims = {f: model.simulate(f, model.tables[f]["enter_long"].to_numpy() == 1)
                for f in sorted(model.folds)}
        fid = E.fidelity(model, sims)
        rec = model.recorded
        live = liveness(ARMS[m][0])
        res = {
            "fidelity": {"recall": round(fid["recall"], 4), "precision": round(fid["precision"], 4),
                         "recorded": fid["recorded"], "simulated": fid["simulated"]},
            "fidelity_pass": min(fid["recall"], fid["precision"]) >= FIDELITY_MIN,
            "entry_violations": violations(rec, model.tables, F.bars_for),
            "trades_per_fold": rec.groupby("fold").size().reindex(sorted(model.folds), fill_value=0)
                                  .astype(int).to_dict(),
            "liveness": live,
            "sell_rows_kept": int(sum((v or {}).get("train_class_counts", [0])[0] for v in live.values())),
            "thresholds": achieved_rates(ARMS[m][0].split("_")[1]),
        }
        out["models"][m] = res
        print(m, json.dumps({k: v for k, v in res.items() if k not in ("liveness", "thresholds")}),
              flush=True)
    (HERE / "instrument.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

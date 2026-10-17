"""037 re-run on TRADEABLE bars only (2026-10-17, user request; preregistration_vol.json).

32% of the screens' data_ok bars have zero volume (gap-filled candles, no trades); no strategy
enters on them, and the tbatr pair results turned out to sit almost entirely on them. This re-runs
the screens with the same locked rules and one change -- rows need volume > 0
(Panel(tradeable_only=True)) -- into research/indicator_screen/vol/ (score feathers in the cache's
out_vol/). The original outputs are untouched. Regime targets and the per-fold models are NOT re-run.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.vol [--only labels,tb,tbatr,mofn]
"""
from __future__ import annotations

import argparse
import json

from research.indicator_screen import harvest as H
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel, scored_columns
from research.indicator_screen.screen_single import HERE, LabelTargets, prereg

VOL = HERE / "vol"
FEATHERS = H.CACHE / "out_vol"
STEPS = ("labels", "tb", "tbatr", "mofn")


def labels(panel: Panel) -> None:
    from research.indicator_screen import screen_pairs as SP
    from research.indicator_screen import screen_single as SS

    SS.run("labels", ["e0", "early", "late"], panel=panel, out_dir=FEATHERS,
           summary_path=VOL / "single_labels.json", red_path=VOL / "redundancy.json")
    pr, t = prereg(), LabelTargets(panel)
    for target in ("label:buy", "label:sell"):
        sel = SP.select(panel, t, target, W.all_windows, scored_columns(), pr)
        res = SP.confirm(panel, t, sel, W.all_windows, pr)
        SP.out_path(target, VOL).write_text(json.dumps(res, indent=1))
        for pop, b in res["populations"].items():
            print(f"{target} [{pop}]: {b['n_passing_rule']} pass, {len(b['survivors'])} survivors, "
                  f"confirmed {sum(s.get('confirmed') == 'yes' for s in b['survivors'])}", flush=True)


def tb_label(name: str) -> None:
    from research.indicator_screen import tb

    tb.DEST, tb.FEATHERS, tb.PANEL_KW = VOL, FEATHERS, {"tradeable_only": True}
    tb.use(name)
    tb.single()
    tb.pairs()
    tb.resid()
    tb.report()


def mofn() -> None:
    from research.indicator_screen import mofn as M

    M.POOL_FROM, M.OUT, M.PANEL_KW = VOL / "pairs_label_tbatr_buy.json", VOL / "mofn.json", {"tradeable_only": True}
    M.main()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=",".join(STEPS))
    a = ap.parse_args(argv)
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    VOL.mkdir(exist_ok=True)
    FEATHERS.mkdir(parents=True, exist_ok=True)
    steps = a.only.split(",")
    if "labels" in steps:
        labels(Panel(tradeable_only=True))
    for name in ("tb", "tbatr"):
        if name in steps:
            tb_label(name)
    if "mofn" in steps:
        mofn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

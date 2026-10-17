"""ARO volume-filter retrain: the HGB sizer's instrument checks (cs_aro's, plus entries without a score
<= 1%) on `cs_arov`, the tag check against the retrained model's own signal table
(research/nnnc_framework/signals/AROV_fNN, linked as signals_fNN in user_data/data/aro_volume). No P&L.

    PYTHONPATH=. ../../.venv/bin/python -m research.aro_volume.instrument
"""
from __future__ import annotations

from pathlib import Path

from research.hgb_sizer import instrument as HI

HERE = Path(__file__).resolve().parent
STRAT = HERE.parents[1]
OUT = STRAT.parents[1] / "user_data" / "data" / "aro_volume"
SIG = STRAT / "research" / "nnnc_framework" / "signals"


def link_tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src in sorted(SIG.glob("AROV_f??.parquet")):
        dst = OUT / f"signals_{src.stem.split('_')[1]}.parquet"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)


if __name__ == "__main__":
    link_tables()
    HI.OUT, HI.HERE = OUT, HERE
    raise SystemExit(HI.main())

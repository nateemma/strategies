"""Read the pooled walk-forward archive into TradeRow records (T008, FR-001).

One row per archived trade. The target is REALISED money under real exits, fees and fills --
not a forward-excursion stamp on a bar -- which is what makes this label-free in the sense
the study means.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import metalabel as M          # noqa: E402


@dataclass(frozen=True)
class TradeRow:
    """An archived primary trade. `features` is filled later by reinfer.py."""

    fold_id: str
    pair: str
    open_date: str
    close_date: str
    profit_abs: float
    profit_ratio: float
    exit_reason: str
    scale_pp_per_abs: float
    features: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """(pair, open_date) — the join key, and what the reproduction gate compares."""
        return (self.pair, self.open_date)

    @property
    def target_win(self) -> bool:
        return self.profit_abs > 0

    @property
    def target_profit(self) -> float:
        """Retained for a future brief; UNUSED here — sizing is an explicit Non-Goal."""
        return self.profit_abs

    def describe(self) -> dict:
        return {"fold_id": self.fold_id, "pair": self.pair, "open_date": self.open_date,
                "close_date": self.close_date, "profit_abs": self.profit_abs,
                "profit_ratio": self.profit_ratio, "exit_reason": self.exit_reason,
                "scale_pp_per_abs": self.scale_pp_per_abs,
                "target_win": self.target_win, "features": self.features}


def load(path: str | Path | None = None) -> list[TradeRow]:
    """Every archived trade, all 15 folds. Filtering to the evaluation span is the caller's
    job — the warm-up folds are training data for the nested design, not waste."""
    p = Path(path) if path else ROOT / M.ARCHIVE_PATH
    d = json.loads(p.read_text())
    rows = [TradeRow(fold_id=t["fold_id"], pair=t["pair"], open_date=t["open_date"],
                     close_date=t["close_date"], profit_abs=t["profit_abs"],
                     profit_ratio=t["profit_ratio"], exit_reason=t["exit_reason"],
                     scale_pp_per_abs=t["scale_pp_per_abs"])
            for t in d["trades"]]
    if len(rows) != M.N_ARCHIVE_TRADES:
        raise ValueError(
            f"archive holds {len(rows)} trades, not the pinned {M.N_ARCHIVE_TRADES}. "
            "The floor and base rate are calibrated to that count and must be recomputed "
            "before anything else runs.")
    return rows


def fold_ranges(path: str | Path | None = None) -> dict[str, dict]:
    """Per-fold test window and trade count, from the archive's own plan."""
    p = ROOT / "regime/wf/folds/plan_nnnc.json"
    plan = json.loads(p.read_text())
    folds = plan.get("folds", plan if isinstance(plan, list) else [])
    return {f["fold_id"]: f for f in folds}


def by_fold(rows: list[TradeRow]) -> dict[str, list[TradeRow]]:
    out: dict[str, list[TradeRow]] = {}
    for r in rows:
        out.setdefault(r.fold_id, []).append(r)
    return out


def entry_keys(rows: list[TradeRow]) -> set[tuple[str, str]]:
    """The set the reproduction gate compares against (FR-005b)."""
    return {r.key for r in rows}

"""035: names of the survivor arms (research R9), shared by build_arms, train, signals and judge."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def class_name(tag: str) -> str:
    return f"NNNC_NI_{tag}"


def plan_tag(tag: str, role: str = "entry") -> str:
    """`ni_po_oi` (training / tables) or `ni_po_oi_sz` (the sizer arm)."""
    return f"ni_{tag.lower()}" + ("_sz" if role == "sz" else "")


def survivors() -> list[str]:
    """S1b's survivors when S1b has run (it replaces S1 as the gate to S2), else S1's."""
    for name in ("survivors_s1b.json", "survivors.json"):
        p = HERE / name
        if p.is_file():
            return list(json.loads(p.read_text()).get("survivors", []))
    return []


def table_plans() -> dict:
    """Signal-table registration for research/nnnc_framework/signals.py: NI_<TAG> -> plan tag."""
    return {f"NI_{t}": plan_tag(t) for t in survivors()}

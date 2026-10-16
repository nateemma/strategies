"""035: names of the survivor arms (research R9), shared by build_arms, train, signals and judge.

`model` is "ARO" (the screen's base, names unchanged) or "GBR" (the raw-feature plain LSTM,
NNNC_GbbRaw_Open; the GBR extension, preregistration_gbr.json): GBR names carry a `GBR_` part.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODELS = ("ARO", "GBR")


def _part(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"model {model!r} not in {MODELS}")
    return "" if model == "ARO" else "GBR_"


def class_name(tag: str, model: str = "ARO") -> str:
    return f"NNNC_NI_{_part(model)}{tag}"


def plan_tag(tag: str, role: str = "entry", model: str = "ARO") -> str:
    """`ni_po_oi` / `ni_gbr_po_oi` (training, tables) or `..._sz` (the sizer arm)."""
    return f"ni_{_part(model).lower()}{tag.lower()}" + ("_sz" if role == "sz" else "")


def table_name(tag: str, model: str = "ARO") -> str:
    return f"NI_{_part(model)}{tag}"


def survivors(model: str = "ARO") -> list[str]:
    """ARO: S1b's survivors when S1b has run (it replaces S1 as the gate to S2), else S1's.
    GBR: S1b-GBR's survivors."""
    names = ("survivors_s1b.json", "survivors.json") if model == "ARO" \
        else ("survivors_s1b_gbr.json",)
    for name in names:
        p = HERE / name
        if p.is_file():
            return list(json.loads(p.read_text()).get("survivors", []))
    return []


def table_plans() -> dict:
    """Signal-table registration for research/nnnc_framework/signals.py: NI_[GBR_]<TAG> -> plan.
    Includes the no-input retrain (NULL) and both32 (B32, ARO only) once their plans exist."""
    folds = HERE.parents[1] / "evaluation" / "folds"
    out = {table_name(t, m): plan_tag(t, model=m) for m in MODELS for t in survivors(m)}
    for m in MODELS:
        if (folds / f"plan_{plan_tag('NULL', model=m)}.json").is_file():
            out[table_name("NULL", m)] = plan_tag("NULL", model=m)
    if (folds / f"plan_{plan_tag('B32')}.json").is_file():
        out[table_name("B32")] = plan_tag("B32")
    return out

"""034: the declared label candidates (research R2) -- the single source every screen script reads.

Parameter rule, fixed before any training: each labeler's defaults, plus the control's horizon (48)
and +-2% thresholds (passed to every labeler as horizon / min_gain / min_loss; a labeler keeps only
the keys it accepts), plus a stop-matched adverse-excursion cap of 0.04 (as GFM) where a labeler has
one. triple_barrier's sell side is declared explicitly: in labels_triple_barrier_sell `sl` is the
DOWN barrier, so the mirror of the buy rule (+2% before -5%) is -2% before +5%.

CONTROL and GFM are not trained: their weights exist (`reuse`). BOG is scored as a sizer only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    tag: str
    method: int
    label_params: dict = field(default_factory=dict)
    label_params_sell: dict | None = None  # None = mirror label_params
    reuse: str | None = None               # existing plan whose weights are reused
    roles: tuple = ("sz", "es")
    order: int | None = None               # training order (None = not trained)


CANDIDATES = (
    Candidate("CONTROL", 27, reuse="wf_aro_g24", roles=()),
    Candidate("GFM", 28, reuse="sw_gfm_ctl_b030"),
    Candidate("FMAE", 0, {"max_drawdown": 0.04}, order=1),
    Candidate("TB", 1, {"pt": 0.02, "sl": 0.05}, {"pt": 0.05, "sl": 0.02}, order=2),
    Candidate("MMR", 3, order=3),
    Candidate("FEXP", 12, order=4),
    Candidate("QF", 2, order=5),
    Candidate("LMF", 4, {"max_drawdown": 0.04}, order=6),
    Candidate("BOG", 24, roles=("sz",), order=7),
)

BY_TAG = {c.tag: c for c in CANDIDATES}
TRAINED = tuple(sorted((c for c in CANDIDATES if c.order is not None), key=lambda c: c.order))


def sell_params(c: Candidate) -> dict:
    return dict(c.label_params if c.label_params_sell is None else c.label_params_sell)


def class_name(c: Candidate, model: str = "ARO") -> str:
    return f"NNNC_LS_{c.tag}" if model == "ARO" else f"NNNC_LS_{model}_{c.tag}"


def plan_tag(c: Candidate, role: str, model: str = "ARO") -> str:
    return f"ls_{c.tag.lower()}_{role}" if model == "ARO" else f"ls_{model.lower()}_{c.tag.lower()}_{role}"


def as_data() -> list[dict]:
    """The table as plain data, for the pre-registration."""
    return [{"tag": c.tag, "method": c.method, "label_params": c.label_params,
             "label_params_sell": c.label_params_sell, "reuse": c.reuse, "roles": list(c.roles),
             "order": c.order} for c in CANDIDATES]


def table_plans() -> dict:
    """`LS_<TAG>` -> the plan whose fold classes produce its signal tables (the training `es` plan;
    the sizer twin shares the weights). GFM's tables already exist as `GFM_fNN`."""
    return {f"LS_{c.tag}": plan_tag(c, "es") for c in TRAINED}

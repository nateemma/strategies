"""The cost-aware objective, with PRE-REGISTERED regularisation (T010, T010a, T011).

FR-009 requires a penalty on distance from the incumbent and on turnover. Both weights are
fixed in data-model.md §1a and READ FROM HERE AS CONSTANTS -- they are inputs, never tuned.
Choosing them after seeing the spans would be exactly the researcher-degrees-of-freedom this
feature exists to quantify, and a feature whose headline number is PBO cannot afford it.

    distance  4.93 pp/yr across the full space   one sweep-noise band, the SAME quantity
                                                 SC-010 uses to kill a move, so the penalty
                                                 and the gate agree by construction
    turnover  1.00 pp/yr per doubling            brief 02 measured 92% of an arm's apparent
                                                 advantage to be turnover; this makes churn
                                                 cost something the search must EARN, on top
                                                 of modelled costs

Both the regularised and unregularised scores are returned (T010a), so the report can show
what the penalty did rather than assert it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from research.labelfree import admissibility as AD
from research.labelfree.es import space as SP

# --- PRE-REGISTERED, data-model.md §1a. Do not tune. ---
W_DISTANCE_PP = 4.93
W_TURNOVER_PP = 1.00


@dataclass(frozen=True)
class Score:
    value: float                  # regularised — what the search optimises
    unregularised: float          # raw annualised return, pp/yr
    distance_penalty: float
    turnover_penalty: float
    admissible: bool
    reason: str | None
    n_trades: int
    trades_per_year: float
    mean_hold_h: float
    mean_exposure: float

    def as_row(self) -> dict:
        return asdict(self)


# Fraction of the INCUMBENT's exposure on the SAME panel that a candidate must retain. The
# absolute MIN_EXPOSURE is the wrong instrument here -- it marks production itself degenerate
# on a 90-day range (measured: f00 0.0196, f01 0.0474 against a 0.05 floor). The question the
# floor should ask is "is this materially less participating than production", and production
# is the only defensible reference for that.
INCUMBENT_EXPOSURE_FRACTION = 0.5


def score(trades, params: dict, span_years: float, incumbent_trades_per_year: float,
          incumbent_exposure: float | None = None) -> Score:
    """Turn a surrogate TradeSeries into the number the search ranks on.

    A candidate that does not trade is INADMISSIBLE with its score withheld (FR-004): it is
    not scored as zero, because this study has four separate instances of abstention being
    compared as though it were performance.
    """
    n = len(trades)
    tpy = n / span_years if span_years > 0 else 0.0
    exposure = trades.mean_exposure
    hold = trades.mean_hold_hours

    floor = (INCUMBENT_EXPOSURE_FRACTION * incumbent_exposure
             if incumbent_exposure else None)
    arm = AD.Arm(name="candidate", value=trades.annualised_pp,
                 mean_exposure=exposure, trades_per_year=tpy, exposure_floor=floor)
    if arm.degenerate:
        return Score(value=float("-inf"), unregularised=trades.annualised_pp,
                     distance_penalty=0.0, turnover_penalty=0.0,
                     admissible=False, reason=arm.reason, n_trades=n,
                     trades_per_year=tpy, mean_hold_h=hold, mean_exposure=exposure)

    d_pen = W_DISTANCE_PP * SP.distance_from_incumbent(params)
    t_ratio = (tpy / incumbent_trades_per_year) if incumbent_trades_per_year > 0 else 1.0
    t_pen = W_TURNOVER_PP * max(0.0, t_ratio - 1.0)

    return Score(value=trades.annualised_pp - d_pen - t_pen,
                 unregularised=trades.annualised_pp,
                 distance_penalty=d_pen, turnover_penalty=t_pen,
                 admissible=True, reason=None, n_trades=n,
                 trades_per_year=tpy, mean_hold_h=hold, mean_exposure=exposure)


def compare_to(arm_value: float, arm_exposure: float, arm_tpy: float,
               control_value: float, control_exposure: float, control_tpy: float,
               purpose: str = "acceptance") -> dict:
    """Every arm-vs-control verdict in this feature goes through the shared gate (FR-005).

    Reused unchanged from research/labelfree/admissibility.py rather than reimplemented --
    the whole point of that module is that the check was rewritten at every site and each
    rewrite checked only one side.
    """
    return AD.compare(
        AD.Arm("candidate", arm_value, arm_exposure, arm_tpy),
        AD.Arm("control", control_value, control_exposure, control_tpy),
        purpose=purpose)

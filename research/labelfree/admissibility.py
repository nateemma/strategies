"""One admissibility gate for every comparison in this study.

WHY THIS EXISTS. A non-trading policy is not forbidden -- cash is a legitimate position, and
forcing participation converts "this method found nothing" into "this method loses money",
which is strictly worse information. `losses.participation_penalty` already tried pushing
harder on the objective: its weight was raised 10 -> 100 and the guard still did not bite,
because w == 0 is a genuine global optimum when nothing is profitable.

The recurring cost is not that policies abstain. It is that ABSTENTION KEEPS BEING COMPARED
AS THOUGH IT WERE PERFORMANCE, and the check preventing that was re-written at every site.

Four instances, all in this study:

  brief 1  six best-ranked configurations below the floor, two flat on 100% and 99.9% of bars
  brief 2  the P&L control lost 52% in a RISING market, so any arm that declined to trade
           scored a large POSITIVE delta against it
  brief 4  the continuous arm was 40% degenerate (4/10 seeds at exposure 0.003-0.012)
  brief 4  brief 1's re-trained comparator ran at exposure 0.0004 -- 125x below the floor --
           and the raw delta read "WORSE", i.e. a policy that trades losing to one that
           abstains

The last one is the argument for centralising: the floor WAS implemented in arms.py, in
metalabel, and in the RL env, and the comparison site still checked only one side.

TWO MEASURES, because they fail differently. Mean exposure catches a policy that is flat.
Trades per year catches one that holds a single position forever -- non-zero exposure,
no decisions, and nothing being tested.
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_EXPOSURE = 0.05          # matches arms.py and rl/__init__.py
MIN_TRADES_PER_YEAR = 4.0    # below this, a "strategy" is one position held across the span


@dataclass(frozen=True)
class Arm:
    name: str
    value: float                       # the return or score being compared
    mean_exposure: float
    trades_per_year: float | None = None
    # OPTIONAL absolute exposure floor, defaulting to MIN_EXPOSURE. Added 2026-09-14 for
    # feature 013, where MIN_EXPOSURE is the WRONG INSTRUMENT and says so loudly: on a 90-day
    # test range the PRODUCTION strategy scores 0.0196-0.0474 exposure and is marked
    # degenerate. It is not abstaining -- it takes 13-20 trades in 90 days at ~4h each, so
    # ~2-5% occupancy is what a selective discrete-trade strategy LOOKS like.
    #
    # MIN_EXPOSURE was calibrated on continuous-exposure policies (briefs 1 and 4), where the
    # weight is held every bar and 0.05 genuinely separates trading from flat. This module's
    # own docstring says the two measures "fail differently"; for a discrete-trade family the
    # exposure measure does not discriminate, and applying it anyway is a FALSE gate that
    # would report a miscalibrated instrument as a research finding.
    #
    # Callers passing nothing are unaffected. A caller that passes a floor must say what it
    # is relative to -- feature 013 uses half the incumbent's exposure on the SAME panel, so
    # "materially less participating than production" is still caught.
    exposure_floor: float | None = None

    @property
    def _floor(self) -> float:
        return MIN_EXPOSURE if self.exposure_floor is None else self.exposure_floor

    @property
    def degenerate(self) -> bool:
        # NaN-safe: `nan < floor` is False, so a NaN exposure would pass SILENTLY — the exact
        # failure this floor exists to catch.
        if not (self.mean_exposure == self.mean_exposure):
            return True
        if self.mean_exposure < self._floor:
            return True
        if self.trades_per_year is not None and self.trades_per_year < MIN_TRADES_PER_YEAR:
            return True
        return False

    @property
    def reason(self) -> str | None:
        if not (self.mean_exposure == self.mean_exposure):
            return "mean exposure is NaN — no usable panel"
        if self.mean_exposure < self._floor:
            return (f"mean exposure {self.mean_exposure:.4f} below the {self._floor:.4f} "
                    f"participation floor")
        if self.trades_per_year is not None and self.trades_per_year < MIN_TRADES_PER_YEAR:
            return (f"{self.trades_per_year:.1f} trades/yr below the "
                    f"{MIN_TRADES_PER_YEAR} floor — one position is not a strategy")
        return None


def compare(arm: Arm, control: Arm, do_nothing: float = 0.0,
            purpose: str = "acceptance") -> dict:
    """The ONLY sanctioned way to turn two arms into a verdict.

    Refuses to emit a performance verdict when EITHER side is degenerate. The raw delta is
    retained rather than discarded -- suppressing it would hide the evidence -- but it is
    labelled so it cannot be quoted as a result.
    """
    raw = arm.value - control.value
    bad = [a for a in (arm, control) if a.degenerate]
    if bad:
        return {"admissible": False,
                "verdict": "INCONCLUSIVE — DEGENERATE " +
                           ("CONTROL" if control.degenerate else "ARM"),
                "degenerate": [{"name": a.name, "reason": a.reason} for a in bad],
                "raw_delta": raw,
                "raw_delta_is_not_a_result":
                    "either side abstains; a delta against abstention measures the "
                    "comparator, not the arm",
                "arm": arm.name, "control": control.name}
    # SECOND FAILURE MODE, and the abstention floor above cannot catch it: a comparator
    # that PARTICIPATES FULLY and simply loses. Brief 02's P&L control ran at exposure
    # 0.441 and 1869 trades/yr -- not degenerate by any measure -- while losing 52% in a
    # market that ROSE 72%. Every arm scored a large positive delta against it, and 92% of
    # the headline advantage turned out to be the gap to a broken comparator.
    #
    # The test is absolute, not relative: if the arm cannot beat DOING NOTHING, then beating
    # the control says something about the control, not the arm.
    # PURPOSE MATTERS, and conflating the two is its own error.
    #
    #   "acceptance"  — is this deployable? Then it must beat CASH. A comparison between
    #                   two loss-makers cannot support a claim of value, however large the
    #                   delta: "loses less than the control" is not a strategy.
    #   "directional" — which of two approaches is better? Legitimate research even when
    #                   both lose (brief 04 asks exactly this: does sampling beat smoothing?).
    #                   Permitted, but STAMPED, so it can never be quoted as performance.
    #
    # The degeneracy floor above applies to BOTH purposes: an abstaining policy answers
    # neither question.
    raw_vs_nothing = arm.value - do_nothing
    if raw_vs_nothing <= 0 and purpose == "directional":
        return {"admissible": True, "verdict": "DIRECTIONAL ONLY — BOTH SIDES LOSE",
                "delta": raw, "delta_vs_do_nothing": raw_vs_nothing,
                "not_a_performance_claim":
                    "both policies lose to cash; this ranks two failures and cannot "
                    "support a claim of value",
                "arm": arm.name, "control": control.name}
    if raw_vs_nothing <= 0:
        return {"admissible": False,
                "verdict": "INCONCLUSIVE — ARM LOSES TO DO NOTHING",
                "delta_vs_control": raw,
                "delta_vs_do_nothing": raw_vs_nothing,
                "raw_delta_is_not_a_result":
                    "the arm is worse than holding cash; a positive delta against the "
                    "control measures the control's losses, not the arm's value",
                "arm": arm.name, "control": control.name}

    return {"admissible": True, "verdict": None, "delta": raw,
            "delta_vs_do_nothing": raw_vs_nothing,
            "arm": arm.name, "control": control.name}


def require_admissible(arm: Arm, control: Arm) -> float:
    """Delta, or raise. For call sites that should never see a degenerate comparison."""
    r = compare(arm, control)
    if not r["admissible"]:
        raise ValueError(f"{r['verdict']}: " +
                         "; ".join(d["reason"] for d in r["degenerate"]))
    return r["delta"]

"""035 T023: train survivor models, one fold at a time (research R9).

The 032 driver (research/candidate_training/train.py) on the survivor's training plan
(`plan_ni_<t>.json`; the sizer twin shares the model key). Early folds f00-f07 by default.
Refuses a tag that is not a survivor, any late fold unless verdicts.json names the tag as the S2
winner, and (the driver's own check) while another chain runs.

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.train --tag MS_VR [--folds f00] [--seed 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.new_inputs import arms as A


HERE = Path(__file__).resolve().parent
EARLY = tuple(f"f{i:02d}" for i in range(8))


def refusals(tag: str, folds: set[str], survivors: list, verdicts: dict | None) -> list[str]:
    out = []
    if tag not in survivors:
        out.append(f"{tag} is not a survivor ({survivors})")
    late = sorted(f for f in folds if f not in EARLY)
    if late and (verdicts or {}).get("s2_winner", {}).get("tag") != tag:
        out.append(f"{tag}: late folds {late} refused -- not the S2 winner in verdicts.json")
    return out


def run(tag: str, folds: set[str] | None, seed: int = 1, *, driver=None) -> list[dict]:
    folds = set(folds) if folds else set(EARLY)
    vp = HERE / "verdicts.json"
    bad = refusals(tag, folds, A.survivors(), json.loads(vp.read_text()) if vp.is_file() else None)
    if bad:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(bad))
    if driver is None:
        from research.candidate_training import train as driver
    key = f"NI_{tag}"
    driver.PLANS[key] = A.plan_tag(tag)  # runtime registration only
    if "research.new_inputs.train " not in driver.BUSY:
        driver.BUSY = (*tuple(driver.BUSY), "research.new_inputs.train ")
    return driver.run(key, folds, seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", help="comma-separated fold ids; default f00-f07")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(a.tag, set(a.folds.split(",")) if a.folds else None, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

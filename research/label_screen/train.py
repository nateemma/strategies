"""Train label-screen candidates, one fold at a time (034 T006, research R5).

The 032 driver (research/candidate_training/train.py) pointed at a candidate's training plan
(`plan_ls_<tag>_es.json`; the sizer twin shares the model key). Early folds f00-f07 by default.
Refuses:
  * a tag that is not trained (CONTROL, GFM reuse existing weights);
  * a tag whose S0a status in audit.json is not PASS;
  * any late fold (f08-f14) unless verdicts.json records this tag as the S2 winner;
  * while another chain runs (the driver's own check).

    PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.train --tag FMAE [--folds f00] [--seed 2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.label_screen import candidates as C


HERE = Path(__file__).resolve().parent
EARLY = tuple(f"f{i:02d}" for i in range(8))


def refusals(tag: str, folds: set[str], audit: dict | None, verdicts: dict | None) -> list[str]:
    out = []
    c = C.BY_TAG.get(tag)
    if c is None or c.order is None:
        return [f"{tag} is not a trained candidate"]
    st = ((audit or {}).get("candidates", {}).get(tag) or {}).get("status")
    if st != "PASS":
        out.append(f"{tag}: S0a status is {st!r}, not PASS (run research.label_screen.audit)")
    late = sorted(f for f in folds if f not in EARLY)
    if late and (verdicts or {}).get("s2_winner", {}).get("tag") != tag:
        out.append(f"{tag}: late folds {late} refused -- not the S2 winner in verdicts.json")
    return out


def _read(name: str) -> dict | None:
    p = HERE / name
    return json.loads(p.read_text()) if p.is_file() else None


def run(tag: str, folds: set[str] | None, seed: int = 1, *, driver=None) -> list[dict]:
    folds = set(folds) if folds else set(EARLY)
    bad = refusals(tag, folds, _read("audit.json"), _read("verdicts.json"))
    if bad:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(bad))
    if driver is None:
        from research.candidate_training import train as driver
    key = f"LS_{tag}"
    driver.PLANS[key] = C.plan_tag(C.BY_TAG[tag], "es")  # runtime registration only
    if "research.label_screen.train " not in driver.BUSY:
        driver.BUSY = tuple(driver.BUSY) + ("research.label_screen.train ",)
    return driver.run(key, folds, seed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, choices=[c.tag for c in C.TRAINED])
    ap.add_argument("--folds", help="comma-separated fold ids; default f00-f07")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(a.tag, set(a.folds.split(",")) if a.folds else None, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

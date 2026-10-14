"""Build the candidate-only training arms (032 T012; one-off generator).

Per model: `NNNC/NNNC_CO_<M>.py` = CandidateTrainingMixin + the 24 h control's base class, its
`.json` (copied from the base, strategy_name set), and `evaluation/folds/plan_co_<m>.json` copied
from the control plan: same folds, params snapshot and config; new classes, model dirs and arm
tags; no thresholds (they are picked after training, research R7).

    PYTHONPATH=. ../../.venv/bin/python -m research.candidate_training.build_arms
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
FOLDS = STRAT / "evaluation" / "folds"
CONTROLS = {"ARO": "wf_aro_g24", "GBR": "fs_gbr_gr"}

TMPL = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Candidate-only training for {m} (feature 032; research/candidate_training/build_arms.py;
GENERATED): the model is trained only on windows whose last bar passes the gbb rule, and enters
only on such bars. Everything else is `{base}` (control `{ctl}`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _candidate_training import CandidateTrainingMixin
from {base} import {base}


class {name}(CandidateTrainingMixin, {base}):
    pass
'''


def build(m: str, ctl: str) -> tuple[str, str]:
    plan = json.loads((FOLDS / f"plan_{ctl}.json").read_text())
    base, name, tag = plan["base_strategy"], f"NNNC_CO_{m}", f"co_{m.lower()}"
    (STRAT / "NNNC" / f"{name}.py").write_text(TMPL.format(m=m, base=base, ctl=ctl, name=name))
    bj = json.loads((STRAT / "NNNC" / f"{base}.json").read_text())
    bj["strategy_name"] = name
    (STRAT / "NNNC" / f"{name}.json").write_text(json.dumps(bj, indent=4) + "\n")
    new = copy.deepcopy(plan)
    new["run_id"], new["base_strategy"] = tag, name
    new.pop("thresholds", None)
    for f in new["folds"]:
        f["strategy_class"] = f"{name}_{f['fold_id']}"
        f["model_dir"] = f"user_data/strategies/saved_data/{name}_{f['fold_id']}"
        f["arm_tag"] = f"{tag}_{f['fold_id']}"
        f["status"] = "planned"
    new["co_note"] = f"032 candidate-only training; control {ctl}; thresholds picked after training"
    (FOLDS / f"plan_{tag}.json").write_text(json.dumps(new, indent=1) + "\n")
    return name, tag


def main() -> int:
    for m, ctl in CONTROLS.items():
        name, tag = build(m, ctl)
        print(m, name, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

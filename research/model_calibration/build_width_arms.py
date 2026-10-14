"""Build the width-book backtest arms (one-off generator; research/model_calibration).

Per model: `NNNC/NNNC_W_<M>.py` = WidthBookMixin + the 24 h control's base strategy, with
WIDTH_CUTS from width_twin.json (fold i uses fold i-1's matched cut; f00 its own -- f00 is NOT
causal), its .json, and `plan_w_<m>.json` copied from the control plan (same folds, snapshot,
model dirs, thresholds).

    PYTHONPATH=. ../../.venv/bin/python -m research.model_calibration.build_width_arms
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FOLDS = STRAT / "evaluation" / "folds"
CONTROLS = {"ARO": "wf_aro_g24", "GBR": "fs_gbr_gr", "GFM": "fs_gfm_gr", "MLX": "fs_mlx_gr"}

TMPL = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Width book for {m} (research/model_calibration/build_width_arms.py; GENERATED): the gbb rule +
bb_width >= this fold's cut instead of the model's prediction; everything else is `{base}`
(control `{ctl}`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from _width_book import WidthBookMixin
from {base} import {base}


class {name}(WidthBookMixin, {base}):
    WIDTH_CUTS = {cuts}
'''


def main() -> int:
    tw = json.loads((HERE / "width_twin.json").read_text())["models"]
    for m, ctl in CONTROLS.items():
        plan = json.loads((FOLDS / f"plan_{ctl}.json").read_text())
        base, name, tag = plan["base_strategy"], f"NNNC_W_{m}", f"w_{m.lower()}"
        cuts, prev = {}, None
        for f in plan["folds"]:
            own = tw[m]["cuts"][f["fold_id"]]
            cuts[f["train_timerange"]] = float(own if prev is None else prev)
            prev = own
        body = "{\n" + "".join(f"        {k!r}: {v!r},\n" for k, v in cuts.items()) + "    }"
        (STRAT / "NNNC" / f"{name}.py").write_text(TMPL.format(m=m, base=base, ctl=ctl,
                                                               name=name, cuts=body))
        bj = json.loads((STRAT / "NNNC" / f"{base}.json").read_text())
        bj["strategy_name"] = name
        (STRAT / "NNNC" / f"{name}.json").write_text(json.dumps(bj, indent=4) + "\n")
        new = copy.deepcopy(plan)
        new["run_id"], new["base_strategy"] = tag, name
        for f in new["folds"]:
            f["strategy_class"] = f"{name}_{f['fold_id']}"
            f["arm_tag"] = f"{tag}_{f['fold_id']}"
            f["status"] = "planned"
        new["width_note"] = f"model-free width book; control {ctl}"
        (FOLDS / f"plan_{tag}.json").write_text(json.dumps(new, indent=1) + "\n")
        print(m, name, tag, "cuts", list(cuts.values())[:3], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

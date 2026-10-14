"""Build the width-relative sizer arms (033 T014; one-off generator; contract C2).

Per model: `NNNC/NNNC_WR_<M>.py` = WidthSizerMixin + WidthBookMixin + WidthRelativeMixin + the 24 h
control's base class, with the width book's own WIDTH_CUTS (read from `NNNC_W_<M>`, so the entries
are exactly the `w_<m>` arm's); its `.json` copied from the base. Two plans copied from the control
plan (same folds and config; new classes, model dirs and arm tags; no thresholds yet):

  * `plan_wr_<m>_sz.json`  -- model exits OFF (the control's params snapshot);
  * `plan_wr_<m>_szx.json` -- model exits ON: a copy of the snapshot with
    `sell.enable_exit_signal = true` (`params_wr_<m>_szx.json`), the exit gate unchanged.

Both resolve to the same model key per fold, so the weights are trained once.

    PYTHONPATH=. ../../.venv/bin/python -m research.width_relative.build_arms
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
FOLDS = STRAT / "evaluation" / "folds"
CONTROLS = {"ARO": "wf_aro_g24", "GBR": "fs_gbr_gr"}

TMPL = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Width-relative sizer for {m} (feature 033; research/width_relative/build_arms.py; GENERATED):
the width book's entries (cuts as `NNNC_W_{m}`), the stake sized by a model trained on the
width-relative label. Everything else is `{base}` (control `{ctl}`). Held-pair add off."""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from _width_book import WidthBookMixin
from _width_relative import WidthRelativeMixin
from _width_sizer import WidthSizerMixin
from {base} import {base}


class {name}(WidthSizerMixin, WidthBookMixin, WidthRelativeMixin, {base}):
    WIDTH_CUTS = {cuts}
'''


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def width_cuts(m: str) -> dict:
    sys.path[:0] = [str(STRAT), str(STRAT / "NNNC")]
    from evaluation.fingerprint import load_strategy_class

    return dict(load_strategy_class(f"NNNC_W_{m}").WIDTH_CUTS)


def build(m: str, ctl: str) -> dict:
    plan = json.loads((FOLDS / f"plan_{ctl}.json").read_text())
    base, name = plan["base_strategy"], f"NNNC_WR_{m}"
    cuts = width_cuts(m)
    body = "{\n" + "".join(f"        {k!r}: {v!r},\n" for k, v in cuts.items()) + "    }"
    (STRAT / "NNNC" / f"{name}.py").write_text(TMPL.format(m=m, base=base, ctl=ctl, name=name,
                                                           cuts=body))
    bj = json.loads((STRAT / "NNNC" / f"{base}.json").read_text())
    bj["strategy_name"] = name
    (STRAT / "NNNC" / f"{name}.json").write_text(json.dumps(bj, indent=4) + "\n")

    snap = STRAT.parent.parent / plan["params_snapshot"]
    snapx = FOLDS / f"params_wr_{m.lower()}_szx.json"
    sx = json.loads(snap.read_text())
    sx["params"].setdefault("sell", {})["enable_exit_signal"] = True
    snapx.write_text(json.dumps(sx, indent=4) + "\n")

    out = {}
    for variant, exits in (("sz", False), ("szx", True)):
        tag = f"wr_{m.lower()}_{variant}"
        cls = f"{name}_{variant.upper()}"
        new = copy.deepcopy(plan)
        new["run_id"], new["base_strategy"] = tag, name
        new.pop("thresholds", None)
        if exits:
            new["params_snapshot"] = str(snapx.relative_to(STRAT.parent.parent))
            new["params_sha"] = _sha(snapx)
        for f in new["folds"]:
            f["strategy_class"] = f"{cls}_{f['fold_id']}"
            f["model_dir"] = f"user_data/strategies/saved_data/{name}_{f['fold_id']}"
            f["arm_tag"] = f"{tag}_{f['fold_id']}"
            f["status"] = "planned"
        new["wr_note"] = (f"033 width-relative sizer, model exits {'ON' if exits else 'OFF'}; "
                          f"control {ctl}; width book w_{m.lower()}; thresholds picked after "
                          f"training (research/width_relative/pick.py)")
        (FOLDS / f"plan_{tag}.json").write_text(json.dumps(new, indent=1) + "\n")
        out[variant] = tag
    return out


def main() -> int:
    for m, ctl in CONTROLS.items():
        print(m, build(m, ctl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

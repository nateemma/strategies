"""T053 follow-up DIAGNOSTIC: how much do the width-book cuts (and w_aro) move with the ARO model's SEED alone?

The width book's per-fold bb_width cut is matched to the ARO model book's own entry count on the previous
fold (research/model_calibration/width_twin.py). The purged rerun moved w_aro 51.44 -> 37.83 entirely
through those cuts (w_aro_pw_lc). This measures the same coupling on the LEGACY windows with ARO's seed-2
weights (cs_aro_s2's), so the purge effect can be read against seed noise:

  pick --fold fNN   seed-2 model book threshold at the incumbent target rate (0.014735), one fold per
                    process                                    -> cut_noise/thr_s2_fNN.json
  signals --fold    the seed-2 model book's test-window signal table (fold class NNNC_ARO_fNN_s2 with
                    its threshold)                              -> signals/ARO_S2_fNN.parquet
  cuts              matched + causal cuts (rebuild._matched_cut), keyed by the legacy timeranges
                                                                -> cut_noise/width_cuts_aro_s2.json
  build             NNNC_W_ARO_S2C (the seed-1 width book with the seed-2 cuts; exits are model-free in
                    effect: w_aro_pw_lc = w_aro exactly) and plan_w_aro_s2c.json

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.train_window.cut_noise pick --fold f00
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
ROOT = STRAT.parents[1]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]
HERE = Path(__file__).resolve().parent
OUT = HERE / "cut_noise"
FOLDS = STRAT / "evaluation" / "folds"
TARGET_RATE = 0.014735
SRC_PLAN = "wf_aro_g24"


def _plan() -> dict:
    return json.loads((FOLDS / f"plan_{SRC_PLAN}.json").read_text())


def _fold_s2(fid: str) -> dict:
    f = next(x for x in _plan()["folds"] if x["fold_id"] == fid)
    return dict(f, strategy_class=f"{f['strategy_class']}_s2", model_dir=f"{f['model_dir']}_s2",
                arm_tag=f"{f['arm_tag']}_s2")


def _params(plan: dict) -> dict:
    return json.loads((ROOT / plan["params_snapshot"]).read_text())


def pick(fid: str) -> int:
    import run_folds as RF

    plan, f = _plan(), _fold_s2(fid)
    RF.generate_fold(f, plan["base_strategy"], plan["family"], _params(plan), seed=2)
    if not RF.fold_has_model(f, plan["base_strategy"], plan["config"], 2):
        raise SystemExit(f"{fid}: no seed-2 ARO weights for {f['strategy_class']} (strict: never trains)")
    OUT.mkdir(exist_ok=True)
    tmp = OUT / f"plan_s2_{fid}.json"
    tmp.write_text(json.dumps({**plan, "folds": [f]}))
    out = OUT / f"thr_s2_{fid}.json"
    subprocess.run([sys.executable, str(STRAT / "evaluation" / "pick_threshold.py"), "--plan", str(tmp),
                    "--fold", fid, "--target-rate", str(TARGET_RATE), "--out", str(out)],
                   cwd=ROOT, check=True, env={"PYTHONPATH": str(STRAT), "FT_MODEL_EVAL_STRICT": "1",
                                              **{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}})
    print(json.dumps({"fold": fid, **json.loads(out.read_text())}))
    return 0


def signals(fid: str) -> int:
    import run_folds as RF
    from research.nnnc_framework.signals import fold_table

    plan, f = _plan(), _fold_s2(fid)
    thr = json.loads((OUT / f"thr_s2_{fid}.json").read_text())[fid]
    RF.generate_fold(f, plan["base_strategy"], plan["family"], _params(plan), threshold=thr, seed=2)
    t = fold_table(f["strategy_class"], f["test_timerange"])
    (HERE / "signals").mkdir(exist_ok=True)
    t.to_parquet(HERE / "signals" / f"ARO_S2_{fid}.parquet")
    print(json.dumps({"fold": fid, "threshold": thr, "entries": int((t["enter_long"] == 1).sum())}))
    return 0


def cuts() -> int:
    from research.train_window.rebuild import _buy, _matched_cut

    plan = _plan()
    matched = {f["fold_id"]: _matched_cut(pd.read_parquet(HERE / "signals" / f"ARO_S2_{f['fold_id']}.parquet"),
                                          _buy(f"{f['strategy_class']}_s2")) for f in plan["folds"]}
    causal = {f["train_timerange"]: matched[plan["folds"][i - 1]["fold_id"] if i else "f00"]
              for i, f in enumerate(plan["folds"])}
    (OUT / "width_cuts_aro_s2.json").write_text(json.dumps({"label": "DIAGNOSTIC: ARO seed-2 matched cuts, legacy windows",
                                                            "matched": matched, "causal": causal}, indent=1))
    print(json.dumps(matched, indent=1))
    return 0


def build() -> int:
    from research.nnmt.build_arms import key_of

    causal = json.loads((OUT / "width_cuts_aro_s2.json").read_text())["causal"]
    (STRAT / "NNNC" / "NNNC_W_ARO_S2C.py").write_text(f'''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""T053 DIAGNOSTIC (research/train_window/cut_noise.py; GENERATED): the w_aro width book with the cuts matched to
ARO's SEED-2 model book (legacy windows). Against w_aro it measures the cut coupling's seed noise."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_W_ARO import NNNC_W_ARO


class NNNC_W_ARO_S2C(NNNC_W_ARO):
    WIDTH_CUTS = {json.dumps(causal, indent=8, sort_keys=True)}
''')
    sys.path.insert(0, str(STRAT / "NNNC"))
    assert key_of("NNNC_W_ARO_S2C") == key_of("NNNC_W_ARO")
    p = json.loads((FOLDS / "plan_w_aro.json").read_text())
    p.update(run_id="w_aro_s2c", base_strategy="NNNC_W_ARO_S2C", note="T053 diagnostic: w_aro with ARO seed-2 cuts")
    for f in p["folds"]:
        f.update(strategy_class=f["strategy_class"].replace("NNNC_W_ARO", "NNNC_W_ARO_S2C"),
                 model_dir=f["model_dir"], arm_tag=f["arm_tag"].replace("w_aro", "w_aro_s2c"), status="planned")
    (FOLDS / "plan_w_aro_s2c.json").write_text(json.dumps(p, indent=1) + "\n")
    print("built NNNC_W_ARO_S2C, plan_w_aro_s2c.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pick", "signals", "cuts", "build"))
    ap.add_argument("--fold")
    a = ap.parse_args(argv)
    return {"pick": lambda: pick(a.fold), "signals": lambda: signals(a.fold), "cuts": cuts, "build": build}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())

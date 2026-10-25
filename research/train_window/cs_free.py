"""The ARO sizer (cs_free) and tr_scale (cs_free_tr) on the model-free w_free book
(research/train_window/preregistration_cs_free.json).

  build        NNNC_CS_FREE / NNNC_CS_FREE_TR (+ .json), plan_cs_free (no thresholds) / plan_cs_free_tr, model keys asserted
  copy-thr     cs_free's picked thresholds -> plan_cs_free_tr (after research.control_sizer.pick on every fold)
  judge        the pre-registered verdicts                    -> verdicts_cs_free.json

Thresholds: research.control_sizer.pick --plan plan_cs_free.json --fold fNN [--verify], one process per fold.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.train_window.cs_free build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]
HERE = Path(__file__).resolve().parent
FOLDS = STRAT / "evaluation" / "folds"
PREREG = HERE / "preregistration_cs_free.json"

CLASSES = {
    "NNNC_CS_FREE": ('''"""cs_aro on the model-free w_free book (research/train_window/cs_free.py; preregistration_cs_free.json):
the ARO p_buy sizer exactly as NNNC_CS_ARO, entries = NNNC_W_FREE's (trailing-90-day width quantile, no model)."""''',
                     "from NNNC_CS_ARO import NNNC_CS_ARO\nfrom NNNC_W_FREE import NNNC_W_FREE\n",
                     "NNNC_CS_ARO", "    WIDTH_CUTS = NNNC_W_FREE.WIDTH_CUTS\n"),
    "NNNC_CS_FREE_TR": ('''"""cs_free + the 040 tr_scale exposure scaler, with NNNC_CS_ARO_TR's settings (preregistration_cs_free.json)."""''',
                        "from _transition_signal import TrScaleMixin\nfrom NNNC_CS_FREE import NNNC_CS_FREE\n",
                        "TrScaleMixin, NNNC_CS_FREE",
                        '    TR_UP_CELL = "up_pair_96"\n    TR_DOWN_CELL = "down_pair_672"\n    TR_SOURCE = "standin"\n'
                        '    TR_STANDIN_ROOT = "transition_v2"\n'),
}
PLANS = {"cs_free": "NNNC_CS_FREE", "cs_free_tr": "NNNC_CS_FREE_TR"}


def _plan(tag: str) -> dict:
    return json.loads((FOLDS / f"plan_{tag}.json").read_text())


def build() -> int:
    from research.nnmt.build_arms import key_of

    for name, (doc, imports, bases, body) in CLASSES.items():
        (STRAT / "NNNC" / f"{name}.py").write_text(
            f"# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301\n# type: ignore\n{doc}\n\nimport sys\n"
            f"from pathlib import Path\n\nsys.path.append(str(Path(__file__).parent))\n\n{imports}\n\nclass {name}({bases}):\n{body}")
        bj = json.loads((STRAT / "NNNC" / "NNNC_CS_ARO.json").read_text())
        bj["strategy_name"] = name
        (STRAT / "NNNC" / f"{name}.json").write_text(json.dumps(bj, indent=4) + "\n")
    src = _plan("cs_aro")
    for tag, name in PLANS.items():
        p = {k: v for k, v in src.items() if k not in ("thresholds", "threshold_picks", "note")}
        p.update(run_id=tag, base_strategy=name, cs_note=f"{tag}: preregistration_cs_free.json; thresholds re-picked on w_free's runs")
        p["folds"] = [dict(f, strategy_class=f"{name}_{f['fold_id']}", model_dir=f"user_data/strategies/saved_data/{name}_{f['fold_id']}",
                           arm_tag=f"{tag}_{f['fold_id']}", status="planned") for f in src["folds"]]
        (FOLDS / f"plan_{tag}.json").write_text(json.dumps(p, indent=1) + "\n")
    sys.path.insert(0, str(STRAT / "NNNC"))
    keys = {n: key_of(n) for n in ("NNNC_CS_ARO", *CLASSES)}
    if len(set(keys.values())) != 1:
        raise SystemExit(f"REFUSED: model keys differ: {keys}")
    import importlib

    if importlib.import_module("NNNC_CS_FREE").NNNC_CS_FREE.WIDTH_CUTS != importlib.import_module("NNNC_W_FREE").NNNC_W_FREE.WIDTH_CUTS:
        raise SystemExit("REFUSED: cs_free's cuts are not w_free's")
    print(json.dumps({"built": list(CLASSES), "key": keys["NNNC_CS_ARO"]}))
    return 0


def copy_thr() -> int:
    src, p = _plan("cs_free"), _plan("cs_free_tr")
    if len(src.get("thresholds", {})) != 15:
        raise SystemExit("REFUSED: cs_free has no 15-fold thresholds yet")
    p["thresholds"], p["threshold_picks"] = src["thresholds"], src["threshold_picks"]
    (FOLDS / "plan_cs_free_tr.json").write_text(json.dumps(p, indent=1) + "\n")
    print("copied", len(p["thresholds"]), "thresholds")
    return 0


def _trade_keys(tag: str, fid: str) -> set:
    from evaluation.pool import pool

    return {(t["pair"], t["open_date"], t["close_date"]) for t in pool(_plan(tag))["trades"] if t["fold_id"] == fid}


def judge() -> int:
    from evaluation.pool import pool
    from research.nnnc_switches.judge import judge_one
    from research.pair_triggers import judge as PJ

    pr = json.loads(PREREG.read_text())
    PJ.PREREG = PREREG
    refused = PJ.refusals({**pr, "arms": list(pr["arms"])})
    if refused:
        raise SystemExit(f"REFUSED: {refused}")
    out = {"label": pr["label"], "instrument": {}, "verdicts": {}, "beside": {}}
    for arm, ref in (("cs_free", "w_free"), ("cs_free_tr", "cs_free")):
        diff = [f for f in pr["folds"] if _trade_keys(arm, f) != _trade_keys(ref, f)]
        mean_stake = lambda tag: sum(t["stake_amount"] for t in pool(_plan(tag))["trades"]) / len(pool(_plan(tag))["trades"])  # noqa: E731
        sa, sr = mean_stake(arm), mean_stake(ref)
        out["instrument"][arm] = {"vs": ref, "folds_trade_set_differs": diff, "stake_ratio": round(float(sa / sr), 4),
                                  "ok": len(diff) <= 2 and (arm != "cs_free" or abs(sa / sr - 1) <= 0.10)}
    if not all(v["ok"] for v in out["instrument"].values()):
        out["verdicts"] = "INCONCLUSIVE (instrument failed; no P&L read)"
    else:
        pairs = {"cs_free": [("w_free",), ("wf_aro_g24",)], "cs_free_tr": [("cs_free",)]}
        for arm, ctls in pairs.items():
            vs = {c[0]: judge_one(pr, arm, c[0]) for c in ctls}
            gains = [g for v in vs.values() for g in (v.get("early_gain_pp"), v.get("late_gain_pp")) if g is not None]
            all_pass = all(v["verdict"] == "PASS" for v in vs.values())
            small = all_pass and any(g < pr["rule"]["small_margin_pp"] for g in gains)
            out["verdicts"][arm] = {"vs": vs, "decision": ("INCONCLUSIVE (small margin: seed-2 repeat)" if small else
                                                           "PASS (PROVISIONAL)" if all_pass else
                                                           "FAIL" if any(v["verdict"] == "FAIL" for v in vs.values()) else "INCONCLUSIVE")}
        for arm in ("cs_free", "cs_free_tr"):
            out["beside"][arm] = {ref: judge_one(pr, arm, ref) for ref in ("cs_aro", "cs_aro_tr")}
    (HERE / "verdicts_cs_free.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"instrument": out["instrument"],
                      "decisions": {a: v["decision"] for a, v in out["verdicts"].items()} if isinstance(out["verdicts"], dict) else out["verdicts"]},
                     indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "copy-thr", "judge"))
    a = ap.parse_args(argv)
    return {"build": build, "copy-thr": copy_thr, "judge": judge}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the stage-2 arms of the NNNC framework study from `selection.json` (one-off generator).

For each candidate and model: an arm class `NNNC_FS_<M>_<KEY>` subclassing the model's base class
(weights resolve to the base's per-fold model key -- 030, no training), its .json (the base's,
renamed), a params snapshot when the candidate changes a .json value, and a plan derived from the
model's 030-baseline control plan (same folds, thresholds, model_dir; new classes and arm tags).

    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.build_arms
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROOT_REL = "user_data/strategies/"
MODELS_R1 = {"ARO": ("wf_aro_b030", "NNNC_A_Raw_Open"), "GBR": ("sw_gbr_ctl_b030", "NNNC_GbbRaw_Open"),
             "GFM": ("sw_gfm_ctl_b030", "NNNC_GbbFwdMae_Open"), "MLX": ("sw_mlx_ctl_b030", "NNNC_MLX")}
#: Round 2: controls carry the adopted 24 h grace (their plans' snapshots pin it).
MODELS_R2 = {"ARO": ("wf_aro_g24", "NNNC_A_Raw_Open"), "GBR": ("fs_gbr_gr", "NNNC_GbbRaw_Open"),
             "GFM": ("fs_gfm_gr", "NNNC_GbbFwdMae_Open"), "MLX": ("fs_mlx_gr", "NNNC_MLX")}
ROUND = os.environ.get("NNNC_FS_ROUND", "1")
SUFFIX = "" if ROUND == "1" else f"_r{ROUND}"
MODELS = {"1": MODELS_R1, "2": MODELS_R2}[ROUND]
#: class / arm-tag prefixes: round 1 NNNC_FS_ / fs_, round 2 NNNC_FS2_ / fs2_
CLS_PREFIX, TAG_PREFIX = ("NNNC_FS", "fs") if ROUND == "1" else (f"NNNC_FS{ROUND}", f"fs{ROUND}")

#: lever -> (short key, how it is declared). Kinds: "family" (FAMILY_EXIT_OVERRIDES.reversion),
#: "attrs" (class attributes, scaled from the base's), "sell"/"roi" (.json snapshot), "thr" (plan).
DECL = {
    "exit:grace_hours": ("GR", "sell", lambda v: {"stoploss_grace_hours": float(v)}),
    "exit:take_profit": ("TP", "sell", lambda v: {"cexit_take_profit": float(v)}),
    "exit:tp_atr_k": ("TK", "family", lambda v: {"take_profit_atr_mult": float(v)}),
    "exit:stop_scale": ("SS", "attrs", lambda v: float(v)),
    "exit:unclog_1d_hours": ("U1", "family", lambda v: {"unclog_1d_hours": float(v)}),
    "entry:thr_shift": ("TH", "thr", lambda v: float(v)),
    # round 2 additions (any lever the screens price can be declared)
    "exit:grace_level": ("GL", "sell", lambda v: {"stoploss_grace_level": float(v)}),
    "exit:max_hold_days": ("MH", "sell", lambda v: {"cexit_max_days": int(v)}),
    "exit:roi": ("RO", "roi", lambda v: {"0": 10.0 if v == "off" else float(v)}),
    "exit:vol_confirm": ("VC", "family", lambda v: {"use_volume_confirmation_stoploss": v == "True"}),
    "exit:unclog_12h_hours": ("U12", "family", lambda v: {"unclog_12h_hours": float(v)}),
    "exit:metric_overbought": ("MO", "family", lambda v: {"metric_overbought_threshold": 10.0}),
    "entry:guard_on": ("GON", "buy", lambda v: {
        "atr_pct": {"entry_atr_pct": 0.006}, "guard": {"entry_guard_threshold": 0.3},
        "close_norm": {"entry_close_norm_threshold": -0.1}, "adx": {"entry_adx_threshold": 38.0},
        "bb_width": {"entry_bb_width_threshold": 0.06}, "rvol": {"entry_rvol_threshold": 2.0}}[v]),
}

CLASS_TMPL = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNNC framework study, stage 2 (research/nnnc_framework/): {lever} = {value} on `{base}`.

One lever changed, nothing else. Weights: the base's per-fold models, found by model key (030).
Pre-registered in research/nnnc_framework/preregistration.json.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from {base} import {base}


class {name}({base}):
{body}
'''


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


def build(c: dict, m: str) -> dict:
    ctl_tag, base = MODELS[m]
    key, kind, fn = DECL[c["lever"]]
    name = f"{CLS_PREFIX}_{m}_{key}"
    tag = f"{TAG_PREFIX}_{m.lower()}_{key.lower()}"
    plan = json.loads((STRAT / "evaluation/folds" / f"plan_{ctl_tag}.json").read_text())
    v = fn(c["value"])
    body = "    pass"
    if kind == "family":
        body = f"    FAMILY_EXIT_OVERRIDES = {{\"reversion\": {v!r}}}"
    elif kind == "attrs":
        import sys
        sys.path.insert(0, str(STRAT))
        from evaluation.fingerprint import load_strategy_class
        b = load_strategy_class(base)
        body = "\n".join(f"    {a} = {getattr(b, a) * v!r}" for a in
                         ("atr_stoploss_multiplier", "atr_stoploss_floor", "atr_stoploss_cap"))
    (STRAT / "NNNC" / f"{name}.py").write_text(CLASS_TMPL.format(
        lever=c["lever"], value=c["value"], base=base, name=name, body=body))
    bj = json.loads((STRAT / "NNNC" / f"{base}.json").read_text())
    bj["strategy_name"] = name
    (STRAT / "NNNC" / f"{name}.json").write_text(json.dumps(bj, indent=4) + "\n")
    new = copy.deepcopy(plan)
    new["run_id"], new["base_strategy"] = tag, name
    snap_path = STRAT.parents[1] / plan["params_snapshot"]
    snap = json.loads(snap_path.read_text())
    if kind in ("sell", "buy", "roi"):
        if kind == "roi":
            snap["params"]["roi"] = v
        else:
            snap["params"][kind].update(v)
        sp = STRAT / "evaluation/folds" / f"params_fs_{name}.json"
        blob = (json.dumps(snap, indent=4) + "\n").encode()
        sp.write_bytes(blob)
        new["params_snapshot"], new["params_sha"] = ROOT_REL + str(sp.relative_to(STRAT)), _sha(blob)
    if kind == "thr":
        base_thr = plan.get("thresholds") or {
            f["fold_id"]: float(snap["params"]["buy"]["prediction_threshold"]) for f in plan["folds"]}
        new["thresholds"] = {k: x + v for k, x in base_thr.items()}
    for f in new["folds"]:
        f["strategy_class"] = f"{name}_{f['fold_id']}"
        f["arm_tag"] = f"{tag}_{f['fold_id']}"
        f["status"] = "planned"
    new["model_source_note"] = f"weights = {plan['folds'][0]['model_dir']} (by model key)"
    (STRAT / "evaluation/folds" / f"plan_{tag}.json").write_text(json.dumps(new, indent=1) + "\n")
    return {"class": name, "arm": tag, "control": ctl_tag}


def main() -> int:
    sel = json.loads((HERE / f"selection{SUFFIX}.json").read_text())
    out = []
    for c in sel["candidates"]:
        built = {m: build(c, m) for m in MODELS}
        out.append({"name": DECL[c["lever"]][0], "lever": c["lever"], "value": c["value"],
                    "early_sum_pp": c["early_sum_pp"],
                    "arms": {m: b["arm"] for m, b in built.items()},
                    "classes": {m: b["class"] for m, b in built.items()},
                    "controls": {m: b["control"] for m, b in built.items()}})
    (HERE / f"arms{SUFFIX}.json").write_text(json.dumps(out, indent=1))
    print("\n".join(f"{o['name']}: {list(o['arms'].values())}" for o in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

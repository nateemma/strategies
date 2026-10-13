"""NNNC framework study, stage 1: how the exit/stop stack treats MODEL trades (no backtests).

On the 030-baseline control trades of the four NNNC models (arm D/ARO, GBR, GFM, MLX) this
  1. checks that 027's faithful replay (`research.exit_coupling.paths`) reproduces the recorded
     exits (>= 90% on reason AND bar per model, else no lever result for that model);
  2. describes the exits: count and summed pp by exit reason, per half; 027's (a)-(d) path
     measurements (stop exits that later reach the label target, killed winners, time to target);
  3. replays every trade under ONE exit/stop lever change at a time (the levers and grids below)
     and reports the pp delta against the NULL replay, EARLY HALF (f00-f07) ONLY.

Only the early half is computed for levers, on purpose: stage 2 pre-registers the candidates and
their values from this output, and the late half (f08-f14) is the out-of-sample test.
Replays ignore re-sequencing (a changed exit frees a pair / wallet slot earlier or later):
they choose candidates, walk-forward arms decide. PROVISIONAL (2023-26 halves only).

    PYTHONPATH=. ../../.venv/bin/python -m research.nnnc_framework.diagnose
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import feasibility as F  # noqa: E402
from research.exit_coupling import paths as P  # noqa: E402


HERE = Path(__file__).resolve().parent
FAMILY = "reversion"  # every model's label is reversion-family (gbb / gbb-path)
#: Round 1. NOTE: exit_setup reads `<prefix>_f00.json`, a GENERATED fold file; NNNC_ARO_f00.json held
#: 6.5 h when round 1 ran (committed diagnose.json) and 24 h since wf_aro_g24 regenerated it -- to
#: re-run round 1 exactly, regenerate the fold files from plan_wf_aro_b030 first.
MODELS_R1 = {  # model -> (030 baseline control tag, base class, per-fold class prefix)
    "ARO": ("wf_aro_b030", "NNNC_A_Raw_Open", "NNNC_ARO"),
    "GBR": ("sw_gbr_ctl_b030", "NNNC_GbbRaw_Open", "NNNC_GBR"),
    "GFM": ("sw_gfm_ctl_b030", "NNNC_GbbFwdMae_Open", "NNNC_GFM"),
    "MLX": ("sw_mlx_ctl_b030", "NNNC_MLX", "NNNC_WF"),
}
_JSON_FOR: dict = {}  # base class -> the per-fold .json that ran (sell/roi identical across folds)


def exit_setup(class_name: str):
    """027's `exit_setup`, plus the strategy's .json operating point: the sell params and ROI the
    walk-forward folds actually ran (027 read SimpleTrigger classes, whose values are class-level;
    NNNC's live in the .json -- without them the replay ran a 2.6 h grace and 21-day hold)."""
    cls, params, roi = _EXIT_SETUP(class_name)
    j = json.loads(_JSON_FOR[class_name].read_text())["params"]
    sell = j["sell"]
    params = dataclasses.replace(
        params, stoploss_grace_hours=float(sell["stoploss_grace_hours"]),
        stoploss_grace_level=float(sell["stoploss_grace_level"]),
        take_profit=float(sell["cexit_take_profit"]), max_hold_days=int(sell["cexit_max_days"]),
        cexit_enable_profit_checks=bool(sell["cexit_enable_profit_checks"]))
    roi_map = j.get("roi") or {}
    return cls, params, (float(roi_map["0"]) if "0" in roi_map else roi)


_EXIT_SETUP = F.exit_setup
F.exit_setup = exit_setup
OFF = 10.0  # a take-profit / threshold no trade reaches

#: One lever at a time. Each value is (ExitParams overrides, Rules overrides). The first value of
#: every lever is today's setting (delta 0 by construction, kept as a check).
LEVERS_R1 = {
    "take_profit": {v: ({"take_profit": v}, {}) for v in (0.022, 0.015, 0.03, 0.04, OFF)},
    "tp_atr_k": {"base": ({}, {})} | {k: ({}, {"take_profit_atr_mult": k})
                                      for k in (2.0, 2.5, 2.897, 3.5, 4.5)},
    "roi": {"base": ({}, {}), 0.03: ({"roi": 0.03}, {}), 0.08: ({"roi": 0.08}, {}),
            "off": ({"roi": None}, {})},
    "grace_hours": {6.5: ({}, {})} | {h: ({}, {"grace_hours": h}) for h in (3.0, 9.0, 12.0, 24.0)},
    "grace_level": {-0.27: ({}, {})} | {lv: ({}, {"grace_level": lv})
                                        for lv in (-0.10, -0.15, -0.20, -0.35)},
    "stop_scale": {1.0: ({}, {})} | {s: ({"_scale": s}, {}) for s in (0.75, 1.25, 1.5, 2.0)},
    "vol_confirm": {True: ({}, {}), False: ({"use_volume_confirmation_stoploss": False}, {})},
    "unclog_12h_hours": {"base": ({}, {})} | {h: ({"unclog_12h_hours": h}, {})
                                              for h in (6.0, 24.0, OFF * 100)},
    "unclog_1d_hours": {24.0: ({}, {})} | {h: ({"unclog_1d_hours": h}, {})
                                           for h in (12.0, 48.0, OFF * 100)},
    "max_hold_days": {10: ({}, {})} | {d: ({"max_hold_days": d}, {}) for d in (2, 3, 5)},
    "metric_overbought": {0.98: ({}, {}), "off": ({"metric_overbought_threshold": OFF}, {})},
}


#: Round 2 (2026-10-13): the same screen on top of the adopted 24 h grace. Controls = arm D run
#: from a fresh 24 h snapshot and the other models' adopted 24 h arms; their per-fold .json pins
#: 24 h. Base values first, as in round 1.
MODELS_R2 = {
    "ARO": ("wf_aro_g24", "NNNC_A_Raw_Open", "NNNC_ARO"),
    "GBR": ("fs_gbr_gr", "NNNC_GbbRaw_Open", "NNNC_FS_GBR_GR"),
    "GFM": ("fs_gfm_gr", "NNNC_GbbFwdMae_Open", "NNNC_FS_GFM_GR"),
    "MLX": ("fs_mlx_gr", "NNNC_MLX", "NNNC_FS_MLX_GR"),
}
LEVERS_R2 = {
    "take_profit": {v: ({"take_profit": v}, {}) for v in (0.022, 0.015, 0.03, 0.04, OFF)},
    "tp_atr_k": {"base": ({}, {})} | {k: ({}, {"take_profit_atr_mult": k})
                                      for k in (2.5, 2.897, 3.5, 4.5, 6.0)},
    "roi": {"base": ({}, {}), 0.03: ({"roi": 0.03}, {}), 0.08: ({"roi": 0.08}, {}),
            "off": ({"roi": None}, {})},
    "grace_hours": {24.0: ({}, {})} | {h: ({}, {"grace_hours": h}) for h in (18.0, 30.0, 36.0, 48.0)},
    "grace_level": {-0.27: ({}, {})} | {lv: ({}, {"grace_level": lv})
                                        for lv in (-0.20, -0.35, -0.45)},
    "stop_scale": {1.0: ({}, {})} | {s: ({"_scale": s}, {}) for s in (0.75, 1.25, 1.5, 2.0, 2.5)},
    "vol_confirm": {True: ({}, {}), False: ({"use_volume_confirmation_stoploss": False}, {})},
    "unclog_12h_hours": {"base": ({}, {})} | {h: ({"unclog_12h_hours": h}, {}) for h in (6.0, 18.0)},
    "unclog_1d_hours": {24.0: ({}, {})} | {h: ({"unclog_1d_hours": h}, {})
                                           for h in (12.0, 36.0, 48.0)},
    "max_hold_days": {10: ({}, {})} | {d: ({"max_hold_days": d}, {}) for d in (2, 3, 5)},
    "metric_overbought": {0.98: ({}, {}), "off": ({"metric_overbought_threshold": OFF}, {})},
}
#: NNNC_FS_ROUND selects the round ("1" default): inputs, grids and output file (`_r2` suffix).
ROUND = os.environ.get("NNNC_FS_ROUND", "1")
SUFFIX = "" if ROUND == "1" else f"_r{ROUND}"
MODELS = {"1": MODELS_R1, "2": MODELS_R2}[ROUND]
LEVERS = {"1": LEVERS_R1, "2": LEVERS_R2}[ROUND]


def lever_params(params, over: dict):
    over = dict(over)
    s = over.pop("_scale", None)
    over.pop("roi", None)
    if s is not None:
        over |= {"atr_stoploss_multiplier": params.atr_stoploss_multiplier * s,
                 "atr_stoploss_floor": params.atr_stoploss_floor * s,
                 "atr_stoploss_cap": params.atr_stoploss_cap * s}
    return dataclasses.replace(params, **over) if over else params


def anatomy(df: pd.DataFrame) -> dict:
    out = {}
    for half, d in df.groupby("half"):
        g = d.groupby("rec_reason")["rec_pp"].agg(["count", "sum", "mean"])
        out[half] = {r: {"n": int(x["count"]), "pp": round(float(x["sum"]), 3),
                         "pp_per_trade": round(float(x["mean"]), 4)} for r, x in g.iterrows()}
        out[half]["_total"] = {"n": len(d), "pp": round(float(d["rec_pp"].sum()), 3)}
    return out


def lever_deltas(df: pd.DataFrame, class_name: str) -> dict:
    cls, params, roi = exit_setup(class_name)
    params = F.family_params(cls, params, FAMILY)
    early = df[df["half"] == "early"]
    res = {}
    for lever, values in LEVERS.items():
        res[lever] = {}
        for v, (pover, rover) in values.items():
            p = lever_params(params, pover)
            r = pover.get("roi", roi) if "roi" in pover else roi
            rules = P.Rules(**rover)
            tot, by_fold = 0.0, {}
            for t in early.itertuples():
                b, _ = F.bars_for(t.pair)
                cf = P.replay(b, t.e, t.open_rate, p, roi=r, roi_suppressed=t.roi_suppressed,
                              end=t.end, rules=rules, tick=t.tick)
                d = (cf.profit - t.null_profit) * t.stake * t.scale
                tot += d
                by_fold[t.fold] = by_fold.get(t.fold, 0.0) + d
            pos = sum(1 for x in by_fold.values() if x > 1e-9)
            neg = sum(1 for x in by_fold.values() if x < -1e-9)
            res[lever][str(v)] = {"early_pp": round(tot, 3), "folds_up": pos, "folds_down": neg}
    return res


def main() -> int:
    path = HERE / f"diagnose{SUFFIX}.json"
    out = (json.loads(path.read_text()) if path.exists() else
           {"label": "PROVISIONAL; lever deltas are EARLY half (f00-f07) only, vs the null replay",
            "models": {}})
    for m, (tag, cls_name, prefix) in MODELS.items():
        if m in out["models"]:
            continue
        _JSON_FOR[cls_name] = STRAT / "NNNC" / f"{prefix}_f00.json"
        df, setup = F.trade_rows(tag, cls_name, FAMILY)
        fid = F.fidelity(df)
        entry = {"tag": tag, "class": cls_name, "setup": setup, "fidelity": fid,
                 "anatomy": anatomy(df),
                 "paths": {h: F.summarise(d) for h, d in df.groupby("half")},
                 "null_vs_recorded_pp": {h: round(float(d["null_profit"].mul(d["stake"] * d["scale"]).sum()
                                                         - d["rec_pp"].sum()), 3)
                                         for h, d in df.groupby("half")}}
        if (fid["null_replay_agreement"] or 0) >= F.FIDELITY_MIN:
            entry["levers_early"] = lever_deltas(df, cls_name)
        print(m, len(df), "trades; fidelity", fid["null_replay_agreement"], flush=True)
        out["models"][m] = entry
        path.write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-component attribution of a SimpleTrigger configuration (feature 023 US2, FR-010..FR-012).

With no model, a trigger's bar set is fixed, so a framework component's effect can be read off
directly: run the full configuration, then one arm per component with ONLY that component set to
its neutral value (`SimpleTrigger/components.py`), and compare each against the full arm.

STATUS per component
  LIVE             the neutralised arm's trades differ; `liveness_evidence` shows the recorded
                   metric that moved (entry-condition counts, exit-reason counts).
  INERT            byte-identical trades in every fold. A DEFECT to investigate (constitution
                   IV) -- the component does nothing here -- never "no effect".
  ALREADY_NEUTRAL  the preset already holds the neutral value (e.g. every guard under the
                   guards-open ARM_D_OPEN); no arm is run.
  ZERO_TRADES      the neutralised arm entered nothing: inconclusive, not a zero return.

Each LIVE row carries the paired verdict (`compare_pooled.compare_data`, neutralised minus full,
so the COMPONENT's contribution is the negative of `delta_pp`), the early/late halves, and its
power: the pooled bootstrap floor (`power.bootstrap_floor`) annualised. Every number is
PROVISIONAL under quality gate 2 (no distant era).

  python -m evaluation.attribution --full-tag st_gbb --base-class SimpleTrigger_Gbb \
      --components all [--generate] [--run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STRAT = ROOT / "user_data/strategies"
sys.path.insert(0, str(STRAT))

from evaluation import bootstrap  # noqa: E402


bootstrap.BT_RESULTS = ROOT / "user_data/backtest_results"

from evaluation import BASELINE_HALF_WIDTH_PP, BASELINE_N_TRADES  # noqa: E402
from evaluation.compare_pooled import compare_data  # noqa: E402
from evaluation.halves import PROVISIONAL, split  # noqa: E402
from evaluation.pool import pool  # noqa: E402
from evaluation.power import bootstrap_floor  # noqa: E402


ARMS = STRAT / "regime/arms"
FOLDS = STRAT / "evaluation/folds"
SRC_PLAN = FOLDS / "plan_NNNC_A_Raw_Open.json"

TEMPLATE = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""{cls} -- {base} with component `{comp}` NEUTRALISED (feature 023 attribution). GENERATED."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from {base} import {base}


class {cls}({base}):
{body}
'''


def arm_class(base: str, comp: str) -> str:
    return f"{base}__C_{comp}"


def arm_tag(full_tag: str, comp: str) -> str:
    return f"{full_tag}__{comp}"


def generate(base_cls, comp) -> Path:
    """Write the neutralised subclass for one component. Overrides MERGE with the base's."""
    base = base_cls.__name__
    ov = comp.overrides
    lines = []
    if "trigger" in ov:
        lines.append(f"    TRIGGER_NAME = {ov['trigger']!r}")
    for section, attr in (
        ("buy", "BUY_OVERRIDES"),
        ("sell", "SELL_OVERRIDES"),
        ("attributes", "ATTR_OVERRIDES"),
    ):
        if ov.get(section):
            merged = {**getattr(base_cls, attr), **ov[section]}
            lines.append(f"    {attr} = {merged!r}")
    body = "\n".join(lines) or "    pass"
    cls = arm_class(base, comp.name)
    path = STRAT / "SimpleTrigger" / f"{cls}.py"
    path.write_text(TEMPLATE.format(cls=cls, base=base, comp=comp.name, body=body))
    return path


def _records(plan: dict) -> list[dict]:
    out = []
    for f in plan["folds"]:
        p = ARMS / f"{f['arm_tag']}_{f['fold_id'].upper()}.json"
        out.append(json.loads(p.read_text()) if p.is_file() else {})
    return out


def _sum_counts(records: list[dict], key: str) -> dict:
    total: dict = {}
    for r in records:
        for k, v in (r.get(key) or {}).items():
            if isinstance(v, dict):
                t = total.setdefault(k, {"pass": 0, "total": 0})
                t["pass"] += v.get("pass", 0)
                t["total"] += v.get("total", 0)
            else:
                total[k] = total.get(k, 0) + v
    return total


def _power(pooled: dict) -> dict:
    trades = pooled["trades"]
    if not trades:
        return {"n_trades": 0}
    years = sum(f["test_days"] for f in pooled["per_fold"]) / 365.0
    bs = bootstrap_floor(
        np.array([t["profit_abs"] for t in trades], dtype=float), trades[0]["scale_pp_per_abs"]
    )
    n = len(trades)
    return {
        "n_trades": n,
        "half_width_pp_yr": bs["half_width_pp"] / years,
        "sqrt_n_predicted_pp_yr": BASELINE_HALF_WIDTH_PP * (BASELINE_N_TRADES / n) ** 0.5,
    }


def _liveness(comp, full_recs, arm_recs) -> dict:
    metric = comp.liveness_metric
    if metric.startswith("entry_condition_counts."):
        k = metric.split(".", 1)[1]
        before = _sum_counts(full_recs, "entry_condition_counts").get(k)
        after = _sum_counts(arm_recs, "entry_condition_counts").get(k)
    elif metric.startswith("exit_reason_counts."):
        k = metric.split(".", 1)[1]
        before = _sum_counts(full_recs, "exit_reason_counts").get(k, 0)
        after = _sum_counts(arm_recs, "exit_reason_counts").get(k, 0)
    else:
        before = [r.get("trades_hash") for r in full_recs]
        after = [r.get("trades_hash") for r in arm_recs]
    return {"metric": metric, "before": before, "after": after, "moved": before != after}


def build_report(full_plan: dict, comp_plans: dict, components, preset: str) -> dict:
    full_pooled = pool(full_plan)
    full_recs = _records(full_plan)
    rows = []
    for comp in components:
        row = {"component": comp.name, "overrides": comp.overrides}
        if preset in comp.neutral_is_identity_under:
            rows.append({**row, "status": "ALREADY_NEUTRAL"})
            continue
        plan = comp_plans.get(comp.name)
        if plan is None:
            rows.append({**row, "status": "NOT_RUN"})
            continue
        recs = _records(plan)
        arm = pool(plan)
        row["arm_tag"] = plan["folds"][0]["arm_tag"].rsplit("_", 1)[0]
        row["fold_detail"] = [
            {"fold": p["fold_id"], "n_trades": p["n_trades"], "return_pct": p["return_pct"]}
            for p in arm["per_fold"]
        ]
        row["liveness_evidence"] = _liveness(comp, full_recs, recs)
        if arm["n_trades_pooled"] == 0:
            rows.append({**row, "status": "ZERO_TRADES"})
            continue
        if all(
            a.get("trades_hash") and a.get("trades_hash") == b.get("trades_hash")
            for a, b in zip(recs, full_recs, strict=True)
        ):
            rows.append({**row, "status": "INERT"})
            continue
        d = compare_data(arm, full_pooled)
        rows.append(
            {
                **row,
                "status": "LIVE",
                "verdict": d["verdict"],
                "delta_pp": d.get("delta_pp"),
                "ci_pp_yr": d.get("ci_pp_yr"),
                "p": (d.get("bootstrap") or {}).get("p"),
                "component_contribution_pp": (
                    -d["delta_pp"] if d.get("delta_pp") is not None else None
                ),
                "halves": split(arm, full_pooled),
                "power": _power(arm),
            }
        )
    return {
        "label": PROVISIONAL,
        "preset": preset,
        "full": {
            "trades": full_pooled["n_trades_pooled"],
            "sum_pp": sum(f["return_pct"] or 0 for f in full_pooled["per_fold"]),
            "power": _power(full_pooled),
        },
        "rows": rows,
    }


def _md(rep: dict, full_tag: str) -> str:
    lines = [
        f"# Component attribution: `{full_tag}` (preset {rep['preset']})",
        "",
        f"**{rep['label']}**",
        "",
        (
            f"Full configuration: {rep['full']['trades']} trades, "
            f"{rep['full']['sum_pp']:+.2f}pp per-fold sum."
        ),
        "",
        "| component | status | contribution pp | paired verdict | early / late | liveness |",
        "|---|---|---|---|---|---|",
    ]
    for r in rep["rows"]:
        live = r.get("liveness_evidence", {})
        h = r.get("halves")
        hl = f"{h['early']['verdict']} / {h['late']['verdict']}" if h else ""
        c = r.get("component_contribution_pp")
        lines.append(
            f"| {r['component']} | {r['status']} | {'' if c is None else f'{c:+.2f}'} | "
            f"{r.get('verdict', '')} | {hl} | "
            f"{live.get('metric', '')}{' moved' if live.get('moved') else ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full-tag", required=True)
    ap.add_argument("--base-class", required=True)
    ap.add_argument("--components", default="all")
    ap.add_argument("--generate", action="store_true", help="write classes + plans")
    ap.add_argument("--run", action="store_true", help="run each component plan (sequential)")
    a = ap.parse_args(argv)

    from evaluation.fingerprint import load_strategy_class
    from evaluation.foldplan import derive
    from SimpleTrigger.components import COMPONENTS

    base_cls = load_strategy_class(a.base_class)
    preset = base_cls.PRESET
    comps = [c for c in COMPONENTS if a.components == "all" or c.name in a.components.split(",")]
    src = json.loads(SRC_PLAN.read_text())
    full_plan = json.loads((FOLDS / f"plan_{a.full_tag}.json").read_text())
    comp_plans = {}
    for c in comps:
        if preset in c.neutral_is_identity_under:
            continue
        pp = FOLDS / f"plan_{arm_tag(a.full_tag, c.name)}.json"
        if a.generate:
            generate(base_cls, c)
            plan = derive(
                src, "SimpleTrigger", arm_class(a.base_class, c.name), arm_tag(a.full_tag, c.name)
            )
            pp.write_text(json.dumps(plan, indent=2) + "\n")
        if a.run and pp.is_file():
            print(f"=== running {c.name}", flush=True)
            rc = subprocess.run(
                [sys.executable, str(STRAT / "evaluation/run_folds.py"), "--plan", str(pp)],
                cwd=ROOT,
                check=False,
            ).returncode
            if rc:
                print(f"*** {c.name}: run_folds exited {rc}")
        if pp.is_file():
            comp_plans[c.name] = json.loads(pp.read_text())
    rep = build_report(full_plan, comp_plans, comps, preset)
    out = STRAT / f"evaluation/attribution_{a.full_tag}"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    out.with_suffix(".md").write_text(_md(rep, a.full_tag))
    print(_md(rep, a.full_tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())

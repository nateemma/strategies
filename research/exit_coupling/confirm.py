"""Confirmatory test of 027's reversion exit couplings on a fresh window (feature 028, contract I1).

Everything this module decides was fixed in a committed pre-registration
(`specs/028-recovery-aware-exits/preregistration.json`) before any judging-window backtest existed.
It refuses (exit 2) if that file has uncommitted changes, if a candidate class's constant differs
from its frozen value, if a judging-window record is older than the pre-registration commit, or
if two arms share a class (the fingerprint cannot see method overrides, so the class name is the
arm's identity).

    feasibility  US1, no new backtests beyond the control: the 027 replay on the control's
                 window trades -- fidelity gate, recovery share of stop exits, forgone P&L, and
                 each candidate's replay gain -> PASS/FAIL.          exit 0 / 3 none / 2
    judge        US2: each passing candidate's arm vs the control on the window folds, under the
                 pre-registered rule (a)-(d) and verdict mapping (research R4). exit 0 / 1 / 2
    judge --rerun-check   finalises the gate-5 re-run for ADOPTED gains under 1 pp.

"Gain" everywhere is the summed per-fold `return_pct` (arm minus control), the number 027 and
rule (d) use. `compare_data`'s paired `delta_pp` is reported, never decides.
PROVISIONAL under quality gate 2 (2022 adjoins f00).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

LABEL = ("CONFIRMATORY (pre-registered) -- PROVISIONAL under quality gate 2 "
         "(2022 adjoins f00; not a distant era)")
FAMILY = "reversion"
RERUN_BELOW_PP = 1.0

_PAIRED = {
    "IMPROVED": "pass", "UNCHANGED": "pass", "WORSENED": "fail",
    "INCONCLUSIVE_UNDERPOWERED": "pass", "INCONCLUSIVE_ARM_LOSES": "pass",
    "INCONCLUSIVE_ABSORBED": "inconclusive", "UNRESOLVABLE": "inconclusive",
}
_PAIRED_FLAGGED = {"INCONCLUSIVE_UNDERPOWERED", "INCONCLUSIVE_ARM_LOSES"}
_RISK = {"IMPROVED": "pass", "UNCHANGED": "pass", "WORSENED": "fail",
         "UNRESOLVABLE": "inconclusive"}


# ----------------------------------------------------------------------------- pure functions


def recovery_share(df: pd.DataFrame) -> tuple[float | None, int, int]:
    """Clean (non-ambiguous) stop exits whose price touched the target after the stop, within
    the max hold -- 027's `a_touch_after_stop_maxhold`."""
    stops = df[(df["rec_reason"] == "stop") & ~df["ambiguous"].astype(bool)]
    n_stop = len(stops)
    n_rec = int(stops["touch_m_after"].astype(bool).sum())
    return (None if not n_stop else n_rec / n_stop), n_stop, n_rec


def candidate_passes(gain_pp: float, share: float | None, floor: float,
                     applies_floor: bool) -> bool:
    if not gain_pp > 0:
        return False
    if not applies_floor:
        return True
    return share is not None and share >= floor


def map_paired(verdict: str) -> str:
    if verdict not in _PAIRED:
        raise ValueError(f"unmapped paired verdict {verdict!r}")
    return _PAIRED[verdict]


def map_risk(verdict: str) -> str:
    if verdict not in _RISK:
        raise ValueError(f"unmapped risk verdict {verdict!r}")
    return _RISK[verdict]


def decide(gain_pp: float, paired_verdict: str, risk: dict, early_gain: float,
           late_gain: float, ctl_trades: int, folds_changed: int,
           prereg: dict) -> tuple[str, list[str], list[str]]:
    """(verdict, failed rule letters, flags) under research R4."""
    p = map_paired(paired_verdict)
    r = [map_risk(risk["verdict_sharpe"]), map_risk(risk["verdict_maxdd"])]
    flags = [paired_verdict] if paired_verdict in _PAIRED_FLAGGED else []
    if (ctl_trades < prereg["min_control_trades"] or folds_changed == 0
            or p == "inconclusive" or "inconclusive" in r):
        return "INCONCLUSIVE", [], flags
    failed = []
    if not gain_pp > 0:
        failed.append("a")
    if p == "fail":
        failed.append("b")
    if "fail" in r:
        failed.append("c")
    if not (early_gain > 0 and late_gain > 0):
        failed.append("d")
    return ("NOT ADOPTED" if failed else "ADOPT"), failed, flags


def ci_half_width_pp(ci_pp_yr, years: float) -> float | None:
    if not ci_pp_yr:
        return None
    lo, hi = ci_pp_yr
    return (hi - lo) / 2.0 * years


def needs_rerun(verdict: str, gain_pp: float) -> bool:
    return verdict == "ADOPT" and gain_pp < RERUN_BELOW_PP


def after_rerun(verdict: str, hash_match: bool) -> str:
    return verdict if hash_match else "INCONCLUSIVE"


def constants_mismatch(class_values: dict, frozen: dict) -> list[str]:
    return [f"{k}: class {class_values.get(k)} != frozen {v}"
            for k, v in frozen.items() if class_values.get(k) != v]


def stale_records(mtimes: dict, commit_ts: float) -> list[str]:
    return sorted(p for p, m in mtimes.items() if m <= commit_ts)


def shared_classes(classes: dict) -> list[str]:
    by: dict = {}
    for arm, cls in classes.items():
        by.setdefault(cls, []).append(arm)
    return [f"{c}: {', '.join(sorted(a))}" for c, a in sorted(by.items()) if len(a) > 1]


# ------------------------------------------------------------------------------------ inputs


def _plan(tag: str) -> dict:
    return json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}.json").read_text())


def _record_path(fold: dict) -> Path:
    return STRAT / "regime" / "arms" / f"{fold['arm_tag']}_{fold['fold_id'].upper()}.json"


def _class_value(cls, key: str):
    """`ATTR` or `ATTR.key` (a dict attribute such as SELL_OVERRIDES)."""
    head, _, tail = key.partition(".")
    value = getattr(cls, head, None)
    return value.get(tail) if tail and isinstance(value, dict) else value


def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(STRAT), *args], text=True).strip()


def refusals(prereg_path: Path, prereg: dict, tags: list[str]) -> list[str]:
    """R6. `tags` are the plans whose judging-window records must post-date the commit."""
    from evaluation.fingerprint import load_strategy_class

    out = []
    rel = prereg_path.resolve().relative_to(STRAT)
    if _git("status", "--porcelain", "--", str(rel)):
        out.append(f"{rel} has uncommitted changes")
    ts = _git("log", "-1", "--format=%ct", "--", str(rel))
    if not ts:
        out.append(f"{rel} is not committed")
    for c in prereg["candidates"]:
        cls = load_strategy_class(c["class"])
        frozen = c["values"]
        out += [f"{c['id']} {m}" for m in
                constants_mismatch({k: _class_value(cls, k) for k in frozen}, frozen)]
    classes = {"control": prereg["control"]["class"]} | {c["id"]: c["class"]
                                                          for c in prereg["candidates"]}
    out += [f"shared class {s}" for s in shared_classes(classes)]
    if ts:
        mt = {}
        for tag in tags:
            for f in _plan(tag)["folds"]:
                p = _record_path(f)
                if p.is_file():
                    mt[str(p.relative_to(STRAT))] = p.stat().st_mtime
        out += [f"record older than the pre-registration commit: {p}"
                for p in stale_records(mt, float(ts))]
    return out


# ------------------------------------------------------------------------------- feasibility


def _replay_gain(df: pd.DataFrame, class_name: str, rules_kw: dict) -> float:
    from research.exit_coupling import feasibility as F
    from research.exit_coupling import paths as P

    cls, params, roi = F.exit_setup(class_name)
    params = F.family_params(cls, params, FAMILY)
    rule = P.Rules(**rules_kw)
    total = 0.0
    for r in df.itertuples():
        b, _ = F.bars_for(r.pair)
        cf = P.replay(b, r.e, r.open_rate, params, roi=roi, roi_suppressed=r.roi_suppressed,
                      end=r.end, rules=rule, tick=r.tick)
        total += (cf.profit - r.null_profit) * r.stake * r.scale
    return float(total)


def run_feasibility(prereg_path: Path, out: Path) -> int:
    from research.exit_coupling import feasibility as F

    prereg = json.loads(prereg_path.read_text())
    ctl = prereg["control"]
    bad = refusals(prereg_path, prereg, [ctl["plan_tag"]])
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    try:
        df, setup = F.trade_rows(ctl["plan_tag"], ctl["class"], FAMILY)
    except FileNotFoundError as exc:
        print(f"*** {exc}")
        return 2
    fid = F.fidelity(df)
    fid_ok = (fid["null_replay_agreement"] or 0) >= prereg["fidelity_min"] and (
        fid["stop_model_agreement"] is None
        or fid["stop_model_agreement"] >= prereg["fidelity_min"])
    share, n_stop, n_rec = recovery_share(df)
    stops = df[(df["rec_reason"] == "stop") & ~df["ambiguous"].astype(bool)
               & df["touch_m_after"].astype(bool)]
    forgone = float(((stops["target_net"] - stops["rec_profit"]) * stops["stake"]
                     * stops["scale"]).sum())
    res = {"label": LABEL, "control": ctl, "trades": len(df),
           "ctl_pp": float(df["rec_pp"].sum()), "fidelity": fid, "fidelity_ok": fid_ok,
           "stop_exits": n_stop, "recovered_maxhold": n_rec, "recovery_share": share,
           "recovery_floor": prereg["recovery_floor"], "forgone_pp": forgone,
           "candidates": {}}
    if fid_ok:
        for c in prereg["candidates"]:
            g = _replay_gain(df, ctl["class"], c["replay_rules"])
            res["candidates"][c["id"]] = {
                "replay_gain_pp": g, "expected_arm_gain_pp": 0.5 * g,
                "floor_applies": c["applies_recovery_floor"],
                "pass": candidate_passes(g, share, prereg["recovery_floor"],
                                         c["applies_recovery_floor"])}
    code = 2 if not fid_ok else (0 if any(v["pass"] for v in res["candidates"].values()) else 3)
    res["exit"] = code
    out.write_text(_render_feasibility(res))
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    print(out.read_text())
    return code


def _render_feasibility(r: dict) -> str:
    f = r["fidelity"]
    s = r["recovery_share"]
    L = ["# 028 US1 — mechanism check on the judging window", "", f"*{r['label']}*", "",
         f"Control `{r['control']['plan_tag']}` ({r['control']['class']}): {r['trades']} trades, "
         f"{r['ctl_pp']:+.2f} pp.", "",
         f"- fidelity: null replay reproduces {f['null_replay_agreement']:.3f} of recorded exits "
         f"(reason AND bar); stop model {f['stop_model_agreement']}; "
         f"{'PASS' if r['fidelity_ok'] else '**FAIL**'} (min 0.90)",
         f"- by recorded reason: `{json.dumps(f['by_recorded_reason'])}`",
         f"- stop exits {r['stop_exits']}, recovered to +2% within the max hold "
         f"{r['recovered_maxhold']} -> recovery share **{'n/a' if s is None else f'{s:.3f}'}** "
         f"(floor {r['recovery_floor']}; 2023–26: 0.634 / 0.575)",
         f"- forgone P&L of the recovered stop exits (reported only): {r['forgone_pp']:+.2f} pp",
         "", "| candidate | replay gain pp | expected arm gain (0.5×) | floor applies | result |",
         "|---|---|---|---|---|"]
    for k, v in r["candidates"].items():
        L.append(f"| {k} | {v['replay_gain_pp']:+.2f} | {v['expected_arm_gain_pp']:+.2f} | "
                 f"{'yes' if v['floor_applies'] else 'no'} | "
                 f"{'**PASS**' if v['pass'] else 'FAIL'} |")
    L += ["", f"exit {r['exit']} — "
          + {0: "at least one PASS", 3: "STOP (none pass)", 2: "NO VERDICT (fidelity)"}[r["exit"]]]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------- judge


def _fold_hashes(tag: str) -> dict:
    out = {}
    for f in _plan(tag)["folds"]:
        p = _record_path(f)
        if p.is_file():
            out[f["fold_id"]] = json.loads(p.read_text())
    return out


def _sum_pp(pooled: dict) -> float:
    return float(sum(f["return_pct"] or 0.0 for f in pooled["per_fold"]))


def run_judge(prereg_path: Path, feas_path: Path, out: Path, rerun_check: bool) -> int:
    from evaluation import risk
    from evaluation.compare_pooled import compare_data
    from evaluation.halves import subset
    from evaluation.pool import pool

    prereg = json.loads(prereg_path.read_text())
    feas = json.loads(feas_path.read_text())
    ctl = prereg["control"]
    passing = [c for c in prereg["candidates"]
               if (feas["candidates"].get(c["id"]) or {}).get("pass")]
    tags = [ctl["plan_tag"]] + [c["plan_tag"] for c in passing]
    if rerun_check:
        tags += [f"{c['plan_tag']}_rr" for c in passing
                 if (STRAT / "evaluation" / "folds" / f"plan_{c['plan_tag']}_rr.json").is_file()]
    bad = refusals(prereg_path, prereg, tags)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    E = {f["fold_id"] for f in prereg["folds"]}
    cp = pool(_plan(ctl["plan_tag"]))
    if cp["folds_missing"]:
        print(f"*** control missing folds {cp['folds_missing']}")
        return 2
    c_sub = subset(cp, E)
    years = sum(f["test_days"] for f in c_sub["per_fold"]) / 365.0
    ctl_recs = _fold_hashes(ctl["plan_tag"])
    rows = []
    for c in prereg["candidates"]:
        row = {"id": c["id"], "class": c["class"], "plan_tag": c["plan_tag"],
               "early_gain_pp": c["early_gain_pp"], "late_gain_pp": c["late_gain_pp"],
               "replay_gain_pp": (feas["candidates"].get(c["id"]) or {}).get("replay_gain_pp")}
        if c not in passing:
            rows.append(row | {"verdict": "STOPPED_AT_FEASIBILITY", "failed": [], "flags": []})
            continue
        ap = pool(_plan(c["plan_tag"]))
        if ap["folds_missing"]:
            print(f"*** {c['id']} missing folds {ap['folds_missing']}")
            return 2
        a_sub = subset(ap, E)
        cmp = compare_data(a_sub, c_sub)
        rk = risk.compare(a_sub["trades"], c_sub["trades"])
        recs = _fold_hashes(c["plan_tag"])
        changed = sum(recs[f]["trades_hash"] != ctl_recs[f]["trades_hash"] for f in sorted(E))
        gain = _sum_pp(a_sub) - _sum_pp(c_sub)
        verdict, failed, flags = decide(gain, cmp["verdict"], rk, c["early_gain_pp"],
                                        c["late_gain_pp"], c_sub["n_trades_pooled"], changed,
                                        prereg)
        ex_a, ex_c = Counter(), Counter()
        for f in sorted(E):
            ex_a.update(recs[f].get("exit_reason_counts") or {})
            ex_c.update(ctl_recs[f].get("exit_reason_counts") or {})
        per_fold = {pf["fold_id"]: pf["return_pct"] for pf in a_sub["per_fold"]}
        ctl_fold = {pf["fold_id"]: pf["return_pct"] for pf in c_sub["per_fold"]}
        row |= {
            "sum_pp_arm": _sum_pp(a_sub), "sum_pp_ctl": _sum_pp(c_sub), "gain_pp": gain,
            "per_fold_gain_pp": {f: (per_fold[f] or 0) - (ctl_fold[f] or 0) for f in sorted(E)},
            "paired": {"verdict": cmp["verdict"], "delta_pp": cmp.get("delta_pp"),
                       "ci_pp_yr": cmp.get("ci_pp_yr"),
                       "p": (cmp.get("bootstrap") or {}).get("p"),
                       "flag": flags[0] if flags else None},
            "risk": rk, "ctl_trades": c_sub["n_trades_pooled"],
            "arm_trades": a_sub["n_trades_pooled"], "folds_changed": changed,
            "exit_reasons_arm": dict(ex_a), "exit_reasons_ctl": dict(ex_c),
            "ci_half_width_pp": ci_half_width_pp(cmp.get("ci_pp_yr"), years),
            "verdict": verdict, "failed": failed, "flags": flags,
            "needs_rerun": needs_rerun(verdict, gain), "rerun_hash_match": None,
        }
        if row["needs_rerun"] and rerun_check:
            rr_tag = f"{c['plan_tag']}_rr"
            rr_plan = STRAT / "evaluation" / "folds" / f"plan_{rr_tag}.json"
            if not rr_plan.is_file():
                print(f"*** {c['id']}: gate-5 re-run plan {rr_plan.name} missing")
                return 2
            rr = _fold_hashes(rr_tag)
            match = bool(rr) and all(rr[f]["trades_hash"] == recs[f]["trades_hash"] for f in rr)
            row["rerun_hash_match"] = match
            row["verdict"] = after_rerun(verdict, match)
        rows.append(row)
    res = {"label": LABEL, "years": years, "folds": sorted(E), "control": ctl,
           "ctl_sum_pp": _sum_pp(c_sub), "ctl_trades": c_sub["n_trades_pooled"],
           "candidates": rows}
    out.write_text(_render_judge(res))
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    print(out.read_text())
    return 0 if any(r["verdict"] == "ADOPT" for r in rows) else 1


def risk_line(rk: dict) -> str:
    from evaluation.risk import line

    return line(rk) if "arm" in rk else f"risk {rk['verdict_sharpe']} / {rk['verdict_maxdd']}"


def _f(x, nd=2):
    return "—" if x is None else f"{x:+.{nd}f}"


def _render_judge(r: dict) -> str:
    L = ["# 028 US2 — verdicts on the judging window", "", f"*{r['label']}*", "",
         f"Folds {', '.join(r['folds'])} ({r['years']:.3f} years). Control "
         f"`{r['control']['plan_tag']}`: {r['ctl_sum_pp']:+.2f} pp, {r['ctl_trades']} trades.", "",
         "| candidate | 2022 gain pp (a) | paired (b) | risk Sharpe / maxDD (c) | 2023–26 early / late (d) "
         "| folds changed | CI half-width pp | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for c in r["candidates"]:
        if c["verdict"] == "STOPPED_AT_FEASIBILITY":
            L.append(f"| {c['id']} | — | — | — | {_f(c['early_gain_pp'])} / {_f(c['late_gain_pp'])} "
                     f"| — | — | STOPPED_AT_FEASIBILITY |")
            continue
        p = c["paired"]
        rk = c["risk"]
        v = c["verdict"] + (f" (failed {', '.join(c['failed'])})" if c["failed"] else "")
        L.append(
            f"| {c['id']} | {_f(c['gain_pp'])} | {p['verdict']} (p {p['p']}) | "
            f"{rk['verdict_sharpe']} / {rk['verdict_maxdd']} | "
            f"{_f(c['early_gain_pp'])} / {_f(c['late_gain_pp'])} | {c['folds_changed']}/3 | "
            f"{_f(c['ci_half_width_pp'])} | **{v}** |")
    L += [""]
    for c in r["candidates"]:
        if c["verdict"] == "STOPPED_AT_FEASIBILITY":
            continue
        L += [f"## {c['id']} (`{c['class']}`)", "",
              f"- arm {c['sum_pp_arm']:+.2f} pp vs control {c['sum_pp_ctl']:+.2f} pp "
              f"({c['arm_trades']} vs {c['ctl_trades']} trades); per fold "
              + ", ".join(f"{k} {v:+.2f}" for k, v in c["per_fold_gain_pp"].items()),
              f"- replay predicted {_f(c['replay_gain_pp'])} pp; arm realised {_f(c['gain_pp'])} pp",
              f"- paired: {c['paired']['verdict']}, delta_pp {_f(c['paired']['delta_pp'])} "
              f"(reported only), CI {c['paired']['ci_pp_yr']} pp/yr, p {c['paired']['p']}"
              + (f"; flagged {c['paired']['flag']}" if c["paired"]["flag"] else ""),
              f"- {risk_line(rk)}",
              f"- exit reasons arm {c['exit_reasons_arm']} vs control {c['exit_reasons_ctl']}",
              f"- gate-5 re-run: {'needed' if c['needs_rerun'] else 'not needed'}"
              + (f", hash match {c['rerun_hash_match']}" if c["rerun_hash_match"] is not None
                 else ""), ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feasibility")
    f.add_argument("--prereg", required=True)
    f.add_argument("--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--prereg", required=True)
    j.add_argument("--feasibility", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--rerun-check", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "feasibility":
        return run_feasibility(Path(a.prereg), Path(a.out))
    return run_judge(Path(a.prereg), Path(a.feasibility), Path(a.out), a.rerun_check)


if __name__ == "__main__":
    sys.exit(main())

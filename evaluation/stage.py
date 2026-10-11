"""Early-choose / late-judge stage runner (feature 025, contract C5).

One stage = one decision: which of several pre-declared candidate arms (already run) replaces the
previous arm. The CHOICE uses folds f00-f07 only; the JUDGEMENT is the late-half (f08-f14) paired
verdict of the chosen arm against the previous one, and the candidate is adopted ONLY
when that verdict is IMPROVED (constitution VIII; 024 FR-012). Every call appends a block
to the decisions file. PROVISIONAL under quality gate 2.

Selection (research R4/R5, analyze G4):
  - degenerate candidates are excluded: a family's combined-arm trades below 25% of its
    single-family arm's in either half (the rule reduced the arm to one family);
  - primary key: whole-arm f00-f07 per-fold wallet sum (higher wins);
  - tie-break: `coexistence` -> fewer early entries lost to the other family's pair;
               `tuneN`       -> higher reversion-side early contribution (pp).

    python -m evaluation.stage --stage coexistence --previous st_fam_combo \\
        --candidates st_coex_cap3,st_coex_cap5 --singles reversion=st_gbb,momentum=st_bgo_mx \\
        --out specs/025-family-coexistence/decisions.md

Exit: 0 adopted, 3 not adopted, 2 missing input. Re-executes itself with PYTHONHASHSEED=0 so the
paired bootstrap (which iterates sets) is reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRAT))

ARMS = STRAT / "regime" / "arms"
FOLDS = STRAT / "evaluation" / "folds"
EARLY = frozenset(f"f{i:02d}" for i in range(8))
DEGENERATE_SHARE = 0.25
LABEL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"


# ----------------------------------------------------------------------------------- selection


def select(rows: list[dict], stage: str) -> dict | None:
    """The early-fold winner, or None. Reads ONLY early-fold fields (test-enforced)."""
    eligible = [r for r in rows if not r["degenerate"]]
    if not eligible:
        return None
    if stage == "coexistence":
        key = lambda r: (r["early_sum"], -r["lost_other_early"])  # noqa: E731
    else:
        key = lambda r: (r["early_sum"], r["reversion_early_pp"])  # noqa: E731
    return max(eligible, key=key)


def decide(chosen: dict | None, late: dict | None) -> dict:
    if chosen is None or late is None:
        return {"chosen": None, "late": None, "adopted": False}
    return {"chosen": chosen["tag"], "late": late, "adopted": late.get("verdict") == "IMPROVED"}


def exit_code(decision: dict) -> int:
    return 0 if decision["adopted"] else 3


# ------------------------------------------------------------------------------------- loading


def _records(tag: str) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(ARMS.glob(f"{tag}_f[0-9][0-9]_F*.json"))]
    if len(recs) != 15 or any(r.get("returncode") != 0 for r in recs):
        raise FileNotFoundError(f"{tag}: need 15 successful fold records, have {len(recs)}")
    return recs


def _pooled(tag: str) -> dict:
    from evaluation.pool import pool

    plan = FOLDS / f"plan_{tag}.json"
    if not plan.is_file():
        raise FileNotFoundError(f"missing plan {plan}")
    return pool(json.loads(plan.read_text()))


def _fold(rec: dict) -> str:
    return rec["tag"].rsplit("_", 1)[-1]


def _wallet(recs, early: bool) -> float:
    return sum(r.get("wallet_profit_pct", 0.0) for r in recs if (_fold(r) in EARLY) == early)


def _degenerate(con: dict, singles, stage: str) -> bool:
    """025 rule: any family below 25% of its single-family trades in either half. 026 R7/H2: for
    `regime`, reversion's LATE count is unconstrained (cutting it is the point of the gate)."""
    for f in singles:
        for h in ("early", "late"):
            if stage == "regime" and f == "reversion" and h == "late":
                continue
            if con[f][h]["combined_trades"] < DEGENERATE_SHARE * con[f][h]["single_trades"]:
                return True
    return False


def row(tag: str, pooled: dict, singles: dict, max_open: int, stage: str = "") -> dict:
    from evaluation.contention import contention
    from utils.FamilyExits import family_of

    recs = _records(tag)
    con = contention(pooled, singles, max_open)
    rev_early = sum(
        (t["profit_abs"] or 0.0) * (t.get("scale_pp_per_abs") or 0.0)
        for t in pooled["trades"]
        if t["fold_id"] in EARLY and family_of(t.get("enter_tag")) == "reversion"
    )
    degenerate = _degenerate(con, singles, stage)
    return {
        "tag": tag,
        "early_sum": round(_wallet(recs, True), 4),
        "late_sum": round(_wallet(recs, False), 4),
        "lost_other_early": sum(con[f]["early"]["lost_by_cause"]["pair_held_by_other_family"]
                                for f in singles),
        "reversion_early_pp": round(rev_early, 4),
        "degenerate": degenerate,
    }


def liveness(tag: str, previous: str, stage: str) -> str:
    recs, prev = _records(tag), _records(previous)
    if stage == "coexistence":
        cap = sum(sum((r.get("coexistence_refusals") or {}).get("pair_cap", {}).values())
                  for r in recs)
        win = sum(((r.get("entry_condition_counts") or {}).get("coexist_recency_refused") or {})
                  .get("pass", 0) for r in recs)
        live = cap + win > 0
        state = "LIVE" if live else "INERT"
        return f"pair_cap refusals {cap}; recency-dropped rows {win} -> {state}"

    def rev(rs):
        out: dict = {}
        for r in rs:
            for k, v in ((r.get("family_exit_reason_counts") or {}).get("reversion") or {}).items():
                out[k] = out.get(k, 0) + v
        return out

    a, b = rev(recs), rev(prev)
    moved = {k: (b.get(k, 0), a.get(k, 0)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    hashes = sum(x["trades_hash"] != y["trades_hash"] for x, y in zip(recs, prev, strict=True))
    live = bool(moved) or hashes > 0
    return (f"reversion exit counts moved {moved or 'none'}; folds with different trades "
            f"{hashes}/15 -> {'LIVE' if live else 'INERT'}")


# -------------------------------------------------------------------------------------- output


def write_block(out: Path, stage: str, previous: str, rows: list[dict], decision: dict,
                live: dict) -> None:
    header = (
        "| candidate | early sum (f00-f07) | lost to other family (early) "
        "| reversion early pp | degenerate | liveness |"
    )
    lines = [f"## Stage: {stage}", "", f"*{LABEL}*", "", f"previous arm: `{previous}`", "",
             header, "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['early_sum']:+.2f} | {r['lost_other_early']} | "
            f"{r['reversion_early_pp']:+.2f} | {r['degenerate']} | {live.get(r['tag'], '')} |"
        )
    lines.append("")
    if decision["chosen"] is None:
        lines.append("chosen: none (no eligible candidate)")
    else:
        late = decision["late"]
        verdict = (
            f"late (f08-f14) vs `{previous}`: **{late['verdict']}**, delta "
            f"{late.get('delta_pp')} pp, CI {late.get('ci_pp_yr')} pp/yr, p {late.get('p')}"
        )
        lines += [f"chosen: `{decision['chosen']}` (best early sum)", "", verdict]
    if decision.get("late_vs_previous"):
        lp = decision["late_vs_previous"]
        prev_line = (
            f"(reported only) late vs the previous arm: **{lp['verdict']}**, delta "
            f"{lp.get('delta_pp')} pp, CI {lp.get('ci_pp_yr')} pp/yr, p {lp.get('p')}"
        )
        lines += ["", prev_line]
    lines += ["", f"adopted: **{'yes' if decision['adopted'] else 'no'}**", "", ""]
    with Path(out).open("a") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--previous", required=True)
    ap.add_argument("--bar", help="arm the late judgement must beat (default: --previous)")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--singles", required=True, help="family=tag,family=tag")
    ap.add_argument("--config", default=str(STRAT / "config" / "config_wf.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cands = [c for c in a.candidates.split(",") if c]
    singles_tags = dict(x.split("=", 1) for x in a.singles.split(","))
    try:
        bar = a.bar or a.previous
        for t in [a.previous, bar, *cands, *singles_tags.values()]:
            _records(t)
            if not (FOLDS / f"plan_{t}.json").is_file():
                raise FileNotFoundError(f"missing plan for {t}")
    except FileNotFoundError as exc:
        print(f"*** {exc}")
        return 2
    if os.environ.get("PYTHONHASHSEED") != "0":  # reproducible paired bootstrap
        os.environ["PYTHONHASHSEED"] = "0"
        # deliberate: re-exec so the bootstrap's set iteration is reproducible
        argv_ = [sys.executable, "-m", "evaluation.stage", *(argv or sys.argv[1:])]
        os.execv(sys.executable, argv_)  # noqa: S606

    from evaluation.fingerprint import load_config
    from evaluation.halves import split

    max_open = int(load_config(a.config)["max_open_trades"])
    singles = {f: _pooled(t) for f, t in singles_tags.items()}
    pooled = {t: _pooled(t) for t in cands}
    rows = [row(t, pooled[t], singles, max_open, a.stage) for t in cands]
    chosen = select(rows, a.stage)
    late = split(pooled[chosen["tag"]], _pooled(bar))["late"] if chosen else None
    decision = decide(chosen, late)
    if chosen and bar != a.previous:  # reported, not used for adoption
        decision["late_vs_previous"] = split(pooled[chosen["tag"]], _pooled(a.previous))["late"]
        decision["bar"] = bar
    live = {t: liveness(t, a.previous, a.stage) for t in cands}
    write_block(Path(a.out), a.stage, bar, rows, decision, live)
    print(Path(a.out).read_text().split("## Stage: ")[-1])
    return exit_code(decision)


if __name__ == "__main__":
    sys.exit(main())

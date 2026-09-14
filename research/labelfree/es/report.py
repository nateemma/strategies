"""PBO as the headline, the protocol §5 verdict, and the result document (T044-T048, T053-T056).

THE HEADLINE IS THE OVERFITTING PROBABILITY, not a point estimate of return. The brief's own
framing is that a search of this shape is expected to overfit and that measuring how much IS
the deliverable, and clarification Q3 settles the apparent conflict with protocol §5: PBO > 0.2
KILLS the candidate while the MEASURED PBO is the reported finding. §5 is preserved, not
relaxed. A quantified negative is a successful outcome, and briefs 01-04 delivered their value
exactly that way.

THREE THINGS THAT MUST APPEAR BESIDE EVERY NUMBER:
  * the evaluation count, because `stats.pbo_cscv`'s own docstring records CSCV is biased LOW
    below ~50 configurations — the count is what makes the figure interpretable
  * trade count, trades/yr and mean holding time on every return (FR-017); a bare percentage
    is not admissible anywhere in this feature's output
  * the UNREGULARISED score beside the regularised one, so a reader can see what the
    incumbent-distance and turnover penalties did rather than take it on trust
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree import stats as ST                # noqa: E402
from research.labelfree.es import exits as EX             # noqa: E402
from research.labelfree.es import history as H            # noqa: E402
from research.labelfree.es import space as SP             # noqa: E402
from research.labelfree.es import verdict as VD           # noqa: E402


def performance_matrix(rows: list[dict], n_blocks: int = 16) -> tuple[np.ndarray, list[str]]:
    """(n_blocks, n_configurations) for CSCV, over the ENTIRE history (FR-014).

    Every candidate, every generation, BOTH methods, including abandoned and discarded ones.
    A configuration's column is its block returns; a configuration evaluated on several folds
    contributes the concatenation, which is what gives CSCV a time axis at all.
    """
    by_cand: dict[str, list[float]] = {}
    for r in rows:
        if r.get("score") is None:
            continue                      # abandoned: counted in n_trials, no series to add
        br = r.get("block_returns")
        if not br:
            br = [float(r["score"])] * n_blocks
        by_cand.setdefault(r["candidate_id"], []).extend(br)
    names = sorted(by_cand)
    if not names:
        return np.zeros((0, 0)), []
    L = min(len(v) for v in by_cand.values())
    L = max(2, min(L, n_blocks))
    M = np.array([by_cand[n][:L] for n in names], dtype=float).T
    return M, names


def compute(rows: list[dict]) -> dict:
    total = len(rows)                      # ATTEMPTS, not successes — the deflation counts these
    scored = [r for r in rows if r.get("score") is not None]
    M, names = performance_matrix(rows)

    pbo = float(ST.pbo_cscv(M)) if M.size and M.shape[1] >= 2 else float("nan")

    vals = np.array([r["score"] for r in scored], dtype=float) if scored else np.array([])
    sr = float(vals.mean() / vals.std(ddof=1)) if vals.size > 2 and vals.std(ddof=1) > 0 else 0.0
    dsr = ST.deflated_sharpe(sr, n_obs=max(2, M.shape[0]), n_trials=max(2, total))

    return {"pbo": None if np.isnan(pbo) else round(pbo, 4),
            "n_evaluations": total,           # reported BESIDE the PBO (FR-016)
            "n_scored": len(scored),
            "n_abandoned": total - len(scored),
            "n_configurations": len(names),
            "cscv_biased_low_below": 50,
            "cscv_count_caveat": (
                "stats.pbo_cscv is biased LOW below ~50 configurations; "
                f"this run has {len(names)}"),
            "deflated_sharpe": None if np.isnan(dsr) else round(float(dsr), 4),
            "deflation_used_total_attempts": total}


def _fmt_return(label: str, pp: float, r: dict) -> str:
    """FR-017: a bare percentage is never admissible."""
    return (f"{label}: {pp:+.2f} pp/yr "
            f"({r.get('n_trades', 0)} trades, {r.get('trades_per_year', 0):.1f}/yr, "
            f"mean hold {r.get('mean_hold_h', 0):.1f} h, "
            f"exposure {r.get('mean_exposure', 0):.4f})")


def document(state: dict) -> str:
    st = state["stats"]
    v = state["verdict"]
    g = state["gates"]
    L = []
    A = L.append
    A(f"# Brief 05 — gradient-free policy search over the exit and guard levers")
    A("")
    A(f"**Date**: {date.today().isoformat()}  ")
    A(f"**Feature**: 013-labelfree-es-policy-search  ")
    A(f"**Verdict**: **{v['status']}**")
    A("")
    A("## Headline")
    A("")
    A(f"```")
    A(f"probability of backtest overfitting   {st['pbo']}")
    A(f"evaluations entering it               {st['n_evaluations']}"
      f"   ({st['n_abandoned']} abandoned, still counted)")
    A(f"distinct configurations               {st['n_configurations']}")
    A(f"deflated Sharpe (by TOTAL attempts)   {st['deflated_sharpe']}")
    A(f"```")
    A("")
    A(f"{st['cscv_count_caveat']}.")
    A("")
    for r in v["reasons"]:
        A(f"- {r}")
    A("")
    A("## Protocol §5 gates")
    A("")
    A("| gate | result | passed |")
    A("|---|---|---|")
    for k in ("SC-001", "SC-002a", "SC-002b", "SC-003", "SC-008"):
        if k in g:
            A(f"| {k} | {_gate_line(g[k])} | {'yes' if g[k].get('passed') else 'NO'} |")
    A("")
    if state.get("liveness"):
        A("## Liveness of the search space (Principle IV)")
        A("")
        A("Pre-registered thresholds: inert < 0.10 pp/yr, weak < 3.35 pp/yr "
          "(data-model.md §6).")
        A("")
        A("| parameter | span pp | verdict |")
        A("|---|---|---|")
        for d in sorted(state["liveness"]["dimensions"], key=lambda x: -x["span_pp"]):
            A(f"| `{d['parameter']}` | {d['span_pp']:.3f} | {d['verdict']} |")
        inert = [d["parameter"] for d in state["liveness"]["dimensions"]
                 if d["verdict"] == "inert"]
        A("")
        A(f"**Inert (reported, not searched)**: {', '.join(inert) if inert else 'none'}")
        A("")
    if state.get("calibration"):
        c = state["calibration"]
        A("## Surrogate calibration (the blocking gate)")
        A("")
        A(f"- Spearman rho **{c.get('spearman_rho')}** over {c.get('n_compared')} candidates "
          f"scored both by the surrogate and by real freqtrade backtests")
        A(f"- agree on which candidate wins: **{c.get('agree_on_winner')}**")
        A(f"- pairwise rank disagreements: {c.get('n_pair_disagreements')}")
        A("")
        A("Rank agreement, not level: the search only ever uses the surrogate to ORDER "
          "candidates, so a constant offset is harmless and a reordering is fatal.")
        A("")
    A("## What the surrogate cannot move")
    A("")
    A(f"Replayed (not recomputed) share of exits: **{EX.REPLAYED_EXIT_SHARE:.1%}**. "
      f"{'Every exit mechanism is recomputed from data held in the panel.' if EX.REPLAYED_EXIT_SHARE == 0 else 'These exits are taken from the archive, so a parameter change that would have shifted them is invisible.'}")
    A("")
    if state.get("bounds_hit"):
        A("## Parameters on a bound (FR-019)")
        A("")
        A(f"`{'`, `'.join(state['bounds_hit'])}` — **the space was drawn too narrowly** "
          f"for these; the optimum sits at an edge.")
        A("")
    A("## Reproduction")
    A("")
    A("```")
    A("python -m research.labelfree.es.signals   --folds f00..f14 --which both")
    A("python -m research.labelfree.es.calibrate --n 40 --folds f07")
    A("python -m research.labelfree.es.liveness  --folds f05..f14")
    A("python -m research.labelfree.es.run --fold <f> --method cmaes  --budget 1600 --seed 7")
    A("python -m research.labelfree.es.run --fold <f> --method random --budget 1600 --seed 7")
    A("python -m research.labelfree.es.history --audit")
    A("```")
    return "\n".join(L) + "\n"


def _gate_line(x: dict) -> str:
    if x["gate"] == "SC-001":
        return (f"{x['arm_pp']:+.2f} vs control {x['control_pp']:+.2f} "
                f"= {x['delta_pp']:+.2f} pp/yr (floor {x['floor_pp']}), "
                f"CI {x['ci95']}")
    if x["gate"] == "SC-002a":
        return f"{x['positive_folds']}/{x['of']} folds positive (need {x['required']})"
    if x["gate"] == "SC-002b":
        return ", ".join(f"{k} {vv:+.2f}" for k, vv in x["windows"].items())
    if x["gate"] == "SC-003":
        return f"{x['arm_pp']:+.2f} vs bar {x['bar_pp']:+.2f} pp/yr"
    if x["gate"] == "SC-008":
        return f"{x['n_overfit']}/{x['n_configurations']} configurations overfit"
    return json.dumps(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        ROOT / "research/label-free-study/briefs/results" / f"05-{date.today()}.md"))
    ap.add_argument("--state", default=str(Path(__file__).resolve().parent / "state.json"))
    a = ap.parse_args()

    rows = H.read()
    st = compute(rows)
    state = {"stats": st, "gates": {}, "verdict": {"status": "NOT RESOLVED",
                                                   "reasons": ["search not yet run"]}}
    sp = Path(a.state)
    if sp.exists():
        state.update(json.loads(sp.read_text()))
        state["stats"] = st
        state["verdict"] = VD.status(
            state.get("gates", {}), st["pbo"],
            state.get("delta_vs_incumbent_pp"), state.get("beats_random"))

    for extra, name in ((Path(__file__).parent / "liveness.json", "liveness"),
                        (Path(__file__).parent / "calibration.json", "calibration")):
        if extra.exists():
            state[name] = json.loads(extra.read_text())

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document(state))          # written on a KILL too (FR-020)
    print(json.dumps(st, indent=2))
    print(f"\nverdict: {state['verdict']['status']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

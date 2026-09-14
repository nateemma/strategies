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


def incumbent_delta(best_params: dict | None, best_pp: float | None,
                    incumbent_pp: float | None,
                    all_selections: list[dict] | None = None) -> dict:
    """T049/T050 — FR-018 and FR-019.

    A difference inside ONE SWEEP-NOISE BAND is reported "not resolved", never an
    improvement: an entire rvol parameter sweep spans 4.93 pp, so a move smaller than that is
    indistinguishable from the sweep's own noise.
    """
    out = {"sweep_noise_pp": VD.SWEEP_NOISE_PP, "bounds_hit": []}
    if best_pp is not None and incumbent_pp is not None:
        d = best_pp - incumbent_pp
        out["delta_vs_incumbent_pp"] = round(d, 4)
        out["within_sweep_noise"] = bool(abs(d) < VD.SWEEP_NOISE_PP)
        out["wording"] = ("not resolved — inside the sweep-noise band"
                          if out["within_sweep_noise"] else "outside the sweep-noise band")
    # FR-019 asks about the OPTIMUM the search reached, and this design selects one candidate
    # PER FOLD — so checking only the best fold under-reports. The analyze pass caught f04
    # sitting on the cexit_max_days bound while `bounds_hit` read empty.
    #
    # `space.on_bound` already exempts hold_window_hours, whose incumbent IS its floor because
    # the mechanism can only widen the window — a false finding on every run.
    hits: dict[str, list[str]] = {}
    for f in (all_selections or []):
        for b in SP.on_bound(f.get("params", {})):
            hits.setdefault(b, []).append(f.get("fold_id", "?"))
    if best_params:
        for b in SP.on_bound(best_params):
            hits.setdefault(b, [])
    out["bounds_hit"] = sorted(hits)
    out["bounds_hit_by_fold"] = hits
    return out


def compute(rows: list[dict]) -> dict:
    total = len(rows)                      # ATTEMPTS, not successes — the deflation counts these
    scored = [r for r in rows if r.get("score") is not None]
    M, names = performance_matrix(rows)

    pbo = float(ST.pbo_cscv(M)) if M.size and M.shape[1] >= 2 else float("nan")

    # DSR deflates a Sharpe computed over TIME, so it must come from the selected
    # configuration's block series -- NOT from the cross-sectional spread of candidate
    # scores, which is a different quantity entirely and was what an earlier version used.
    best = max(scored, key=lambda r: r["score"], default=None)
    series = np.array((best or {}).get("block_returns") or [], dtype=float)
    if series.size > 2 and series.std(ddof=1) > 0:
        sr = float(series.mean() / series.std(ddof=1))
        dsr = ST.deflated_sharpe(sr, n_obs=series.size, n_trials=max(2, total))
    else:
        sr, dsr = 0.0, float("nan")

    return {"pbo": None if np.isnan(pbo) else round(pbo, 4),
            "n_evaluations": total,           # reported BESIDE the PBO (FR-016)
            "n_scored": len(scored),
            "n_abandoned": total - len(scored),
            "n_configurations": len(names),
            "cscv_biased_low_below": 50,
            "cscv_count_caveat": (
                "stats.pbo_cscv is biased LOW below ~50 configurations; "
                f"this run has {len(names)}"),
            "block_sharpe": round(sr, 4),
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
    A("# Brief 05 — gradient-free policy search over the exit and guard levers: "
      f"RESULT ({date.today().isoformat()})")
    A("")
    A("**Feature**: [`specs/013-labelfree-es-policy-search/`]"
      "(../../../../specs/013-labelfree-es-policy-search/)  ")
    A("**Analyze**: [pre-implementation]"
      "(../../../../specs/013-labelfree-es-policy-search/analyze.md)  ")
    A(f"**Verdict: {v['status']}.** " + (v["reasons"][0] if v["reasons"] else ""))
    A("")
    A("## Hypothesis")
    A("")
    A("The levers gradient descent cannot reach — the ATR stop multiplier, `stoploss_grace_*`,")
    A("the guard thresholds and minimum holding time — *jointly* matter, but have only ever")
    A("been swept one at a time on windows that cannot resolve them. Searching them jointly")
    A("against an honest objective may find an operating point one-at-a-time sweeps cannot.")
    A("")
    A("Prior evidence is against it: an entire rvol sweep spans **4.93 pp** inside a **7.633 pp**")
    A("noise floor, so the space's total range of effect is smaller than the uncertainty on")
    A("measuring it. A quantified negative was the expected deliverable.")
    A("")
    A("## Headline")
    A("")
    A("```")
    A(f"probability of backtest overfitting   {st['pbo']}")
    A(f"evaluations entering it               {st['n_evaluations']}"
      f"   ({st['n_abandoned']} abandoned, still counted)")
    A(f"distinct configurations               {st['n_configurations']}")
    A(f"deflated Sharpe (by TOTAL attempts)   {st['deflated_sharpe']}")
    A("```")
    A("")
    A(f"{st['cscv_count_caveat']}.")
    A("")
    for r in v["reasons"]:
        A(f"- {r}")
    A("")
    A("## The exit-reason census that shaped the search space")
    A("")
    A("Measured over the 378 archived walk-forward trades, before any code:")
    A("")
    A("| exit reason | trades | share | share of P&L | mechanism |")
    A("|---|---|---|---|---|")
    A("| `take_profit` | 195 | 51.6% | +179.7% | `cexit_take_profit` |")
    A("| `trailing_stop_loss` | 98 | 25.9% | −120.2% | **the ATR + grace stop stack** |")
    A("| `model_exit` | 45 | 11.9% | +21.9% | the model's own signal |")
    A("| `unclog_12h` | 27 | 7.1% | +7.6% | the 12h time rule |")
    A("| `roi` | 10 | 2.6% | +14.8% | `minimal_roi` |")
    A("| `stop_loss` | 1 | 0.3% | −4.7% | the static −0.05 initial stop |")
    A("")
    A("`trailing_stop_loss` is **not** freqtrade's native trailing stop, which is off")
    A("(`BaseStrategy.trailing_stop = False`). `interface.py:1654-1658` upgrades a stop exit to")
    A("`TRAILING_STOP_LOSS` whenever the stop that fired differs from the initial one, and")
    A("`BaseNNStrategy.py:145` sets `use_atr_adaptive_stoploss = True`. **Those 98 exits are the")
    A("brief's own ATR + grace stack.** Read naively, the `stop_loss` row alone would say the")
    A("headline lever governs 0.3% of exits; it governs 26.2% and the entire loss side.")
    A("")
    A("That is why `cexit_take_profit` was **added** to the search space — 51.6% of exits and")
    A("the mechanism that opposes the stop stack — giving ten dimensions, not the brief's eight.")
    A("")
    A("## Protocol §5 gates")
    A("")
    A("| gate | result | passed |")
    A("|---|---|---|")
    for k in ("SC-001", "SC-002a", "SC-002b", "SC-003", "SC-005", "SC-006", "SC-008"):
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


def write_shared_summary(run_id: str, extra: dict | None = None) -> dict:
    """T047/SC-018: ONE summary row per search run in the shared trials.jsonl.

    The row carries the sidecar's path, line count and sha256, so the shared log POINTS AT
    the full record instead of duplicating ~20 MB of evaluations into git. The audit then
    verifies both files — an audit that only saw the summary would certify a record it never
    examined.
    """
    row = {"run_id": run_id, "feature": "013-labelfree-es-policy-search",
           **H.summarise(), **(extra or {})}
    with H.SHARED.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


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
    if x["gate"] == "SC-005":
        return (f"delta {x['delta_at_1x_pp']:+.2f} pp/yr at 1x fees, "
                f"{x['delta_at_1_5x_pp']:+.2f} at 1.5x")
    if x["gate"] == "SC-006":
        return (f"recorded {x['recorded_pp']:.4f} pp/yr, reproduced "
                f"{x['reproduced_pp']:.4f} from the config file alone")
    return json.dumps({k: v for k, v in x.items() if k != "per_fold"})


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
        # SC-002b, computed here because it needs the fold calendar (Principle II).
        if state.get("per_fold"):
            from research.labelfree.folds import folds as _folds
            fd = {f.fold_id: (f.test_start, f.test_end) for f in _folds()}
            w = VD.windows_from_folds(state["per_fold"], fd)
            state["windows"] = w
            state.setdefault("gates", {})["SC-002b"] = {
                **VD.single_window(w["window_pp"]), "detail": w}
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

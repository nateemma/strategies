"""The protocol §6 result document for brief 06 (T035-T038).

Written WHATEVER THE OUTCOME, including a calibration kill (FR-021).

Leads with the generator census, because a reader who sees only the bootstrap will ask why
there is one generator when the brief planned four, and the answer is a measurement.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent


def _load(name: str):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def synthetic_training_check() -> dict:
    """T035a / FR-017 — assert nothing was trained on synthetic data.

    The brief's entire framing is that a generator cannot add information the real bars lack,
    so this is the claim most worth making CHECKABLE rather than implicit.

    PARSED, NOT GREPPED. A substring search produced false positives immediately: gencensus.py
    carries "GANInterface" and regex patterns as STRING LITERALS in its probe table, and a grep
    cannot tell a literal from a call. `ast` can, so the check walks the syntax tree and looks
    for two things that would actually constitute training:

      * an import of anything under `GANs/`, or
      * a call to a training entry point (`fit`, `train`, `partial_fit`, `backward`, `step`)

    Comments, docstrings and string literals are invisible to the parse, which is the point.
    """
    import ast

    TRAIN_CALLS = {"fit", "train", "partial_fit", "backward", "step", "train_on_batch"}
    hits = []
    for f in sorted(HERE.glob("*.py")):
        tree = ast.parse(f.read_text(), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""])
                for m in mods:
                    if m.split(".")[0] == "GANs":
                        hits.append(f"{f.name}:{node.lineno}: imports {m}")
            elif isinstance(node, ast.Call):
                fn = node.func
                name = (fn.attr if isinstance(fn, ast.Attribute)
                        else fn.id if isinstance(fn, ast.Name) else None)
                if name in TRAIN_CALLS:
                    hits.append(f"{f.name}:{node.lineno}: calls {name}()")
    return {"method": ("AST parse of this feature's modules for GAN imports and training "
                       "calls; string literals and comments are invisible to it"),
            "training_calls_searched": sorted(TRAIN_CALLS),
            "hits": hits,
            "passed": not hits,
            "claim": "no policy is trained on synthetic data by this feature (FR-017)"}


def document() -> str:
    census = _load("census.json")
    cal = _load("calibration.json")
    sweep = _load("calibration_sweep.json")
    stress = _load("stress_report.json")
    train = synthetic_training_check()

    L, A = [], None
    A = L.append
    A(f"# Brief 06 — a generative stress harness: RESULT ({date.today().isoformat()})")
    A("")
    A("**Feature**: [`specs/014-labelfree-generative-stress/`]"
      "(../../../../specs/014-labelfree-generative-stress/)  ")
    A("**Analyze**: [both passes]"
      "(../../../../specs/014-labelfree-generative-stress/analyze.md)  ")
    if cal is None:
        A("**Verdict: NOT RUN.** No calibration on record.")
    elif not cal.get("passed"):
        A("**Verdict: KILL.** The block bootstrap failed calibration, and clarification Q3 "
          "left no fallback generator. No harness is built on an uncalibrated generator.")
    else:
        v = (stress or {}).get("verdict", "—")
        A(f"**Verdict: harness ACCEPTED** (SC-001 passed). Stress verdict on the incumbent: "
          f"**{v}**.")
    A("")
    A("## Hypothesis")
    A("")
    A("A policy that clears the pooled-WF floor on real data will either keep its P&L and")
    A("drawdown shape on plausible unseen paths — the edge is structural — or collapse, in")
    A("which case it was fitted to path-specific features of the 15 folds. This is")
    A("**infrastructure, not an alpha source**: no policy is trained on synthetic data, because")
    A("a generator cannot add information the ~1.5M real bars lack.")
    A("")

    A("## Why there is one generator and not four")
    A("")
    if census:
        A("The brief planned adapters sampling coherent sequences from every GAN type. Measured")
        A("from the running code, cited by `file:line`, with no model instantiated:")
        A("")
        A("| generator | emits sequences | returns | evidence |")
        A("|---|---|---|---|")
        for r in census["rows"]:
            A(f"| `{r['gan_type']}` | **{r['emits_sequences']}** | `{(r['returns'] or '—')[:46]}` "
              f"| `{r['evidence']}` |")
        A("")
        A("**This is not a defect.** They are class-conditional *row* samplers built for class")
        A("balancing and they do that job. It is a statement about what they cannot be reused")
        A("for: a row sampler cannot produce a price *path*, and the policy under test is")
        A("path-dependent — stops, grace windows, holding limits and take-profit all depend on")
        A("the order bars arrive in.")
        A("")
        A("CGAN is the only sequence-capable type and is unmaintained, so it was excluded by")
        A("clarification Q3. That leaves a **stationary block bootstrap** as the sole generator —")
        A("which the brief required as a baseline regardless. Roughly three quarters of the")
        A("planned work disappeared for minutes of probing.")
        A("")
        A(f"*{census['scorecard_note']}*")
        A("")

    A("## Calibration — the Tail-GAN criterion")
    A("")
    if cal:
        A("Compared **surrogate-on-synthetic against surrogate-on-real-panels**: the evaluator is")
        A("held constant so only the paths vary. Comparing against the 378 archived *freqtrade*")
        A("trades would have conflated synthetic-vs-real paths with the known surrogate-vs-")
        A("freqtrade gap (−5.30 ± 23.46 pp), and an unattributable failure is a kill here.")
        A("")
        A("Tolerances **pre-registered** in `research.md` R3 before the first run:")
        A("")
        A("| statistic | measured | tolerance | |")
        A("|---|---|---|---|")
        t = cal["tolerances"]
        A(f"| KS distance | **{cal['ks_distance']}** | ≤ {t['ks_max']} | "
          f"{'PASS' if cal['ks_distance'] <= t['ks_max'] else 'FAIL'} |")
        A(f"| ES ratio (5%) | **{cal['es_ratio']}** | {tuple(t['es_ratio'])} | "
          f"{'PASS' if t['es_ratio'][0] <= cal['es_ratio'] <= t['es_ratio'][1] else 'FAIL'} |")
        A(f"| drawdown ratio | **{cal['dd_ratio']}** | {tuple(t['dd_ratio'])} | "
          f"{'PASS' if t['dd_ratio'][0] <= cal['dd_ratio'] <= t['dd_ratio'][1] else 'FAIL'} |")
        A("")
        A(f"{cal['n_real_trades']} real trades against {cal['n_synthetic_trades']} synthetic, "
          f"block length {cal['block_length']}.")
        A("")
        A(f"**Caveat, stated rather than glossed**: {cal['caveat']}")
        A("")
    if sweep:
        A("### Block length swept, not chosen")
        A("")
        A("| L (bars) | KS | ES ratio | drawdown ratio | |")
        A("|---|---|---|---|---|")
        for k in sorted(sweep, key=int):
            r = sweep[k]
            A(f"| {k} | {r['ks_distance']} | {r['es_ratio']} | {r['dd_ratio']} | "
              f"{'PASS' if r['passed'] else 'FAIL'} |")
        A("")
        A("Reporting a single value as *the* answer would hide the sensitivity (FR-003a,")
        A("SC-006a).")
        A("")

    A("## Stress result")
    A("")
    if stress and "pnl_quantiles_pp_yr" in stress:
        q = stress["pnl_quantiles_pp_yr"]
        A(f"Policy: **{stress.get('policy')}**, {stress['n_scored']} paths scored of "
          f"{stress['n_paths']} ({stress['n_inadmissible']} inadmissible).")
        A("")
        A("```")
        A(f"P&L pp/yr    p5 {q['p5']:+.2f}   p25 {q['p25']:+.2f}   p50 {q['p50']:+.2f}   "
          f"p75 {q['p75']:+.2f}   p95 {q['p95']:+.2f}")
        A(f"loss-path fraction   {stress['loss_path_fraction']:.3f}")
        A("```")
        A("")
        A(f"**Verdict: {stress['verdict']}.** {stress['verdict_meaning']}")
        A("")
    elif stress:
        A(f"**{stress.get('verdict')}** — {stress.get('note', '')}")
        A("")
    else:
        A("Not run.")
        A("")

    A("## Limitations — properties of the method, not discoveries")
    A("")
    A("- **A shared block index preserves cross-pair co-movement at the cost of path")
    A("  diversity.** Pairs are resampled with one index, so pairs that crashed together still")
    A("  do. Resampling them independently would have understated joint drawdown — and the")
    A("  drawdown ratio is a calibration statistic, so the bias would have pointed straight at a")
    A("  gate it had to pass.")
    A("- **The evaluator's error is inherited.** Stress verdicts use feature 013's exit")
    A("  surrogate, whose residual divergence from freqtrade is measured (`roi` 22 against a")
    A("  real 13 on f07). A stress verdict is no more accurate than its evaluator.")
    A("- **A resampled path cannot contain a market state the data has never shown.** This")
    A("  measures orderings of observed conditions, not the true data-generating process.")
    A("")
    A("## Scope")
    A("")
    A("- The multi-task GAN variants (MT_DDPM, MT_WGAN, MT_CTAB_GAN) were **out of scope** and")
    A("  appear nowhere in these results (SC-012).")
    A(f"- No policy was trained on synthetic data (FR-017): **{'verified' if train['passed'] else 'CHECK FAILED'}** — "
      f"{len(train['hits'])} training entry points found in this feature's modules.")
    A("- Protocol §5's gates apply to the **policies this harness vetoes**, not to the harness")
    A("  itself (clarification Q1, SC-013). A tool has no annualised return.")
    A("")
    A("## Reproduction")
    A("")
    A("```")
    A("python -m research.labelfree.stress.gencensus")
    A("python -m research.labelfree.stress.bank --build --folds f05..f14 --block 192 "
      "--paths 1000 --seed 7")
    A("python -m research.labelfree.stress.calibrate --bank <bank>")
    A("python -m research.labelfree.stress.run --bank <bank> --policy incumbent")
    A("```")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        ROOT / "research/label-free-study/briefs/results" / f"06-{date.today()}.md"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document())
    chk = synthetic_training_check()
    print(f"FR-017 synthetic-training check: {'PASS' if chk['passed'] else 'FAIL'} "
          f"({len(chk['hits'])} hits)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

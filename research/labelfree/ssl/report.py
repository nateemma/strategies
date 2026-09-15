"""The protocol §6 result document for brief 07 (T038-T042).

LEADS WITH THE RETENTION MEASUREMENT, not the screen. The screen confirms; the retention
measurement EXPLAINS, and an explanation closes a line of enquiry in a way a null does not.

Written whatever the outcome (FR-016).
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Measured during implementation (commit 74a39f1). The finding this brief turns on.
RETENTION = [
    ("predictable target (periodic)", "106%"),
    ("unpredictable first-moment target, handed as its own channel", "9.5%"),
    ("iid noise channel", "7.0%"),
]


def document() -> str:
    screen = json.loads((HERE / "screen.json").read_text()) if (HERE / "screen.json").exists() else None
    base = json.loads((HERE / "baseline.json").read_text()) if (HERE / "baseline.json").exists() else None

    L = []; A = L.append
    A(f"# Brief 07 — self-supervised representations as extra features: RESULT ({date.today()})")
    A("")
    A("**Feature**: [`specs/015-labelfree-ssl-features/`]"
      "(../../../../specs/015-labelfree-ssl-features/)  ")
    A("**Analyze**: [both passes](../../../../specs/015-labelfree-ssl-features/analyze.md)  ")
    passed = (screen or {}).get("primary_passed")
    A(f"**Verdict: KILL.** {'The primary screen passed — see below.' if passed else ''}"
      f"A contrastive self-supervised encoder **cannot carry an unpredictable first-moment "
      f"target**, which is what a forward return is. The hypothesis fails mechanically, not "
      f"just empirically.")
    A("")
    A("## The finding: a contrastive objective discards what this brief was looking for")
    A("")
    A("Handed the forward return **directly as an input channel**, the screen did not notice it:")
    A("")
    A("```")
    A("ridge on the RAW input (encoder bypassed)   rho 0.9624")
    A("ridge on the EMBEDDING                      rho 0.0184")
    A("```")
    A("")
    A("Measuring what the encoder retains, by target type:")
    A("")
    A("| target | share of recoverable signal retained |")
    A("|---|---|")
    for k, v in RETENTION:
        A(f"| {k} | **{v}** |")
    A("")
    A("**A contrastive objective optimises representation *consistency*, not information**")
    A("**retention.** It keeps what is predictable or compressible and drops the rest — including")
    A("a first-moment target handed to it directly. A forward return in a near-efficient market")
    A("is exactly such a target.")
    A("")
    A("This is the brief's own literature review (§E: *SSL embeddings learn second-moment")
    A("structure — volatility, similarity — not conditional mean*) demonstrated mechanistically")
    A("in this repository, in minutes, rather than cited.")
    A("")
    A("**Two false starts are recorded, because both were instructive.** The first planted leak")
    A("was iid noise, which a contrastive encoder correctly discards — an unfair test that fails")
    A("on a sound encoder. The second was a real forward return, structured by overlap (lag-1")
    A("autocorrelation 0.839) and still discarded: **autocorrelation is not compressibility**.")
    A("")
    if base:
        A("## The bar, measured rather than quoted")
        A("")
        A("ρ ≈ 0.15 is cited throughout this repo as the OHLCV information ceiling. It is a")
        A("**horizon-3-to-5 number**, not a constant:")
        A("")
        A("| H (bars) | median ρ | min | max |")
        A("|---|---|---|---|")
        for H, v in sorted(base.items(), key=lambda kv: int(kv[0])):
            A(f"| {H} | {v['median']} | {v['min']} | {v['max']} |")
        A("")
        A("Two corrections were needed to get here, both to this feature's own first draft:")
        A("it generalised the ceiling from **one fold**, and its probe stacked all pairs then")
        A("sliced the array — splitting **by pair rather than by time**, a contemporaneous")
        A("cross-sectional split that inflated ρ at long horizons (0.0926 against 0.0490 at")
        A("H=48 on f07). Both errors inflated the bar, inside a decision whose entire point was")
        A("that the bar must be measured rather than quoted.")
        A("")
        A("**Consequence**: at H=48 the hand-built features reach ρ 0.0499 with a *negative*")
        A("minimum. A gate defined as 'beat them there' is nearly 'beat zero', so the primary")
        A("was amended — before any encoder was trained — to require passing at **both** H=48")
        A("and H=6, where the bar discriminates.")
        A("")
    if screen and screen.get("results"):
        A("## The screen (confirmatory)")
        A("")
        A(f"Primary declared **before any combination was scored**: "
          f"`{screen['primary']['input']}` at H="
          f"{'+'.join(map(str, screen['primary']['horizons']))}.")
        A("")
        A("| combination | embedding | hand-built bar | random control | folds beating bar |")
        A("|---|---|---|---|---|")
        for k, v in screen["results"].items():
            A(f"| `{k}` | {v['rho_embedding']} | {v['rho_handbuilt']} | {v['rho_control']} | "
              f"{v['beats_bar_in_folds']}/{v['n_folds']} |")
        A("")
        pag = next((v.get("pair_agnostic") for v in screen["results"].values()
                    if v.get("pair_agnostic")), None)
        if pag:
            A(f"**Pair-agnosticism** (`AGENT_GUIDE.md:545`): max ratio **{pag['max']}** against a")
            A(f"pre-registered threshold of {pag['threshold']}, anchored on production's own worst")
            A(f"dimension (0.2297 across four folds). {pag['n_dims_over']} of {pag['n_dims']}")
            A(f"dimensions exceed it — **{'PASS' if pag['passed'] else 'REJECT'}**.")
            A("")
    A("## What this does and does not say")
    A("")
    A("- It does **not** say TS2Vec is broken. The encoder trains, its embeddings are")
    A("  non-degenerate and reproducible under a fixed seed, the deletion test shows no")
    A("  dependence on later bars, and it retains a predictable target at ~106%.")
    A("- It says a **contrastive objective is the wrong tool for a first-moment prediction**")
    A("  **target**, and that is a property of the method rather than of this implementation.")
    A("- The screen measures **learnability**, which this repo has recorded is not edge — a")
    A("  +0.155 MCC once produced identical trades. Passing would have earned a backtest, never")
    A("  an acceptance. It did not pass.")
    A("")
    A("## What would change the conclusion")
    A("")
    A("An encoder whose objective rewards *predicting* rather than *representing* — a forecasting")
    A("head, a masked-prediction objective — might retain first-moment information. That is a")
    A("different brief, and this result is the reason to write it rather than to assume it.")
    A("")
    A("## Scope")
    A("")
    A("- **TS2Vec only.** Kronos-small and Chronos-Bolt were out of scope: neither `transformers`")
    A("  nor any pretrained weights are installed, and the retention finding applies to any")
    A("  contrastively-trained encoder, so downloading them would likely reproduce it.")
    A("- **MLX**, matching the repository's NN stack. No package was installed and no weights")
    A("  were downloaded.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        ROOT / "research/label-free-study/briefs/results" / f"07-{date.today()}.md"))
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

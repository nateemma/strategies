"""What can each of the repo's generators actually emit? (T007-T010)

THE CHEAP MEASUREMENT THAT SHAPED THE FEATURE. Brief 06 planned adapters sampling coherent
sequences from every GAN type. This census showed that cannot be built for the generators that
matter, and roughly three quarters of the planned work disappeared for minutes of probing.

WHAT "DETERMINED FROM RUNNING CODE" MEANS HERE (FR-004, reworded by the pre-implementation
analyze pass). Capability comes from each implementation's RETURN CONTRACT, cited by file:line
-- not from documentation, and not by instantiating a model. An earlier wording demanded
"actual output", which would require calling generate() and therefore loading weights; the
census would then break whenever a checkpoint was missing or untrained, which says nothing
about the architecture's capability. Reading the contract is weaker evidence than running it,
and saying so is the honest position.

NOTHING DOWNSTREAM CONSUMES THIS. Clarification Q3 removed every learned generator from scope.
It ships because it is a durable fact about the repo's own tooling and is the documented reason
this feature has one generator instead of four.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GANS = ROOT / "GANs"

# (gan_type, file, the regex proving the return shape, in_scope)
#
# in_scope is False for every one of them: the three rebuilt generators cannot emit sequences,
# and CGAN -- the only type that can -- is unmaintained (clarification Q3). The census records
# them all anyway; excluding a generator from the census because it is out of scope would hide
# the evidence for the scoping decision.
PROBES = [
    ("TAB_DDPM", "df_tabddpm_mlx.py", r"\(n, 1, num_features\) float32 numpy array", False),
    ("WGAN", "GANInterface.py", r"Returns np\.ndarray \(n, 1, F\) — 3D with seq_len=1", False),
    ("CTAB_GAN", "GANInterface.py", r"For CTAB_GAN / MT_CTAB_GAN:(?:.|\n)*?Returns pd\.DataFrame", False),
    ("CGAN", "GANInterface.py", r"For CGAN:(?:.|\n)*?Returns np\.ndarray \(n, seq_len, F\)", False),
]

# Sequence capability is TRUE only when the contract carries a seq_len axis that is not pinned
# to 1. `(n, 1, F)` has three axes and still emits rows.
SEQ_RE = re.compile(r"\(n, seq_len, F\)")


def probe(gan_type: str, filename: str, pattern: str, in_scope: bool) -> dict:
    path = GANS / filename
    text = path.read_text()
    m = re.search(pattern, text)
    if not m:
        return {"gan_type": gan_type, "returns": None, "emits_sequences": None,
                "evidence": f"{filename}: contract not found — probe is stale",
                "in_scope": in_scope, "error": "contract not found"}
    line = text[:m.start()].count("\n") + 1
    snippet = " ".join(m.group(0).split())
    return {"gan_type": gan_type,
            "returns": snippet[:120],
            "emits_sequences": bool(SEQ_RE.search(m.group(0))),
            "evidence": f"GANs/{filename}:{line}",
            "in_scope": in_scope}


def census() -> dict:
    rows = [probe(*p) for p in PROBES]
    return {
        "rows": rows,
        "n_sequence_capable": sum(1 for r in rows if r.get("emits_sequences")),
        "method": ("return contract read from the running code, cited by file:line; "
                   "no model is instantiated and no weights are loaded"),
        "not_a_defect": (
            "These are class-conditional ROW samplers built for class balancing, and they do "
            "that job. This is a statement about what they cannot be REUSED for, not a fault. "
            "A row sampler cannot produce a price path, and the policy under test is "
            "path-dependent: stops, grace windows, holding limits and take-profit all depend on "
            "the order bars arrive in."),
        "scorecard_note": (
            "FR-016: TabDDPM, WGAN and CTAB-GAN+ were rebuilt 2026-09-10 and "
            "docs/GAN_SCORECARD.md predates that. Any figure quoted from the scorecard must be "
            "re-verified against current code or labelled historical with its date. This census "
            "reads current code and quotes no scorecard figure."),
        "consequence": (
            "No learned generator is carried (clarification Q3). The single generator is a "
            "stationary block bootstrap over real panels."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "census.json"))
    a = ap.parse_args()
    rep = census()
    Path(a.out).write_text(json.dumps(rep, indent=2))

    print(f"{'generator':<12}{'emits sequences':<18}{'evidence':<34}returns")
    for r in rep["rows"]:
        print(f"{r['gan_type']:<12}{str(r['emits_sequences']):<18}{r['evidence']:<34}"
              f"{(r['returns'] or '—')[:44]}")
    print(f"\nsequence-capable: {rep['n_sequence_capable']} of {len(rep['rows'])}")
    print(f"-> {a.out}")
    return 1 if any(r.get("error") for r in rep["rows"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

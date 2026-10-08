"""Allowlist duplication guard for the ATR-stop / grace-window exit arithmetic.

SC-002 / FR-003 (specs/021-label-exit-coordination). See research.md R3 for why this is an
explicit allowlist and not an AST or text-pattern matcher: the invariant we care about is
"nobody re-implements this arithmetic", and re-implementing it requires NAMING the parameters
it reads (`atr_stoploss_multiplier`, `atr_stoploss_cap`, `atr_stoploss_floor`,
`stoploss_grace_level`, `stoploss_grace_hours`, `use_volume_confirmation_stoploss`). A pattern
matcher for "a negated multiply followed by a clamp" would miss a copy written with
`np.clip`, fire on unrelated clamps, and create false confidence in exactly the invariant it
is supposed to protect (R3's rejected alternatives).

THIS TEST IS EXPECTED TO FAIL until T006 lands `utils/ExitPolicy.py` and makes
`Framework/BaseStrategy.py` / `NNPredict/NNPredictStrategy.py` thin callers over it. Until
then, `utils/ExitPolicy.py` is on the allowlist but absent from the scan (no file there yet),
which fails this test's equality check -- correctly: there is currently no canonical home for
the arithmetic to point at.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

#: the exit-arithmetic parameter names (research.md R3). Naming one of these is what makes a
#: file a candidate duplicate -- configuring a value also names it, which is why the allowlist
#: below exists and is organised by role rather than being empty.
PARAMETER_NAMES: tuple[str, ...] = (
    "atr_stoploss_multiplier",
    "atr_stoploss_cap",
    "atr_stoploss_floor",
    "stoploss_grace_level",
    "stoploss_grace_hours",
    "use_volume_confirmation_stoploss",
)

_PATTERN = re.compile(r"\b(?:" + "|".join(PARAMETER_NAMES) + r")\b")

#: user_data/strategies
STRATEGIES_ROOT = Path(__file__).resolve().parents[2]

#: Directory names pruned outright, by exact match, wherever they occur in the tree.
_SKIP_DIR_NAMES = frozenset({"__pycache__", ".venv", "specs"})


def _is_skipped_dir(name: str) -> bool:
    """Prune caches, the venv, spec-kit scaffolding and any hidden/tooling directory.

    Hidden directories (leading '.') are never source: `.git`, `.idea`, `.ruff_cache`,
    `.specify`, per-feature `.cache/` dirs, and `regime/.scratch/` (itself declared ephemeral
    in `regime/.gitignore`) hold ~150K generated per-fold file copies that would make this
    test slow and non-deterministic across machines without changing its answer -- none of
    them currently reference the exit parameters, verified by scanning regime/.scratch/
    directly when this test was written.
    """
    return name in _SKIP_DIR_NAMES or name.startswith(".")


def _iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_skipped_dir(d)]
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(dirpath) / filename


def scan_exit_parameter_references(root: Path = STRATEGIES_ROOT) -> frozenset[str]:
    """Return the set of files (POSIX paths, relative to `root`) naming any parameter."""
    found = set()
    for path in _iter_py_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _PATTERN.search(text):
            found.add(path.relative_to(root).as_posix())
    return frozenset(found)


#: Extending this allowlist is a DELIBERATE act, not a way to silence a failing test. Every
#: entry below carries a reason. A new entry needs one too: either it legitimately
#: CONFIGURES one of these parameters (sets a value, forwards it by name, or reads the live
#: configured value for an offline calibration/search), or it is a test. A file that computes
#: the stop LEVEL itself -- the clamp, the grace branch, the running-minimum/trail, the
#: volume-tighten rescale -- is a duplicate of `utils.ExitPolicy` and does not belong here;
#: it belongs refactored into a caller of it instead (T006's job, not this allowlist's).
ALLOWED_REFERENCES: frozenset[str] = frozenset({
    # --- canonical primitive --------------------------------------------------------------
    # Owns the arithmetic (atr_stop_fraction, atr_stop_fraction_array,
    # effective_stop_fraction) per contracts/exit-policy.md. The array form is primary and
    # the scalar delegates to it, so the clamp expression exists ONCE even though it is
    # consumed both per-trade (live) and per-bar (offline, ~200k bars/pair).
    "utils/ExitPolicy.py",

    # --- thin callers: must NAME the parameters to pass them through to the primitive ------
    "Framework/BaseStrategy.py",
    "NNPredict/NNPredictStrategy.py",
    # Was a FOURTH copy of the clamp -- beyond the three catalogued in research R1 -- found
    # by this very test on first run. Refactored into a caller of utils.ExitPolicy (FR-002)
    # rather than allowlisted as an exception; it still names the parameters because it
    # forwards them from a dict. Its 233 labelfree tests passed unchanged across the
    # refactor, which is what licensed touching closed work.
    "research/labelfree/es/exits.py",

    # --- strategies/configs that CONFIGURE a value, never implement the clamp --------------
    # Plain class-attribute overrides of the ATR multiplier/floor/cap triple.
    "NNMT/NNMT_Band20.py",
    "NNMT/NNMT_Band30.py",
    "NNMT/NNMT_Band40.py",
    "NNMT/NNMT_Band50.py",
    "NNNC/NNNC_Band20.py",
    "NNNC/NNNC_Band25.py",
    "NNNC/NNNC_Band30.py",
    "NNNC/NNNC_Band50.py",
    "NNNC/NNNC_Ora_Comb.py",
    "NNNC/NNNC_Ora_StopMid.py",
    "NNNC/NNNC_Ora_StopTight.py",
    "NNNC/NNNC_Ora_StopWide.py",
    # Class-attribute overrides (multiplier/floor/cap) plus hyperopt DecimalParameter
    # declarations (grace hours/level) -- still configuration, never arithmetic.
    "NNMT/NNMTStrategy.py",
    "NNNC/NNNCStrategy.py",
    # A frozen dataclass of pinned baseline values plus a CLI-arg builder
    # (`--set-sell key=value`) that forwards them by name; computes nothing.
    "regime/exit/baseline.py",
    # `--stoploss`'s help string names atr_stoploss_multiplier/floor/cap only to tell the
    # reader where the REAL lever lives; `--set-sell KEY=VALUE` is a generic passthrough.
    "regime/run_arm.py",
    # Writes the parameter names into a GENERATED strategy subclass's class attributes /
    # JSON sell-params for calibration backtests -- the same configuration pattern as the
    # NNNC_Band*/NNNC_Ora_* files above, not an independent implementation.
    "research/labelfree/es/calibrate.py",
    # Reads the LIVE incumbent value (regex off BaseStrategy.py's source, or off the
    # strategy's own hyperopt JSON) to build search-space bounds. Reads and bounds a
    # configured value; does not compute a stop level.
    "research/labelfree/es/space.py",

    # --- tests -------------------------------------------------------------------------------
    "NNPredict/tests/test_stop_aware_target.py",
    "research/labelfree/tests/test_es_exits.py",
    "research/labelfree/tests/test_es_space.py",
    "Framework/tests/test_exit_policy_single_source.py",  # this file: names the parameters
                                                            # as literal strings, computes
                                                            # nothing.
    "Framework/tests/test_exit_policy_characterisation.py",  # T004/SC-001: asserts the
                                                               # refactored path reproduces a
                                                               # PRE-refactor fixture bar for
                                                               # bar; builds strategy instances
                                                               # with these as attributes,
                                                               # computes no stop level itself.
})


def test_exit_parameter_references_match_allowlist():
    found = scan_exit_parameter_references()
    unexplained = sorted(found - ALLOWED_REFERENCES)
    stale = sorted(ALLOWED_REFERENCES - found)

    if unexplained or stale:
        lines = []
        if unexplained:
            lines.append(
                "Files reference an exit-arithmetic parameter name "
                f"({', '.join(PARAMETER_NAMES)}) but are NOT on the allowlist in "
                "Framework/tests/test_exit_policy_single_source.py:"
            )
            for f in unexplained:
                lines.append(
                    f"  - {f}\n"
                    "    Decide which this is, then act on it -- do not just add the file:\n"
                    "    * CONFIGURING (sets/forwards/reads a value, computes no stop "
                    "level) -> add it to the allowlist's 'configures a value' section "
                    "with a one-line reason.\n"
                    "    * IMPLEMENTING (computes the clamp, the grace branch, the "
                    "trailing/running-minimum, or the volume-tighten rescale) -> it is a "
                    "duplicate of the arithmetic. Refactor it into a caller of "
                    "utils.ExitPolicy instead of allowlisting it."
                )
        if stale:
            lines.append(
                "Files are on the allowlist but no longer reference any exit-arithmetic "
                "parameter (or do not exist yet):"
            )
            for f in stale:
                lines.append(f"  - {f}")
            lines.append(
                "If this is utils/ExitPolicy.py, T006 has not landed yet and this failure "
                "is expected pre-refactor. Otherwise remove the stale entry."
            )
        raise AssertionError("\n".join(lines))

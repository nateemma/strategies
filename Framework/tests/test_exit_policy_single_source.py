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
    # Added for feature 021's FR-001 amendment (T021 follow-up): custom_exit_decision's
    # take_profit/max_hold thresholds are now shared-primitive arithmetic too, not just the
    # stop's. metric_overbought/unclog_12h/unclog_1d's thresholds are NOT added here -- they
    # are fixed literals with exactly one occurrence repo-wide (verified by grep), not
    # freqtrade `Parameter`s anyone could plausibly re-declare or override.
    "cexit_take_profit",
    "cexit_max_days",
    # Feature 028 C2: the ATR-scaled take-profit's multiplier. Its clamp lives once, in
    # utils.ExitPolicy.atr_take_profit; naming it elsewhere must be configuration only.
    "take_profit_atr_mult",
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


#: GENERATED, gitignored build artefacts (feature 023): per-component attribution arms written
#: by `evaluation/attribution.py` as `SimpleTrigger/<Base>__C_<component>.py`. Each is a
#: one-class subclass carrying a MERGED override dict (configuration, never arithmetic), and
#: they exist only after an attribution run -- allowlisting them would make this equality check
#: depend on whether that run has happened on this machine.
_GENERATED = re.compile(r"__C_[a-z_]+\.py$")


def _iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_skipped_dir(d)]
        for filename in filenames:
            if filename.endswith(".py") and not _GENERATED.search(filename):
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
    # T012 (specs/021-label-exit-coordination): `ExitConfig.from_operating_point`
    # reads atr_stoploss_multiplier/_cap/_floor and use_volume_confirmation_stoploss
    # off the live `BaseStrategy` class attributes, and stoploss_grace_hours/_level
    # off NNNC_MLX_Open.json's pinned sell params, then forwards them into
    # `ExitParams` by keyword. Configures a value for the outcome table's exit
    # model; computes no stop level itself (`simulate_forward` does).
    "research/outcomes/table.py",

    # --- configures a value: cexit_take_profit/cexit_max_days only (feature 021's FR-001
    # amendment added these two tokens; see PARAMETER_NAMES's comment for why
    # metric_overbought/unclog's literals were NOT added). Every entry below either
    # declares/overrides the freqtrade Parameter, forwards it by name/dict, reads its live
    # value for a derived bound, or mentions the pinned value in a comment/docstring --
    # none compute the take_profit/max_hold comparison itself (that is
    # `utils.ExitPolicy.custom_exit_decision`'s job as of this amendment).
    "SimpleStrategies/TriggerOnly.py",     # feature 021's trigger-only arms: pins the
                                             # Open operating point as class attributes
                                             # (minimal_roi/stoploss/atr_*/sell_params) so
                                             # every arm closes the way the outcome table
                                             # simulated it. Sets values ONLY -- the stop
                                             # level, the grace branch and the
                                             # take-profit comparison all come from
                                             # BaseStrategy -> utils.ExitPolicy. Its six
                                             # subclasses set TRIGGER_NAME and nothing
                                             # else, which is why they are NOT here.
    "Anomaly/NNAnomalyStrategy.py",            # toggles .optimize on the live BaseStrategy
                                                 # Parameter by name; sets no value, computes
                                                 # nothing.
    "Debug/DebugNNMTStrategy.py",              # DecimalParameter/IntParameter declarations
    "Debug/DebugNNStrategy.py",                # (and a pinned sell_params dict), same
    "Debug/DebugRegimeIndicator.py",           # pattern as NNMTStrategy/NNNCStrategy above --
    "Debug/DebugTradingType.py",               # configuration, never arithmetic.
    "NNMT/NNMT_Ctl_s42.py",                    # docstrings documenting a pinned JSON value
    "NNMT/NNMT_Ctl_s7.py",                     # ("Its .json pins the control configuration,
    "NNMT/NNMT_Lbl02_s42.py",                  # including cexit_take_profit 0.032") -- prose,
    "NNMT/NNMT_Lbl02_s7.py",                   # not code that computes anything.
    "NNMT/NNMT_MLX_Base.py",
    "NNNC/NNNC_MLX_thr02.py",                  # docstring: "notably cexit_take_profit 0.032
                                                 # instead of the tuned 0.024" -- documents a
                                                 # deviation, computes nothing.
    "NNNC/NNNC_Ora_L024.py",                   # docstrings: "ALIGNMENT test -- label target
    "NNNC/NNNC_Ora_L026.py",                   # X% against cexit_take_profit 2.4%" -- prose
    "NNNC/NNNC_Ora_L032.py",                   # comparing a label threshold to the pinned
                                                 # live value, computes nothing.
    "NNNC/_hold_window.py",                    # module docstring + a code comment name
                                                 # cexit_max_days only to explain which exits
                                                 # the mixin does NOT defer (it defers
                                                 # unclog_12h, whose 12h/0.5% literals are not
                                                 # scanned tokens -- see PARAMETER_NAMES);
                                                 # computes no take_profit/max_hold decision.
    "SimpleStrategies/Ehlers_Ichimoku.py",     # a pinned sell_params dict (class attribute),
                                                 # same pattern as the Debug/* files above.
    # --- feature 023 (SimpleTrigger) -- all CONFIGURING; none computes a stop level ---------
    "SimpleTrigger/presets.py",                # literal preset dicts copied from the resolved
                                                 # reference classes (values, not arithmetic).
    "SimpleTrigger/components.py",             # neutral override values per component.
    "SimpleTrigger/coupling.py",               # maps label targets to cexit_take_profit /
                                                 # cexit_max_days VALUES via ExitPolicy's own
                                                 # net_from_price_ratio; no stop arithmetic.
    "SimpleTrigger/tests/test_coupling.py",    # asserts those mapped values.
    "SimpleTrigger/tests/test_presets.py",     # asserts preset attribute values.
    # 023 Addenda 3-7 momentum-exit arms: each sets SELL_OVERRIDES cexit_take_profit=10.0 (a
    # VALUE, "take-profit off") plus HoldWindowMixin's hour count; no stop arithmetic.
    "SimpleTrigger/SimpleTrigger_Breakout_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutVol_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutSqueeze_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutBb_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutGbb_MomExit.py",
    "SimpleTrigger/SimpleTrigger_Gbb_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutGm_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutVolGm_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutSqueezeGm_MomExit.py",
    "SimpleTrigger/SimpleTrigger_BreakoutGbbNoFresh_MomExit.py",
    # feature 024 C1: builds stub attributes (incl. use_volume_confirmation_stoploss) to prove
    # the trade hook is neutral; the stop it asserts is computed by ExitPolicy.atr_stop_fraction.
    "Framework/tests/test_exit_params_trade_hook.py",
    "utils/FamilyExits.py",                    # 024 C2: maps override NAMES to ExitParams fields
                                                 # and dataclasses.replace()s VALUES per trade;
                                                 # the stop/exit arithmetic stays in ExitPolicy.
    "utils/tests/test_family_exits.py",        # builds ExitParams fixtures, asserts the swap.
    # 025 US2 tuning configs: each sets ONE reversion FAMILY_EXIT_OVERRIDES value (atr floor /
    # volume-confirm flag); no stop arithmetic.
    "SimpleTrigger/SimpleTrigger_Tune3F05.py",
    "SimpleTrigger/SimpleTrigger_Tune3F08.py",
    "SimpleTrigger/SimpleTrigger_Tune4VcOff.py",
    "evaluation/fingerprint.py",               # lists the attribute NAMES it hashes; reads
                                                 # values, computes nothing from them.
    "evaluation/tests/test_fingerprint.py",    # tweaks one attribute to prove the hash moves.
    "utils/StopInvariants.py",                 # READS atr_stoploss_floor/stoploss_grace_hours
                                                 # to CHECK the stop the parent returned
                                                 # (bound = open*(1+floor)); returns that value
                                                 # unchanged and never computes a stop itself.
    "utils/tests/test_stop_invariants.py",     # sets those attributes on a stub.
    "Framework/tests/test_post_grace_stop_signfix.py",
                                                 # CONFIGURING: sets grace hours/level on a
                                                 # stub to drive the live post-grace branch.
                                                 # Its `_freqtrade_stop_price` helper restates
                                                 # FREQTRADE's own `adjust_stop_loss`
                                                 # arithmetic (`current_price * (1 -
                                                 # abs(ratio))`), deliberately, as an
                                                 # INDEPENDENT oracle -- the defect it pins was
                                                 # invisible in the returned ratio and only
                                                 # shows in the stop PRICE that ratio causes.
                                                 # It computes no stop LEVEL of ours: it never
                                                 # touches the ATR clamp, the grace branch or
                                                 # the volume tighten, all of which it reaches
                                                 # only through BaseStrategy.custom_stoploss.
    "regime/gbb_dist.py",                      # a print() string reporting the live
                                                 # cexit_take_profit value; computes nothing.
    "regime/path_label_sell.py",               # a literal preempt-level list with a comment
                                                 # naming cexit_take_profit as its source;
                                                 # computes a PRE-EMPTION DIAGNOSTIC (how often
                                                 # a forward move would have crossed each
                                                 # literal level), not the exit decision.
    "regime/trade_forensics.py",               # a dict mapping close-reason labels to the
                                                 # Parameter name they came from, plus a CLI
                                                 # help string -- forensic reporting, computes
                                                 # no exit decision.
    "research/labelfree/es/__init__.py",       # a planning-notes comment naming the token.
    "research/labelfree/es/panel.py",          # reads cexit_max_days's search-space UPPER
                                                 # BOUND (off space.py's DIMS) to size a
                                                 # leak-purge window; reads and bounds a
                                                 # configured value, computes no exit decision.
    "research/labelfree/es/report.py",         # report-text generation naming the token in a
                                                 # literal string; computes nothing.

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
    "Framework/tests/test_exit_policy_simulate_forward.py",  # T009: names
                                                               # use_atr_adaptive_stoploss via
                                                               # ExitParams kwargs to pin the
                                                               # close-reason state machine;
                                                               # delegates all arithmetic to
                                                               # simulate_forward.
    "Framework/tests/test_exit_policy_grace.py",  # T010: names the grace parameters as
                                                    # ExitParams kwargs / stub-strategy
                                                    # overrides to pin the grace asymmetry;
                                                    # computes no stop level itself.
    "Framework/tests/test_exit_policy_fixed_params.py",  # T011/FR-019: names the parameters
                                                           # as literal expected values read
                                                           # off BaseStrategy / the Open JSON;
                                                           # computes no stop level itself.
    "research/outcomes/tests/test_table.py",  # T012/T014-T019: names the parameters as
                                                # ExitConfig/ExitParams kwargs and as literal
                                                # expected values (the Open operating point's
                                                # grace hours/level) to pin ExitConfig's
                                                # sourcing; computes no stop level itself.
    # FR-001's 2026-10-08 amendment (T021 follow-up): characterises/pins custom_exit's
    # decision and simulate_forward's new branches via ExitParams/_StubStrategy kwargs;
    # delegates every comparison to BaseStrategy.custom_exit / custom_exit_decision /
    # simulate_forward, computes none of it itself.
    "Framework/tests/test_custom_exit_characterisation.py",
    "Framework/tests/test_exit_policy_custom_exit_decision.py",
    "Framework/tests/test_exit_policy_simulate_forward_amendment.py",
    # 2026-10-08 fee double-count fix: names the grace/stop parameters as ExitParams
    # kwargs only to build fixtures for `simulate_forward`/`custom_exit_decision`;
    # delegates every comparison to those functions, computes no stop level itself.
    "Framework/tests/test_exit_policy_fee_netting.py",

    # --- features 027-028 (target-aware / recovery-aware exits) --------------------------------
    # The exit REPLAY: simulates freqtrade's stop PLACEMENT around our levels (custom_stoploss
    # called with the candle high, dir_correct, round-up to the tick, ROI suppression; see
    # docs/EXIT_AND_BACKTEST_MECHANICS.md). Every stop LEVEL comes from ExitPolicy
    # (effective_stop_fraction, atr_take_profit); what stays local is freqtrade's CALL
    # SCHEDULE (fixed at fill, ratchet after grace) and the 027 coupling hooks (Rules) that
    # override grace hours/level by name. 99.6-99.7% of recorded exits reproduced.
    "research/exit_coupling/paths.py",
    "research/exit_coupling/feasibility.py",   # reads the resolved grace hours/level off a
                                                 # class's sell_params into ExitParams; computes
                                                 # nothing.
    "research/exit_coupling/tests/test_paths.py",  # builds ExitParams fixtures (grace, ATR
                                                     # band) for the replay; asserts its exits.
    # 027/028 exploratory arms: SELL_OVERRIDES stoploss_grace_hours=12.0 (C4) -- a VALUE.
    "SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_XC4.py",
    "SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_XC42.py",
    # 028 promoted C2: FAMILY_EXIT_OVERRIDES take_profit_atr_mult=k -- a VALUE.
    "SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_P2.py",
    # research/st_grace arms: SELL_OVERRIDES stoploss_grace_hours=24.0 -- a VALUE.
    "SimpleTrigger/SimpleTrigger_BreakoutGbb_MomExit_G24.py",
    "SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_P2_G24.py",
    "research/held_pair/screen.py",            # sets the grace hours on the study's ExitParams for
                                                 # the 027 replay; computes nothing.
    "research/st_grace/screen.py",             # reads the resolved grace hours and passes the C2
                                                 # multiplier to the 027 replay as Rules; computes
                                                 # nothing.
    "utils/tests/test_exit_switches.py",       # sets the switch on stubs; asserts the
                                                 # resolved take-profit, computes none of it.
    "NNNC/NNNC_Live_C2.py",                    # 029 liveness arm: FAMILY_EXIT_OVERRIDES
                                                 # take_profit_atr_mult=k -- a VALUE.
    # NNNC switch experiment: the same declaration on four models -- VALUES.
    "NNNC/NNNC_SW_ARO_C2.py",
    "NNNC/NNNC_SW_GBR_C2.py",
    "NNNC/NNNC_SW_GFM_C2.py",
    "NNNC/NNNC_SW_MLX_C2.py",
    # NNNC framework study (research/nnnc_framework/): stage-2 arms declare ONE value each --
    # ATR take-profit k (TK) or the post-grace stop's K/floor/cap scaled x2 (SS). VALUES.
    "NNNC/NNNC_FS_ARO_SS.py",
    "NNNC/NNNC_FS_GBR_SS.py",
    "NNNC/NNNC_FS_GFM_SS.py",
    "NNNC/NNNC_FS_MLX_SS.py",
    "NNNC/NNNC_FS_ARO_TK.py",
    "NNNC/NNNC_FS_GBR_TK.py",
    "NNNC/NNNC_FS_GFM_TK.py",
    "NNNC/NNNC_FS_MLX_TK.py",
    "NNNC/NNNC_FS2_ARO_SS.py",
    "NNNC/NNNC_FS2_GBR_SS.py",
    "NNNC/NNNC_FS2_GFM_SS.py",
    "NNNC/NNNC_FS2_MLX_SS.py",
    "NNNC/NNNC_FS2_ARO_TK.py",
    "NNNC/NNNC_FS2_GBR_TK.py",
    "NNNC/NNNC_FS2_GFM_TK.py",
    "NNNC/NNNC_FS2_MLX_TK.py",
    "research/nnnc_framework/build_arms.py",   # writes those declarations and .json/snapshot
                                                 # values (grace hours, take-profit); computes none.
    "research/nnnc_framework/diagnose.py",     # builds ExitParams levers (scaled K/floor/cap,
                                                 # grace, take-profit) for paths.replay; no arithmetic.
    "NNNC/tests/test_model_key_nnnc.py",       # 030: declares the C2 value on a stub to show
                                                 # an exit-only variant keeps its model key.
    "NNNC/tests/test_nnnc_merge.py",           # 029: sets grace hours/floor on a stub NNNC to
                                                 # drive StopInvariantMixin; computes nothing.
    "evaluation/tests/test_fingerprint_switches.py",  # declares grace 12.0 on a subclass to
                                                        # assert its hash; computes nothing.
})


def test_duplication_guard_can_fail():
    """T008: prove the guard above is not a tautology.

    A test that cannot fail is worthless -- and this repository has already lost a
    result to a wiring check that passed while the logic it guarded never ran (see
    MEMORY.md "Verify training steps by measurement"). So this plants a real file
    under `user_data/strategies/` that names `atr_stoploss_multiplier` and
    re-implements the clamp, calls `scan_exit_parameter_references` directly (the
    same function `test_exit_parameter_references_match_allowlist` uses), and
    asserts the planted file is flagged -- i.e. that it would fail the allowlist
    equality check above. The file is removed in a `finally` block so a crashed run
    cannot leave it behind; the final assertion re-scans and confirms it is gone.

    Observed failure message when this was run manually (T008), for the record:

        AssertionError: Files reference an exit-arithmetic parameter name
        (atr_stoploss_multiplier, ...) but are NOT on the allowlist in
        Framework/tests/test_exit_policy_single_source.py:
          - Framework/_planted_duplicate_exit_arithmetic.py
            Decide which this is, then act on it -- ...
    """
    planted = Path(__file__).parent / "_planted_duplicate_exit_arithmetic.py"
    planted.write_text(
        '"""T008 self-test plant -- re-implements the clamp. Must not survive the test."""\n'
        "atr_stoploss_multiplier = 2.5\n\n\n"
        "def clamp(x):\n"
        "    return max(min(x, -0.02), -0.04)\n"
    )
    try:
        found = scan_exit_parameter_references()
        relative = planted.relative_to(STRATEGIES_ROOT).as_posix()
        assert relative in found, (
            f"planted duplicate file {relative} was NOT flagged by the scan -- the "
            "duplication guard cannot detect a real duplicate and is worthless"
        )
        assert relative not in ALLOWED_REFERENCES, (
            "planted file name collided with an existing allowlist entry; rename it"
        )
    finally:
        planted.unlink(missing_ok=True)

    # Cleanup verified: the file is gone and no longer flagged.
    assert planted.exists() is False
    found_after = scan_exit_parameter_references()
    assert planted.relative_to(STRATEGIES_ROOT).as_posix() not in found_after


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

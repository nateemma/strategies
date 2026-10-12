# Research: Recovery-Aware Reversion Exits (028)

Every decision below goes into the pre-registration, and none may change after it is committed.

## R1 — The 2022 window, its folds and warm-up

**Decision**: three 100-day test folds, `e00` 2022-03-07 → 2022-06-15, `e01` → 2022-09-23 and
`e02` → 2023-01-01 (end exclusive; it touches f00's start without overlapping it). Universe: the 10
pairs of `config_wf.json`, unchanged.

**Rationale**:
- The reversion arms set `startup_candle_count = 4096` bars (42.7 days) so the volreg gate's
  2,880-bar windows are warm (026 R-warm-up).
- The last pair to list on binance.us is NEAR, on 2022-01-20.
- 2022-03-07 minus 42.7 days is 2022-01-23, so every pair has a full warm-up at the first test bar.
- Three equal folds use all 300 days. Fold ids are free strings: pool, compare and run_arm key on
  the literal `fold_id`; only `stage.py` and `books._halves` assume `fNN`, and neither is used.

**Verification (setup task, before pre-registration)**:
- a no-backtest probe that the volreg instrument is non-NaN for every pair at 2022-03-07, computed
  from data truncated at the warm-up start;
- `run_folds --dry-run` on a 2022 plan.

**Alternatives**:
- 90-day folds plus a 30-day remainder: below `min_final_days` 45, so it would be merged anyway.
- A start at 2022-02-01: NEAR would have only 12 days of warm-up.

## R2 — Is 2022 really unseen?

**Audit (2026-10-12)**:
- `regime/arms/` has 0 records with a 2022 timerange.
- `user_data/backtest_results/` has 0 SimpleTrigger runs overlapping 2022-03…12.
- The volreg threshold (1.3658) came from 2023+ trades (026 feasibility_stack).
- So 2022 is unseen for the four candidates, the gbb trigger and the gate.

**Caveat**:
- 325 NN-era backtests spanned 2022 (for example `NNNC_Unguarded` 2022-01-01 → 2025-08-31).
- The shared base exit (grace 6.5 h / −27%, ATR band, take-profit 2.2%) was settled in that era.
- 2022 is therefore not fresh for the base exit. That is common to control and candidates, so it
  shifts both arms' level, not the paired delta. The report states it.

## R3 — Feasibility condition (US1)

**Decision**, computed on the 2022 control trades:
1. **Fidelity first.** The null replay must reproduce at least 90% of recorded exits (reason AND
   bar). Otherwise exit 2: no verdict, and the replay must be fixed before going on.
2. **Recovery share** = stop exits whose price touches +2% gross after the stop, within the max
   hold, divided by all stop exits. This is the mechanism C4, C1 and C4+C2 act on (they keep
   stopped trades alive).
   - The floor is **0.50**: a majority of stop exits recover.
   - Reversion was 0.634 and 0.575 in 2023–26. Momentum, where C4 failed late, was 0.465 and 0.451.
   - The floor applies to C4, C1 and C4+C2. C2 alone changes the take-profit, not the stop, so
     the floor does not apply to it.
3. **Replay gain > 0** for the candidate on 2022, against the null replay. This applies to every
   candidate.
   - The 2× discount (027 calibration: arms realise 44–68%) is reported as the expected arm gain.
   - It is never a threshold: halving does not change a sign.
4. **Reported only:** the forgone P&L of the recovered stop exits (target net minus realised,
   × stake × scale, in pp).

A candidate PASSES if 1, the applicable part of 2, and 3 hold. If none passes, the feature stops
(exit 3) with the negative recorded.

**Alternative rejected**: 027's killed-winner and ATR-tercile conditions. They measure label-horizon
killing, which 027 showed is rare (2–3%).

## R4 — Decision rule (US2), as clarified (Q2: B)

A candidate is **ADOPTED** only if ALL of the following hold, with arm and control on e00–e02:
- **(a)** Σ per-fold `return_pct` (arm) − Σ per-fold `return_pct` (control) > 0. This is the
  "gain" everywhere in this feature, the same per-fold sum 027 and rule (d) use.
  - `compare_data`'s `delta_pp` (the observed paired trade-level delta) is reported but never
    decides; the two can differ in sign.
- **(b)** The paired P&L verdict is not WORSENED. It is computed by `compare_pooled.compare_data` on
  `halves.subset(arm, E)` and `halves.subset(ctl, E)`, where E = {e00, e01, e02}, called
  directly: the repository's one verdict implementation.
- **(c)** `risk.compare` (pinned seed and blocks) on the same subsets: Sharpe not WORSENED AND max
  drawdown not WORSENED.

**Verdict mapping (pre-registered).** Every value the two verdicts can return:

| verdict value | rule (b), P&L | rule (c), each risk statistic |
|---|---|---|
| IMPROVED | pass | pass |
| UNCHANGED | pass | pass |
| WORSENED | **fail** | **fail** |
| INCONCLUSIVE_UNDERPOWERED (< MIN_DIFF_TRADES paired differences) | pass, flagged | — |
| INCONCLUSIVE_ARM_LOSES (arm total ≤ 0) | pass, flagged | — |
| INCONCLUSIVE_ABSORBED (no trade differs) | → INCONCLUSIVE (liveness) | — |
| UNRESOLVABLE | → INCONCLUSIVE | → INCONCLUSIVE |

Why ARM_LOSES passes (b):
- That refusal exists because a delta against a broken or abstaining comparison arm measures the
  comparison arm (`compare_pooled` admissibility note).
- Here the comparison arm is the same strategy with today's exit, run on the same folds. An arm
  that loses less than it in a bear market is the effect being tested.
- The flag is shown in `verdicts.md`.

UNDERPOWERED passes (b) for the same reason UNCHANGED does: the rule is a persistence test, not a
significance test.
- **(d)** The arm's gain is > 0 in BOTH 2023–26 halves (f00–f07 and f08–f14).
  - It holds for C4, C2 and C1 per 027 `exploratory.md`.
  - For C4+C2 it is measured in setup, before pre-registration. If it fails there, C4+C2 is still
    run on 2022 for the record but can't be adopted.

**INCONCLUSIVE** instead of a verdict when either:
- the 2022 control has fewer than **200** trades (the 2023–26 pace is about 2.5/day, or about 750
  in 300 days, so 200 allows a gate that is mostly off in a bear market); or
- the candidate changed trades in **0** of 3 folds.

**Always reported**:
- the paired p;
- the 95% CI;
- its half-width as the window's power: the smallest gain this window could call IMPROVED.
  - `compare_data` gives the interval in pp per year (`ci_pp_yr`).
  - Half-width in pp over the window = (hi − lo) / 2 × years, where years = Σ test_days / 365
    (300/365).
  - This is the same unit as rule (a)'s gain.

**Run-to-run variance (constitution quality gate 5).**
- For any ADOPTED candidate whose gain is under 1 pp, its fold with the largest absolute per-fold
  gain is re-run under a fresh tag (`st_e22_<id>_rr`).
- Its `trades_hash` must equal the original; that shows the deterministic, model-free arm has
  zero run-to-run variance.
- If it differs, the verdict becomes INCONCLUSIVE and the cause is reported.

**Recommendation among the adopted**:
- the one with the largest 2022 gain;
- the combination only if C4+C2 is itself ADOPTED;
- a part that is not ADOPTED is never brought back through the combination.

## R5 — The C4+C2 class

**Decision**: `SimpleTrigger_RevOnlyVolreg2_XC42(SimpleTrigger_RevOnlyVolreg2_XC2)` with
`SELL_OVERRIDES = {**parent, "stoploss_grace_hours": 12.0}`. That is exactly XC4's override on top of
XC2's `_exit_params`.
- Its replay rule: `Rules(grace_hours=12.0, take_profit_atr_mult=2.8970711921574543)`.
- It runs on f00–f14 (`plan_st_xpl_rev_c42.json`) in setup; that is 2023–26 only, not the judging
  window.

**Arm identity (FR-006)**: the fingerprint cannot see method overrides. `confirm.py` therefore
identifies each arm by `strategy` (class name) in the record and refuses if two plans share a class.

## R6 — Pre-registration mechanics

**Decision**:
- `preregistration.json` holds candidates (class, plan tag, frozen value), folds, universe, floor,
  rule thresholds and minimum trades. `preregistration.md` is its readable copy.
- They are committed together with the five 2022 plan files and before any 2022 run.
- **Refusals by `confirm.py`:**
  - (i) any class constant differs from the JSON value: `TP_ATR_MULT`, `STOP_MIN_DEPTH`, the
    resolved `stoploss_grace_hours`;
  - (ii) any 2022 arm record's mtime is older than the pre-registration commit time
    (`git log -1 --format=%ct -- preregistration.json`);
  - (iii) `preregistration.json` has uncommitted changes. The readable `.md` is not checked; it
    receives the commit-sha follow-up commit.
- A deviation after commit is a new, dated section in `preregistration.md`, and the report lists it.

**Alternative rejected**: a timestamp inside the record. Arm records have none, and adding one
would change `run_arm`.

## R7 — Promotion design (US3; built only for what is ADOPTED)

- **C4**:
  - The grace length is an existing sell parameter and is shared by definition
    (`FamilyExits.FORBIDDEN_KEYS`).
  - Adoption = the adopting strategy declares `stoploss_grace_hours: 12.0`. No framework change.
  - A test asserts that a strategy declaring it resolves the same fingerprint as XC4. The
    fingerprint does see SELL_OVERRIDES.
  - Making grace per-family is out of scope (reversion-only scope).
- **C1**:
  - New field `ExitParams.stop_min_depth: float | None = None`, applied by one pure function in
    `ExitPolicy`. It is called from `BaseStrategy._resolve_normal_stop` on the post-grace path,
    which is the single source the replay's `paths._normal` also mirrors.
  - It can be overridden per family through `FAMILY_EXITS`.
- **C2**:
  - New field `ExitParams.take_profit_atr_mult: float | None = None`, with the clamp constants
    [0.01, 0.05] moved from `paths.py` into `ExitPolicy` (one copy).
  - It is resolved per trade AFTER family overrides: signal-bar `atr_pct_roll` at
    open_date − 1 timeframe, cached per trade, exactly as XC2 does.
- **Fingerprint (FR-010)**:
  - Adding fields changes `dataclasses.asdict(ExitParams)` inside `FAMILY_EXITS_RESOLVED`.
  - New fields whose value is None are omitted from the hashed dict, so every recorded `framework_sha`
    is unchanged.
  - A test pins the shas of `st_bgo_mx`, `st_book_rev2` and arm D.
- **Invariants (FR-012)**:
  - `stop_min_depth` must lie in [atr_stoploss_floor, 0); this is refused at class creation.
  - The runtime `StopInvariants` bound uses min(floor, stop_min_depth) as the loosest allowed stop.
- **Equivalence (FR-011)**: a promoted class with the switch on must reproduce the XC class's
  `trades_hash` on f00–f14 and e00–e02. The single-source allowlist test is updated for the new
  clamp site.

## R8 — Evidence labels

- 2023–26 results stay "EXPLORATORY (post-hoc)".
- 2022 results are "CONFIRMATORY (pre-registered), PROVISIONAL under quality gate 2". 2022 is a
  different regime but adjoins f00. Gate 2 asks for a temporally distant era, and one adjacent year
  is not one.

## R9 — What is out of scope

- the momentum family;
- per-family grace;
- re-tuning;
- NNNC runs (a later feature turns the adopted switch on for NNNC);
- book or portfolio combination.

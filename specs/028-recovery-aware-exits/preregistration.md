# Pre-registration: 028 Recovery-Aware Reversion Exits

**Committed before any 2022 backtest or replay.** The machine-readable copy is
[preregistration.json](preregistration.json); `confirm.py` reads only that file and refuses to run if
it has uncommitted changes. Commit: `e645016` (2026-10-12).

## What is judged

Control: `SimpleTrigger_RevOnlyVolreg2` (gated reversion alone, today's exit), plan `st_e22_rev2`.

| id | class | frozen value | recovery floor applies | 2023–26 gain early / late (pp) |
|---|---|---|---|---|
| C4 | `SimpleTrigger_RevOnlyVolreg2_XC4` | grace 6.5 h → **12.0 h**, same −27% depth | yes | +8.439 / +6.267 |
| C2 | `SimpleTrigger_RevOnlyVolreg2_XC2` | take-profit = **2.8970711921574543** × signal-bar ATR, clamped [0.01, 0.05] | no | +4.461 / +4.172 |
| C1 | `SimpleTrigger_RevOnlyVolreg2_XC1` | post-grace stop never tighter than **−0.03710764139923586** | yes | +3.285 / +1.878 |
| C42 | `SimpleTrigger_RevOnlyVolreg2_XC42` | C4 and C2 together | yes | +4.226 / +4.081 |

The 2023–26 gains are EXPLORATORY (027 `exploratory.md`; C42 from `baseline_c42.md`). They enter
only rule (d), by sign.

## Where

- **Folds:** e00 2022-03-07 → 2022-06-15, e01 → 2022-09-23, e02 → 2023-01-01 (100 days each, 0.822
  years).
- **Universe:** the 10 `config_wf.json` pairs. XRP has no binance.us data in 2022, so 9 trade, the
  same 9 for every arm (as in f00–f02).
- **Warm-up:** 4096 bars. Every pair with data has a full warm-up at 2022-03-07, and volreg is
  non-NaN there (`baseline.md`).
- **Freshness:** no SimpleTrigger arm has been backtested on 2022. The shared base exit dates from
  NN-era runs that spanned 2022; that applies to control and candidates alike (`baseline.md`).

## US1 feasibility (control trades only)

1. The null replay must reproduce **≥ 0.90** of recorded exits (reason AND bar), and the stop model
   must agree on ≥ 0.90. Otherwise: no verdict.
2. **Recovery share** = clean stop exits that touch +2% after the stop within the max hold, ÷ clean
   stop exits. **Floor 0.50**, applied to C4, C1 and C42 (not C2).
3. The candidate's **replay gain > 0**. 0.5 × that gain is reported as the expected arm gain.
4. Reported only: the P&L the recovered stop exits forgo.

A candidate that fails does not run on 2022 and is recorded STOPPED_AT_FEASIBILITY. If none passes,
the feature stops.

## US2 decision rule

**Gain** = Σ per-fold `return_pct` (arm) − Σ per-fold `return_pct` (control) over e00–e02.
`compare_data`'s `delta_pp` is reported only.

A candidate is **ADOPTED** only if ALL of these hold:
- **(a)** gain > 0;
- **(b)** the paired P&L verdict maps to pass;
- **(c)** both risk verdicts (Sharpe, max drawdown) map to pass;
- **(d)** its 2023–26 early AND late gains are > 0.

| paired P&L verdict | maps to | risk verdict | maps to |
|---|---|---|---|
| IMPROVED, UNCHANGED | pass | IMPROVED, UNCHANGED | pass |
| WORSENED | **fail** | WORSENED | **fail** |
| INCONCLUSIVE_UNDERPOWERED, INCONCLUSIVE_ARM_LOSES | pass, flagged | UNRESOLVABLE | inconclusive |
| INCONCLUSIVE_ABSORBED, UNRESOLVABLE | inconclusive | | |

**INCONCLUSIVE** instead of a verdict if:
- the control has fewer than **200** trades on e00–e02; or
- the candidate changed trades in **0** of 3 folds; or
- any mapped verdict is inconclusive.

**Always reported:** the paired p, the CI, and the window's power, which is the CI half-width in pp
over the window ((hi − lo)/2 × years).

**Gate 5:** an ADOPTED gain under 1 pp has its largest-|gain| fold re-run under a fresh tag. A
different `trades_hash` turns it INCONCLUSIVE.

**Recommendation:** the ADOPTED candidate with the largest 2022 gain. The combination only if C42
is itself ADOPTED; a part that is not ADOPTED is never brought back through it.

**Evidence label:** CONFIRMATORY (pre-registered), PROVISIONAL under quality gate 2 (2022 adjoins
f00).

## Deviations

None at commit. Any later change is added here, dated, with its reason.

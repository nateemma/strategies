# Report: Target-Aware Exits (027)

*PROVISIONAL — quality gate 2 (no distant era); halves split only.*

## SUMMARY

- **Verdict: STOP at the feasibility gate (US1).**
  - None of the five candidate couplings met its pre-declared rule in both halves, for either
    family.
  - By the spec, nothing was built (US2/US3 not run), and every recorded arm is untouched.
- **The measuring instrument is trustworthy.**
  - The exit replay reproduces 99.6% of recorded exits (reason AND bar) in both reference arms.
  - Getting there exposed five freqtrade backtest mechanics the first version got wrong (below).
    They are now recorded in `docs/EXIT_AND_BACKTEST_MECHANICS.md`.
- **What the exit does relative to the label target (≥ 2% within 48 bars):**
  - The stop rarely kills a trade on its way to the target: 2–3% of label-positive entries.
    Most targets are touched within ~9 bars, while the −27% grace stop is still in force.
  - But 45–63% of stop exits reach +2% LATER, within the 10-day max hold. Only 2–7% do so within
    the 12 h horizon.
  - The "stop too tight" problem is real, but on the max-hold timescale, not the label-horizon
    timescale the rules were written for.
- **Why reversion still failed.** For the reversion family, three couplings gain in the replay in
  BOTH halves:
  - grace extended to 12 h, +19.05 / +12.76 pp;
  - ATR-scaled take-profit, +8.21 / +8.31;
  - deeper stop at p90, +4.86 / +3.23.

  Each failed its rule's DESCRIPTIVE condition (killed winners, or the ATR-tercile hit-rate
  ratio), not its gain condition. Those conditions were written for horizon-scale killing, which
  the measurement shows is rare. See "Reading" for what that means and the options.

## Fidelity (T006): what the replay had to learn

The null replay (no coupling) must reproduce ≥ 90% of recorded exits before any verdict.
- **First version:** momentum 79.7% (FAIL), reversion 93.2%.
- **After five fixes:** 99.6% in both, with the stop model agreeing on 99.4–99.6% of stop exits.

| fix | freqtrade source | effect |
|---|---|---|
| `custom_stoploss` is called with the candle HIGH | `ft_stoploss_adjust`: `current_rate=(bound or current_rate)` | at-market stops sit under the high |
| no stop update once the low is through it | `dir_correct` | the old stop fires |
| stop rounded UP to the price tick | `adjust_stop_loss` → `price_to_precision(ROUND_UP)` | coarse-tick pairs (ZEC 0.1): flat candles trigger the stop |
| ROI suppressed per candle by `enter_long` | `ignore_roi_if_entry_signal` | momentum ROI agreement 42% → 99.9% |
| force exit stamped one bar after the last candle | backtest end handling | 0% → 100% on force exits |

## Measurements (feasibility.md has all rows)

| | momentum early | momentum late | reversion early | reversion late |
|---|---|---|---|---|
| trades | 3,411 | 2,745 | 2,092 | 1,292 |
| label-positive (touch ≤ 48 bars) | 1,864 | 1,421 | 1,384 | 832 |
| (a) stop exits touching +2% after, within 48 bars of entry | 2.0% | 1.5% | 6.7% | 4.5% |
| (a) … within the 10-day max hold | 46.5% | 45.1% | 63.4% | 57.5% |
| (b) killed winners (stop breached before the touch) | 1.9% | 1.7% | 3.4% | 2.4% |
| (b) winners' drawdown before touch, p50/p90/p95 | 0.7/2.9/3.9% | 0.7/2.9/4.0% | 0.9/3.7/4.8% | 0.7/3.4/4.9% |
| (c) bars to touch, median | 9 | 10 | 9 | 8 |
| (c) open past 48 bars without a touch: n / pp | 609 / −12.58 | 509 / −8.17 | 287 / −2.18 | 134 / −1.68 |
| (d) touched +2% but exited below it: n / mean give-back | 908 / 3.3% | 675 / 3.3% | 264 / 3.2% | 142 / 3.8% |

## Couplings: replay gain (pp, early / late) and verdict

Values were chosen on f00–f07; the gain is against the null replay.

| | momentum | reversion |
|---|---|---|
| C1 stop no tighter than winners' p-th drawdown | p80: +0.00 / +0.00 (p90: −0.12 / +1.21) — NOT SUPPORTED (killed share < 15%) | **p90 (−3.7%): +4.86 / +3.23** — NOT SUPPORTED (killed share < 15%) |
| C2 take-profit = k × ATR | N/A (no take-profit) | **k = 2.90: +8.21 / +8.31** — NOT SUPPORTED (tercile hit-rate ratio not < 0.5) |
| C3 at 48 bars without touch | break-even: +0.64 / −0.63 (close: −0.31 / −2.22) — NOT SUPPORTED | break-even: −0.82 / +0.26 (close: −1.21 / +0.23) — NOT SUPPORTED |
| C4 grace window to 12 h | same depth: +16.94 / −2.44 (p95 depth: −25.14 / −10.34) — NOT SUPPORTED | **same depth: +19.05 / +12.76** (p95: −12.18 / +6.85) — NOT SUPPORTED (killed-in-window share of LP < 10%) |
| C5 lock after the target | λ 0.5: −0.50 / −4.64 (λ 0: −8.07 / −6.59) — NOT SUPPORTED | λ 0: −4.32 / −2.77 (λ 0.5: −4.71 / −2.64) — NOT SUPPORTED |

Replays ignore re-sequencing: a trade held longer occupies its pair, and later entries change
(023 Addendum 7). They choose what to test; they do not measure the gain.

## Reading

1. **Locking gains after the target (C5) costs money in both families, in both halves.** The
   trades that touch +2% and give it back are outweighed by those that go on to bigger wins. The
   exit should not be coupled to the target on the upside.
2. **The horizon time rule (C3) is noise in both families.** Trades still open at 12 h without a
   touch lose (momentum −12.6 / −8.2 pp), but closing them, or moving the stop to break-even,
   does not recover it in the replay.
3. **Momentum's exit is consistent with its target.** Every coupling is ≤ 0 late, except C1 p90
   (+1.21 late, −0.12 early).
4. **Reversion's exit is too tight for the max hold, and the rules missed it.**
   - Extending grace to the label horizon (C4), an ATR-scaled take-profit (C2) and a modestly
     deeper stop (C1 p90) each gain in both halves of the replay.
   - The descriptive conditions guarding them assumed the damage would show up as label-horizon
     "killed winners". It shows up instead as stop exits that recover later (57–63%).
   - These were pre-declared conditions, so the verdict stands: STOP.
   - The feasibility result does not say "the reversion exit is consistent with its target". It
     says the pre-declared rules did not support a build.

## What this does and does not license (decision for the researcher)

**What can't be done:** test reversion C4/C2/C1 now and call the result out-of-sample. Their
late-half replay gains have been seen, so choosing them now uses late information. A walk-forward
late verdict would be contaminated: it would no longer be a clean early-choose / late-judge test.

**Fix-oriented options:**
1. **Run them as labelled post-hoc exploratory arms**, on reversion only:
   - arms: C4 same depth, C2 k = 2.90, C1 p90;
   - report late P&L and the risk verdict, labelled EXPLORATORY;
   - adopt nothing until a distant era confirms.

   This is cheap: 3 arms, about 20 min.
2. **Write a follow-up feature with the right pre-declared condition.** For example, "share of stop
   exits that recover to the target within the max hold" instead of "killed winners". Gate it on
   a fresh evaluation window: a distant era, or later data as it arrives.
3. **Do both:** run option 1 for direction, and let option 2 decide adoption.

## Tasks

- T001–T007 done.
- T008–T021 not applicable (STOP), per tasks.md.
- T022–T024 done.

## Addendum 1: exploratory arms (option 1, 2026-10-12)

**EXPLORATORY, not out-of-sample.** These couplings were chosen after their late-half replay results
were seen. Nothing is adopted. The arms are implemented as subclasses of `st_book_rev2`, with no
framework change:
- `SimpleTrigger_RevOnlyVolreg2_XC4`: grace window extended via the existing sell parameter;
- `SimpleTrigger_RevOnlyVolreg2_XC1`: overrides `_resolve_normal_stop`;
- `SimpleTrigger_RevOnlyVolreg2_XC2`: overrides `_exit_params`.

Full output is in [exploratory.md](exploratory.md).

| reversion arm vs `st_book_rev2` (early +50.23 / late +4.20) | early / late pp | arm gain early / late | replay predicted | late paired | late Sharpe | late max DD |
|---|---|---|---|---|---|---|
| C4 grace 6.5 → 12 h | +58.67 / +10.46 | +8.44 / +6.27 | +19.05 / +12.76 | UNCHANGED, p 0.32 | 0.90 vs 0.31, UNCHANGED (p 0.18) | 7.46 vs 9.19, UNCHANGED |
| C2 TP = 2.90 × ATR | +54.70 / +8.37 | +4.46 / +4.17 | +8.21 / +8.31 | UNCHANGED, p 0.56 | 0.71, UNCHANGED (p 0.24) | 6.31, UNCHANGED |
| C1 stop ≥ −3.71% | +53.52 / +6.07 | +3.29 / +1.88 | +4.86 / +3.23 | UNCHANGED, p 0.50 | 0.46, UNCHANGED (p 0.46) | 9.05, UNCHANGED |

All three change trades in 15/15 folds.

**Reading:**
- **The direction holds.** All three are positive in BOTH halves, as the replay predicted.
- **The arms realise about half the replay gain:** 44–68%, versus an early/late replay of
  +19/+13 for C4. Holding a trade longer occupies the pair and changes later entries, which the
  replay ignores (023 Addendum 7). That is a useful calibration for future replays: discount
  about 2×.
- **C4 is the largest effect:** late +10.46 vs +4.20, the highest late Sharpe, and a lower
  drawdown.
- **Nothing is resolved,** and these arms are not out-of-sample. Adoption is the job of option 2:
  a follow-up feature whose pre-declared condition targets recovery within the max hold, judged
  on a fresh window.

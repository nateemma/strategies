# Contract — the exit-diagnosis arm classes

New strategy classes under `NNMT/` and `NNNC/`, following the existing oracle
pattern. **Every one is a LOOKAHEAD CEILING arm and none is promotable.**

## Common requirements

1. **Each carries its own `<Strategy>.json`.** Without one, an arm silently
   tracks whatever the shared class defaults become.
2. **Each overrides `get_model_path`** to alias the production weights. Omitting
   it makes a renamed strategy train inside its own backtest, fitting the
   evaluation window and returning different numbers with no error.
3. **No `Framework/` modification.** Every lever is a class attribute or a
   runtime parameter (research R4). The 12-hour unclog is the sole exception and
   `NNNC/_hold_window.py`'s HoldWindowMixin already handles it — import it.
4. **Entry side untouched.** These arms differ from their parent in the EXIT
   only. The entry guards are never opened.

## The levers, and how each is reached

| lever | how |
|---|---|
| ATR stop on/off | `use_atr_adaptive_stoploss = False` (class attribute) |
| ATR band | `atr_stoploss_multiplier` / `_floor` / `_cap` (class attributes) |
| take-profit | `--set-sell cexit_take_profit=` |
| max hold | `--set-sell cexit_max_days=` |
| exit signal + gate | `--set-sell enable_exit_signal=` / `exit_rvol_threshold=` |
| ROI table | `--roi` |
| 12-hour unclog | `HoldWindowMixin` from `NNNC/_hold_window.py` |

A variant whose only change is a runtime parameter needs NO new class — it runs
against the control's pinned `.json` with explicit overrides, which satisfies
requirement 1 because nothing falls back to shared defaults.

## `<Family>_Ora_Patient`

The joint bound. ATR stop disabled, ROI disabled, take-profit lifted clear,
unclog suppressed, held to the label horizon. **A BOUND, not a candidate** — it
is never promotable under any result, and its purpose is to decide whether the
remaining phases run at all.

## `<Family>_Ora_NoStop` and `<Family>_Ora_Band*`

One mechanism each, against the patient arm or the control as the decomposition
requires. Feature 004's `NNNC_Ora_StopWide` / `StopMid` / `StopTight` / `Comb`
already exist and were measured at label 0.007; they are RE-RUN at the current
label rather than rebuilt (research R1).

## Reporting

Every arm is judged on the acceptance window only; the others are reported and
labelled in-sample **with their net/gross ratio**, since a window can have a
respectable trade count and still be uninformative. Resolution is paired BY TRADE
where arms share entries and differ only in exits, and by set membership only
where the entry set actually changes. Any arm used as evidence must come from a
run confirmed successful, and any arm whose trades match its control is
inconclusive rather than refuted.

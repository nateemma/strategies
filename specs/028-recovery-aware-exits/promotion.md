# 028 US3 — C2 promoted into the shared exit

Only C2 was ADOPTED ([verdicts.md](verdicts.md)). C4 and C1 stopped at feasibility, so
`stop_min_depth` was not built (T017 N/A). C4 would only ever have been a parameter value.

## The switch

| piece | where | what |
|---|---|---|
| field | `utils/ExitPolicy.ExitParams.take_profit_atr_mult` | `None` = off (default). It must be > 0 when set. |
| clamp | `utils/ExitPolicy.atr_take_profit(k, atr_signal)` | `clamp(k × atr, ATR_TAKE_PROFIT_MIN 0.01, ATR_TAKE_PROFIT_MAX 0.05)`, the one copy. The 027 replay (`paths.c2_take_profit`) delegates to it. |
| strategy attribute | `Framework/BaseStrategy.take_profit_atr_mult` | `None` by default; copied into `ExitParams` by `_exit_params` |
| per-trade resolution | `BaseStrategy._apply_atr_take_profit(params, trade)` | Uses the signal bar, i.e. the candle before the entry candle, from the analysed dataframe. Caches the take-profit per trade. If the bar isn't visible, the preset take-profit stays. With no trade, no dataprovider, or a class standing in for an instance, it does nothing. Idempotent. |
| families | `utils/FamilyExits.FamilyExitMixin._exit_params` | Re-resolves after a family override, so a family can switch it on or off and its fixed `take_profit` can't displace a resolved ATR one. Setting it both strategy-wide and in a family set is refused at class creation (F3). |
| fingerprint | `evaluation/fingerprint.py` | `take_profit_atr_mult` is hashed only when set: it's in `OPTIONAL_ATTRIBUTE_MEMBERS`, and dropped from the resolved family sets while `None`. |

**Off is inert by construction.**
- `_exit_params` returns before calling anything new when the field is `None`.
- All 10 recorded fingerprints are unchanged; the test pins them.
- A control re-run is byte-identical (below).

## Declaring it

```python
# SimpleTrigger (per family; merges along the MRO):
FAMILY_EXIT_OVERRIDES = {"reversion": {"take_profit_atr_mult": 2.8970711921574543}}
# any BaseStrategy (all trades):
take_profit_atr_mult = 2.8970711921574543
```

The promoted arm is `SimpleTrigger/SimpleTrigger_RevOnlyVolreg2_P2.py`. Its fingerprint is
`2a1bb9f427343c56`, distinct from the control's. XC2 hashed like the control, because the
fingerprint can't see method overrides.

## Invariants

C2 changes only the take-profit, not the stop, so the stop invariants (023 FR-017) are untouched.
No stop-invariant violation appears in any T021 run log.

## Equivalence (T021, FR-011)

`SimpleTrigger_RevOnlyVolreg2_P2` (switch on through the shared exit) vs `SimpleTrigger_RevOnlyVolreg2_XC2` (027 method override):

| fold | XC2 trades_hash | P2 trades_hash | trades | |
|---|---|---|---|---|
| f00 | 657bfcf2edb6ef6a | 657bfcf2edb6ef6a | 235 | = |
| f01 | 173d59c478e9fd7b | 173d59c478e9fd7b | 148 | = |
| f02 | 009f70970e59e871 | 009f70970e59e871 | 107 | = |
| f03 | d51d47d2bedea17a | d51d47d2bedea17a | 295 | = |
| f04 | bea779ffcf2adac9 | bea779ffcf2adac9 | 412 | = |
| f05 | b9118e4a374916c3 | b9118e4a374916c3 | 204 | = |
| f06 | 694572c018a5c2ff | 694572c018a5c2ff | 144 | = |
| f07 | 151b507ded27a5a5 | 151b507ded27a5a5 | 512 | = |
| f08 | 59af3bcad75ce936 | 59af3bcad75ce936 | 255 | = |
| f09 | ae6666a04bbb96fd | ae6666a04bbb96fd | 131 | = |
| f10 | 95282e4c277af891 | 95282e4c277af891 | 166 | = |
| f11 | 42cee54507d4054c | 42cee54507d4054c | 292 | = |
| f12 | 82d08fa247714982 | 82d08fa247714982 | 145 | = |
| f13 | ccd9ef624b742469 | ccd9ef624b742469 | 140 | = |
| f14 | a50ca94dbe754a84 | a50ca94dbe754a84 | 161 | = |
| e00 | 004912580e903925 | 004912580e903925 | 432 | = |
| e01 | 052d4242a7335a7b | 052d4242a7335a7b | 218 | = |
| e02 | a6832e2c2cdf2762 | a6832e2c2cdf2762 | 269 | = |

**18/18 folds identical.**

## Switch off (T021, FR-010)

- `st_rev2_recheck` f00 (control class re-run after the change, fresh tag): `26869046bc547532` vs recorded `st_book_rev2` f00 `26869046bc547532` -> identical
- fingerprints after the change: control `0e552be31732d222`, momentum `95dd915b90fc6217`, arm D `303456c1fa322b4c`, P2 `2a1bb9f427343c56`
  (recorded: 0e552be31732d222, 95dd915b90fc6217, 303456c1fa322b4c; all 10 recorded shas pinned by `evaluation/tests/test_fingerprint_switches.py`)
- Stop-invariant violations in the 19 T021 run logs: **0**.

# Contract C1: generated candidate classes

## `NNNC_LS_<TAG>` (entry-source role; also the training class)

```python
class NNNC_LS_<TAG>(LabelParamsMixin, NNNC_A_Raw_Open):
    TRAINING_TYPE = <method id>
    LABEL_PARAMS = <declared extras>          # from research/label_screen/candidates.py
    LABEL_PARAMS_SELL = None                  # mirror; TB declares {"pt": 0.05, "sl": 0.02}
    prediction_threshold = DecimalParameter(0.0, 1.0, default=0.5, decimals=4, space="buy",
                                            load=True, optimize=False)
```

- `.json` copied from the control's base `.json`, `strategy_name` replaced.
- Entries: the control's own path (`ENTRY_MODE = "model"`, no trigger gate, the control's guards).
- Model key: differs from the control's and from every other candidate's (TRAINING_TYPE and
  LABEL_PARAMS are key attributes).

## `NNNC_LS_<TAG>_SZ` (sizer role)

```python
class NNNC_LS_<TAG>_SZ(WidthSizerMixin, WidthBookMixin, NNNC_LS_<TAG>):
    WIDTH_CUTS = <NNNC_W_ARO.WIDTH_CUTS>
```

- Entries exactly the width book's; stake 1.333 / 0.667 by the `:agree` / `:disagree` tag.
- **Same model key as `NNNC_LS_<TAG>` for every fold** (tested), so weights train once.

## GFM reuse

`NNNC_LS_GFM_SZ(WidthSizerMixin, WidthBookMixin, NNNC_GbbFwdMae_Open)` and an entry-source class on
`NNNC_GbbFwdMae_Open` with the 4-decimal threshold; model key equal to GFM's existing fold keys.

## Invariants (tests)

1. Control classes' model keys and fingerprints unchanged.
2. Every declared LABEL_PARAMS key is accepted by its labeler (strict keys raise otherwise).
3. Sizer twin key == entry-source key, per fold.
4. Feature column list of each candidate == the control's.
5. Thresholds load at 4 decimals.

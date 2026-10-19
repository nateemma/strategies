# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""039 operating point (research/nnpredict/preregistration_op.json): the NNNC reference operating
point applied to NNPredict, nothing tuned.

- guards OPEN, with NNNC_A_Raw_Open's own values (its .json): the guard chain rejects ~97% of bars;
- the magnitude floor in NNPredict's existing scale-invariant mode ("percentile", its default
  magnitude_pctile 0.60): the absolute floor 0.10 sits ~5 sigma out on the compressed Ridge
  predictions (f00: 54 of 78,345 bars signalled, 3 passed the guards, 2 trades).

Both act after the model (MODEL_KEY_IGNORED), so a class with this mixin reuses its base model's
trained weights exactly.
"""

OPEN_GUARDS = {
    "entry_adx_threshold": -1.0,
    "entry_atr_pct": -1.0,
    "entry_bb_width_threshold": -1.0,
    "entry_close_norm_threshold": 5.0,
    "entry_enable_guards": True,  # keeps the volume > 0 check live (BaseStrategy)
    "entry_guard_threshold": 5.0,
    "entry_rvol_threshold": -1.0,
}


class OperatingPointMixin:
    magnitude_floor_mode = "percentile"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        base = next((k.buy_params for k in cls.__mro__[1:] if "buy_params" in k.__dict__), {})
        cls.buy_params = {**base, **OPEN_GUARDS}

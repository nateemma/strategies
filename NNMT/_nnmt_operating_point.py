# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 operating point (research R4; research/nnmt/preregistration_models.json): ARO's operating point
applied to NNMT, nothing tuned -- NNNC_A_Raw_Open.json's buy params (guards open, prediction_threshold
0.80). NNMT's exits (its exit signal, bail-out and grace) are unchanged.

All of these act after the model (MODEL_KEY_IGNORED), so a class with this mixin reuses its base
model's trained weights exactly.
"""

ARO_OPERATING_POINT = {
    "entry_adx_threshold": -1.0,
    "entry_atr_pct": -1.0,
    "entry_bb_width_threshold": -1.0,
    "entry_close_norm_threshold": 5.0,
    "entry_enable_guards": True,  # keeps the volume > 0 check live (BaseStrategy)
    "entry_guard_threshold": 5.0,
    "entry_rvol_threshold": -1.0,
    "prediction_threshold": 0.8,
}


class OperatingPointMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        base = next((k.buy_params for k in cls.__mro__[1:] if "buy_params" in k.__dict__), {})
        cls.buy_params = {**base, **ARO_OPERATING_POINT}

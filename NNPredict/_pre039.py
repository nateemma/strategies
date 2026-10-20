# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNPredictStrategy's post-model settings BEFORE the 039 adoption (2026-10-21): the model-only entry,
the absolute magnitude floor and the original entry guards. The as-shipped 039 records (the models
stage: nnp_ridge, nnp_ridge_sa, nnp_lstm_raw, nnp_lstm_ind, nnp_lstm_ret) ran these; their
<Class>_Pre039 twins pin them so the records reproduce. Post-model only: the weights are reused.
"""

PRE039_GUARDS = {
    "entry_adx_threshold": 0.02,
    "entry_atr_pct": 0.001,
    "entry_bb_width_threshold": 0.0,
    "entry_close_norm_threshold": 0.0,
    "entry_enable_guards": True,
    "entry_guard_threshold": -0.2,
    "entry_rvol_threshold": 1.0,
}


class Pre039DefaultsMixin:
    ENTRY_MODE = "model"
    TRIGGER_NAME = None
    magnitude_floor_mode = "absolute"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        base = next((k.buy_params for k in cls.__mro__[1:] if "buy_params" in k.__dict__), {})
        cls.buy_params = {**base, **PRE039_GUARDS}

# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNMTStrategy's post-model defaults BEFORE the 040 adoptions (2026-10-24), pinned on the 040 record classes
so their records reproduce (their plans point at the twins). Post-model only: the weights are reused.

- stop-loss grace 2.6 h at -0.17 (inherited from BaseStrategy.sell_params) -- adopted 24 h;
- no regime-transition exposure scaler (tr_scale), and model entries untagged (MODEL_ENTRY_FAMILY None)
  -- adopted: tr_scale on, entries tagged reversion:model (the scaler reads the tagged mask).

  PreG24GraceMixin  both (every 040 record except the g24 arm and the width-book / sizer arms):
                    <Class>_PreG24
  PreTrMixin        tr_scale off and entries untagged only: <Class>_PreTR for the g24 arm (it ran 24 h)
                    and NNMT_LR_RI's mt_lr_ri_def
  PreTrScaleOffMixin tr_scale off, family kept: <Class>_PreTR for the width-book / sizer arms, whose own
                    MODEL_ENTRY_FAMILY "reversion" the sizer needs
"""

PRE_G24_GRACE_HOURS = 2.6


class PreTrScaleOffMixin:
    TR_UP_CELL = None
    TR_DOWN_CELL = None
    TR_SOURCE = "tables"


class PreTrMixin(PreTrScaleOffMixin):
    MODEL_ENTRY_FAMILY = None


class PreG24GraceMixin(PreTrMixin):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        base = next((k.sell_params for k in cls.__mro__[1:] if "sell_params" in k.__dict__), {})
        cls.sell_params = {**base, "stoploss_grace_hours": PRE_G24_GRACE_HOURS}

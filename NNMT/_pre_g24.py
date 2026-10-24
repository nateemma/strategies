# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""NNMTStrategy's stop-loss grace BEFORE the 040 adoption (2026-10-24): 2.6 h at -0.17, inherited from
BaseStrategy.sell_params. Every 040 record ran it except the g24 arm and the width-book / sizer arms
(w_aro's framework, 24 h). Their <Class>_PreG24 twins pin it so the records reproduce; their plans
point at the twins. Post-model only (the grace is model-key IGNORED): the weights are reused.
"""

PRE_G24_GRACE_HOURS = 2.6


class PreG24GraceMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        base = next((k.sell_params for k in cls.__mro__[1:] if "sell_params" in k.__dict__), {})
        cls.sell_params = {**base, "stoploss_grace_hours": PRE_G24_GRACE_HOURS}

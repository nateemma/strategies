# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 input sets for the NNMT model variants (research R6), on 039's causal raw8 builder
(NNPredict/_inputs.py: every added column is a trailing z-score clipped to +-5 / 5, registered in
pre_normalized_columns, so no fitted scaler ever sees it -- constitution XI).

  "raw"      raw8 only (r8_ret1, r8_ret4, r8_log_hc, r8_log_lc, r8_log_oc, r8_range, r8_logvol_z,
             r8_vol_ratio). NNPredict's "raw" also carries its current_gain / recent_gain, which NNMT
             does not compute, so the set is narrowed here.
  "raw_ind"  NNMT's 24 indicators + raw8
"""

from NNPredict._inputs import R8, InputSetMixin


class NNMTInputSetMixin(InputSetMixin):
    INPUT_SET = "raw"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.INPUT_SET not in ("raw", "raw_ind"):
            raise ValueError(f"{cls.__name__}: NNMT INPUT_SET must be 'raw' or 'raw_ind'")

    def model_inputs(self) -> list[str]:
        if self.INPUT_SET == "raw":
            return list(R8)
        return list(self.include_list)  # raw_ind: the 24 indicators + raw8 (added by the mixin)

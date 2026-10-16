"""both32: ARO's 24 indicators plus GBR's 8 raw-price features (035 follow-up, research/new_inputs/
preregistration_b32.json).

The 018 classifier study ran `both32` on forward_mae only (MCC); on the production gbb_path label
it was never trained. A class with this mixin lists `include_list` = its base's + RAW8_COLUMNS and
reads `main_scaler_both32`, a composite of the two existing fitted RobustScalers (each column keeps
the centre/scale its own control model uses; research/new_inputs/both32_scaler.py), so the only
change from ARO is the 8 added columns.
"""

from pandas import DataFrame

from _raw8_features import add_raw8_features


class Both32Mixin:
    main_scaler_name = "main_scaler_both32"

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        return add_raw8_features(super().add_additional_indicators(dataframe))

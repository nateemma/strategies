import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from pandas import DataFrame

from _ablation_config import SkipColumnCheck
from _raw8_features import add_raw8_features, RAW8_COLUMNS
from NNNC_A_Raw_Open import NNNC_A_Raw_Open


class NNNC_GbbRaw_Open(SkipColumnCheck, NNNC_A_Raw_Open):
    """DECOMPOSITION arm: gbb labels on RAW OHLCV ONLY, guards open, seq_len 16.

    Splits the forward_mae arm's loss into its label component and its feature
    component. That arm differed from the control in three ways at once -- label,
    feature set and window -- and returned -0.22pp over 3.68 years against the
    control's +39.98pp. The exit has already been eliminated (swapping it back to
    the incumbent moved the result 0.6pp). This arm changes ONLY the feature set
    relative to the control: same gbb label, same seq_len 16, same LSTM_RAW, same
    guards-open gate, same exit.

      lose  -> the raw feature set is the culprit, and the classifier study's
               headline recommendation (drop the 24 indicators, which beat ind24
               on 8 of 8 architectures) is refuted for trading.
      hold  -> raw features are fine and forward_mae itself is what does not trade.

    NOTE on the prior: the gbb label IS a two-threshold rule on `guard_metric` and
    `bb_width`, both of which are in ind24 and NEITHER of which is in raw8. So this
    arm must reconstruct its own label's defining inputs from price rather than being
    handed them. Expect a large drop in learnability; the question is what it does to
    realised return.

    Scaler: deliberately reuses `main_scaler_fwdmae_raw8`. A scaler is fitted per
    FEATURE, so an identical feature set gives an identical scaler regardless of
    label -- and sharing it guarantees both raw8 arms are scaled identically, which
    is what makes them comparable.
    """

    include_list = list(RAW8_COLUMNS)
    main_scaler_name = "main_scaler_fwdmae_raw8"

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        """raw8 ON TOP OF the default regime/risk/flow/momentum set -- `get_predictions`
        hard-requires `regime` and `flow`. See NNNC_ForwardMAE_Raw for the full note."""
        dataframe = super().add_additional_indicators(dataframe)
        return add_raw8_features(dataframe)

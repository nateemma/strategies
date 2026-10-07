# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from pandas import DataFrame

from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02
from NNNClassifierMLX import ClassifierTypeMLX
from Framework.TrainingSignals import LabelMethod
from _raw8_features import add_raw8_features, RAW8_COLUMNS


class NNNC_ForwardMAE_Raw(NNNC_MLX_Lbl02):
    """forward_mae label on the raw8 feature set (research/clfstudy raw8),
    seq_len 32, horizon 12 (3h at 15m).

    A deliberately different arm from its NNNC_MLX_Lbl02 parent in three
    independent ways: the LABEL (forward_mae MFE/MAE-cap instead of
    gbb_path), the FEATURES (8 causal raw-OHLCV columns instead of the
    ind24 indicator set), and the architecture's SEQ_LEN (32 instead of 16).
    Entry/exit guards, the prediction threshold and the base gate are left
    untouched -- inherited from NNNC_MLX_Lbl02 (guards open, threshold 0.80)
    by design, so this arm isolates the label+feature+seq_len change.

    LABEL_MAX_DRAWDOWN = 0.10 (not the labeler's own 0.02 default) --
    the study's winning forward_mae configuration. See the plumbing added to
    BaseNNStrategy.get_train_buy_signals/get_train_sell_signals: the
    params dict omits "max_drawdown" unless LABEL_MAX_DRAWDOWN is set, so
    every other strategy is unaffected by this arm's choice.

    MUST BE TRAINED BEFORE EVALUATION -- DO NOT BACKTEST UNTRAINED.
    Unlike NNNC_MLX_Open (which reuses NNNC_MLX_Lbl02's weights via a
    get_model_path override because it changes only pinned entry/exit
    parameters), this arm trains a genuinely different model: different
    label, different features, different seq_len. get_model_path is
    deliberately NOT overridden here, so it resolves via
    BaseNNStrategy.get_model_path's default (keyed off
    ``self.__class__.__name__``) to saved_data/NNNC_ForwardMAE_Raw/, which
    starts EMPTY. Per the documented hazard in AGENT_GUIDE.md, running a
    backtest against this strategy before it has been deliberately trained
    (via the walk-forward harness's train step) will silently train a fresh
    model INSIDE the backtest, fitting the evaluation window -- verify
    ``saved_data/NNNC_ForwardMAE_Raw/`` is non-empty before trusting any
    backtest number from this class.
    """

    classifier_type = ClassifierTypeMLX.LSTM_RAW
    seq_len = 32

    TRAINING_TYPE = LabelMethod.forward_mae
    HORIZON = 12
    MIN_BUY_GAIN_THRESHOLD = 0.015
    MIN_SELL_LOSS_THRESHOLD = 0.015
    LABEL_MAX_DRAWDOWN = 0.10

    # The 8 raw8 columns ONLY -- the ind24 indicator set computed upstream
    # by DataframePopulator.add_indicators(MINIMAL) is still present in the
    # dataframe (guards read it directly), but rolling_dataframe_normalise /
    # clean_for_tensor drop everything not in include_list, so the model
    # never sees it.
    include_list = list(RAW8_COLUMNS)

    # Own scaler -- isolated from the shared production main_scaler so the
    # live MomentumRegimeBasket15mFast dry run (and every other NN strategy
    # reading main_scaler) is never touched.
    main_scaler_name = "main_scaler_fwdmae_raw8"

    def add_additional_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Add the raw8 features instead of BaseNNStrategy's default
        regime/risk/flow/momentum set.

        Those four defaults read atr_norm / di_diff_scaled / aroonosc_scaled
        via check_columns_included, none of which are in this arm's
        raw8-only include_list, so calling super() here would raise.
        Skipping them is safe: entry/exit guards (rvol, guard_metric,
        close_norm) read columns from DataframePopulator's MINIMAL set,
        added to the dataframe *before* this hook runs, and are therefore
        unaffected by include_list or by this override.
        """
        return add_raw8_features(dataframe)

# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 regime-state amendment (research/nnmt/preregistration_regime_state.json): retarget NNMT's regime head
to the FUTURE causal state Y = S_H(t+H) (BEAR 0 / SIDEWAYS 1 / BULL 2, MarketRegime's coding).

S_H(t) is the transition study's state (research/nnmt/transition.py): z_H = log(close / SMA_H) / (sigma *
sqrt(H)), sigma = trailing std of 1-bar log returns over SIGMA_WINDOW bars; BULL if z >= +DEADBAND, BEAR if
z <= -DEADBAND, else SIDEWAYS. A row whose state or future state is unknown -- the warm-up, and the last H
bars of the training frame, whose future lies after the training range -- carries NO regime loss: its label
is UNKNOWN and one-hot encodes to an all-zero row (zero focal loss, skipped by the validation metrics).

The cell (H, level) and the monitor key are weight-shaping, so this mixin declares them in the model key;
only classes that include it get new keys.
"""

import numpy as np
import pandas as pd

from utils.DataframeUtils import DataframeUtils

UNKNOWN = -1


class _MaskedOneHot(DataframeUtils):
    """one_hot_encode with UNKNOWN (-1) -> an all-zero row (instead of clipping it to class 0)."""

    def one_hot_encode(self, labels, num_classes):
        labels = np.asarray(labels.values if isinstance(labels, pd.Series) else labels).astype(int)
        unknown = labels == UNKNOWN
        out = super().one_hot_encode(np.where(unknown, 0, labels), num_classes)
        out[unknown] = 0.0
        return out


def future_state(close: pd.Series, H: int, deadband: float, sigma_window: int, sigma_min: int) -> np.ndarray:
    """MarketRegime-coded S_H(t+H) (0/1/2), UNKNOWN where S_H(t) or S_H(t+H) is unknown."""
    c = pd.Series(np.asarray(close, float))
    lc = np.log(c)
    sma = c.rolling(H, min_periods=H).mean()
    sig = lc.diff().rolling(sigma_window, min_periods=sigma_min).std()
    z = (lc - np.log(sma)) / (sig * np.sqrt(H))
    s = pd.Series(np.where(z >= deadband, 2.0, np.where(z <= -deadband, 0.0, 1.0))).where(z.notna())
    fut = s.shift(-H)
    return np.where(s.notna() & fut.notna(), fut, UNKNOWN).astype(int)


class RegimeStateMixin:
    REGIME_STATE_H = 288
    REGIME_STATE_LEVEL = "pair"
    REGIME_STATE_DEADBAND = 0.5
    REGIME_STATE_SIGMA = (2880, 960)
    _CLASSIFIER_MONITOR_KEY = None

    MODEL_KEY_ATTRIBUTES = ("REGIME_STATE_H", "REGIME_STATE_LEVEL", "REGIME_STATE_DEADBAND", "REGIME_STATE_SIGMA",
                            "_CLASSIFIER_MONITOR_KEY")

    def get_market_target(self, dataframe: pd.DataFrame) -> np.ndarray:
        if self.REGIME_STATE_LEVEL != "pair":
            raise NotImplementedError("regime-state: only the pair level is wired (BTC needs its close in the frame)")
        if "close" not in dataframe.columns:
            raise ValueError("regime-state: the training frame has no close column")
        if self.dataframeUtils is not None and not isinstance(self.dataframeUtils, _MaskedOneHot):
            self.dataframeUtils.__class__ = _MaskedOneHot  # same state, masked one-hot
        w, m = self.REGIME_STATE_SIGMA
        return future_state(dataframe["close"], int(self.REGIME_STATE_H), float(self.REGIME_STATE_DEADBAND), int(w), int(m))

    def _apply_classifier_overrides(self, clf) -> None:
        super()._apply_classifier_overrides(clf)
        if clf is not None and self._CLASSIFIER_MONITOR_KEY:
            clf.monitor_key = self._CLASSIFIER_MONITOR_KEY

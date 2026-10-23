# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 T045 (preregistration_transition.json "head_version"): retarget NNMT's regime head to the R11 transition
label for one (H, level) cell -- DOWN 0 / NONE 1 / UP 2 (MarketRegime's slots: BEAR / SIDEWAYS / BULL).

UP if S_H(t) != BULL and S_H(t+H) == BULL; DOWN if S_H(t) != BEAR and S_H(t+H) == BEAR; else NONE. S_H is the
transition study's state (research/nnmt/transition.py). A row whose state or future state is unknown -- the
warm-up and the last H bars of the training frame -- is UNKNOWN: an all-zero one-hot row, no regime loss.
The cell and the target are weight-shaping (model key); the rest is RegimeStateMixin's.
"""

import numpy as np
import pandas as pd

from _regime_state import UNKNOWN, RegimeStateMixin, future_state


def transition_target(close: pd.Series, H: int, deadband: float, sigma_window: int, sigma_min: int) -> np.ndarray:
    """0 DOWN / 1 NONE / 2 UP for the H-bar transition; UNKNOWN where either end's state is unknown."""
    fut = future_state(close, H, deadband, sigma_window, sigma_min)  # S_H(t+H), UNKNOWN unless both ends known
    c = pd.Series(np.asarray(close, float))
    lc = np.log(c)
    z = (lc - np.log(c.rolling(H, min_periods=H).mean())) / (lc.diff().rolling(sigma_window, min_periods=sigma_min).std() * np.sqrt(H))
    cur = np.where(z >= deadband, 2, np.where(z <= -deadband, 0, 1))
    known = fut != UNKNOWN
    up = known & (cur != 2) & (fut == 2)
    down = known & (cur != 0) & (fut == 0)
    return np.where(~known, UNKNOWN, np.where(up, 2, np.where(down, 0, 1))).astype(int)


class TransitionHeadMixin(RegimeStateMixin):
    REGIME_TARGET = "transition"
    MODEL_KEY_ATTRIBUTES = ("REGIME_TARGET",)  # merged along the MRO (Framework/ModelKey.py)

    def get_market_target(self, dataframe: pd.DataFrame) -> np.ndarray:
        super().get_market_target(dataframe)  # validates the frame, installs the masked one-hot
        w, m = self.REGIME_STATE_SIGMA
        return transition_target(dataframe["close"], int(self.REGIME_STATE_H), float(self.REGIME_STATE_DEADBAND),
                                 int(w), int(m))

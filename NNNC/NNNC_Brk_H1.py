import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from _hold_window import HoldWindowMixin
from NNNC_BreakoutOpen import NNNC_BreakoutOpen


class NNNC_Brk_H1(HoldWindowMixin, NNNC_BreakoutOpen):
    """Feature 003 arm H1 -- the HOLD WINDOW ALONE (US2, T023, FR-011).

    Carries the CONTROL's label (h48 / 0.005, unchanged) and changes only the
    forced-exit window, 12h -> 24h. That is what makes the decomposition in
    T028 valid: H1 and L2 share nothing but their exit machinery, so

        hold window alone            = H1 - control
        tail label given that window = L2 - H1

    24h is the H=96 label's own horizon at the 15m timeframe (96 bars = 24h),
    not a swept value. The tail candidate's payoff peaks at a median 17.0h,
    inside that window and outside the 12h one.

    NO RETRAIN: this arm reuses the host's weights. Only the exit rule differs,
    so `get_model_path` is inherited and points at saved_data/NNNC_BreakoutOpen.

    Its `exit_config_hash` is EXPECTED to differ from the control -- the hold
    window is the parameter under test, the one permitted exception to FR-016.
    """

    HOLD_WINDOW_HOURS = 24.0

    def get_model_path(self) -> str:
        """Reuse the HOST's weights -- H1 changes only the exit rule.

        The base implementation keys the path off ``self.__class__.__name__``,
        which would send this arm looking for a model that was never trained.
        H1 carries the control's label unchanged, so retraining it would only
        add seed noise to a comparison whose whole purpose is to isolate the
        exit-side change. Pointing at NNNC_BreakoutOpen guarantees the model is
        BYTE-IDENTICAL to the control's and the only difference is the hold
        window.
        """
        root_dir = self.get_storage_location()
        name = "NNNC_BreakoutOpen"
        return root_dir + name + "/" + name + ".keras"

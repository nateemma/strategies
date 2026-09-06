import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_MLX_Lbl02 import NNNC_MLX_Lbl02


class NNNC_MLX_Open(NNNC_MLX_Lbl02):
    """Model-driven NNNC: label 0.02, quality guards OPEN, prediction 0.80.

    The one configuration found in the 2026-09-05/06 study whose edge comes from
    the MODEL rather than the guard chain. Everything it needs is pinned in
    NNNC_MLX_Open.json so it runs standalone -- the study reached it through
    run_arm --set overrides, which is not a deployable form.

    Measured, guards open, threshold 0.80:

        window   trades    ret%     DD%    PF   win%   Calmar   Sortino
        W1          431   +6.94    7.90  1.16   65.4     4.61      0.87
        W2          763  +16.91    8.54  1.31   65.0    10.39      3.74
        W3          295  +10.25    2.34  1.49   68.1    22.98      3.11
        summed          +34.10                              (1489 trades)

    against production NNNC_MLX (label 0.007, guards on, 0.69):
        +10.45 / +10.31 / +3.37 = +24.13 over 332 trades, worst DD 5.25%.

    WHY THIS RATHER THAN PRODUCTION. Production wins W1, the acceptance window,
    by 3.51pp; this arm wins summed by 9.97pp and wins W2 and W3. Worst-window
    drawdown is 8.54% against 5.25%, and return per worst-window drawdown is
    3.99 against 4.60 -- close, with better Sortino on two windows of three.

    The deciding argument is headroom. At the production gate a PERFECT model is
    worth 0.68pp more than the real one, so no model work there can ever pay. At
    this operating point the volume-matched gap is ~25pp and the model captures
    47% of it. It is the only configuration in this repo where improving the
    model can move the result, and at 431 W1 trades a 1pp effect is measurable
    where at production's 95 it is not.

    NOT YET LIVE-VALIDATED. See regime/CANDIDATE_STATUS.md for what remains.
    """

    def get_model_path(self) -> str:
        """Reuse NNNC_MLX_Lbl02's weights.

        `BaseNNStrategy.get_model_path` keys the path off
        ``self.__class__.__name__``, so without this override a renamed
        configuration looks for weights that do not exist and TRAINS ITS OWN
        INSIDE THE BACKTEST -- which both breaks the comparison (290 trades
        against the study arm's 431) and fits on the evaluation window.

        This class differs from NNNC_MLX_Lbl02 only in pinned parameters, so it
        must load the identical model for its numbers to mean anything.
        """
        return self.get_storage_location() + "NNNC_MLX_Lbl02/" + "NNNC_MLX_Lbl02.keras"

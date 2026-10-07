import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNNC_A_Raw import NNNC_A_Raw


class NNNC_A_Raw_Open(NNNC_A_Raw):
    """CONTROL for the forward_mae package test: LSTM_RAW on gbb labels, the
    24-indicator set and seq_len 16 -- the incumbent configuration -- at the
    GUARDS-OPEN, threshold-0.80 operating point.

    Exists only because `NNNC_A_Raw.json` pins guards ON (entry_adx_threshold
    38.0). The gate must be held constant across arm and control: the guard
    chain rejects ~97% of bars, so comparing a guards-open arm against a
    guards-on control would measure the gate, not the label. Its .json is
    NNNC_MLX_Open.json's params verbatim, which is the operating point with
    enough headroom and trade count for a model-side change to be measurable.

    No code differences from NNNC_A_Raw -- the parameters are the whole point.
    It gets its own class so it gets its own .json and its own saved_data/.
    """

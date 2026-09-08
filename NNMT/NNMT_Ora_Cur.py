# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""
NNMT_Ora_Cur — the CURRENT exit, at a raised slot limit. A LOOKAHEAD CEILING.

Exists only to de-confound the patient bound. NNMT_Ora_Patient holds ~5x longer
(median 52 bars against 11) and SATURATES max_open_trades at 10/10, so entries
were being rejected for want of a slot: its lower trade count mixed the exit
change with a 26% smaller entry set.

This arm is NNMT_Oracle_Lbl02 unchanged except for the same raised slot limit
pinned in its .json, so the patient comparison varies the EXIT alone.

Not promotable; nothing measured with an oracle is.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from NNMT_Oracle_Lbl02 import NNMT_Oracle_Lbl02


class NNMT_Ora_Cur(NNMT_Oracle_Lbl02):
    pass

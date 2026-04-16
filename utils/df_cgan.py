# pragma: no cover
# Backward-compatibility shim. The implementation has moved to GANs/.
# This file exists so that existing `from utils.X import Y` statements
# continue to work without modification.
from GANs.df_cgan import *  # noqa: F401, F403

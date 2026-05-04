# pragma: no cover
# Backward-compatibility shim. The implementation lives in GANs/.
# This file exists so that any `from utils.X import Y` statement
# continues to work without modification.
from GANs.df_mt_ctab_gan_mlx import *  # noqa: F401, F403

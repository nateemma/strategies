# type: ignore
# NNMT GAN-quality study: AE filter OFF (retrains). Mirrors the NNNC AE-off test.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from NNMT_DDPM_MLX import NNMT_DDPM_MLX
class NNMT_DDPM_MLX_aeoff(NNMT_DDPM_MLX):
    gan_synth_autoencoder_threshold = None   # AE filter OFF

"""
NNNC_DDPM_MLX_LSTM — NNNC MLX-LSTM classifier with TabDDPM augmentation.

Same architecture as NNNC_CGP_MLX_LSTM (MLX LSTM backbone, NNNC family);
swaps the augmentation backend from CTAB-GAN+ to TabDDPM by setting
``gan_type = GANType.TAB_DDPM``.

Train the GAN first with CreateTabDDPM, then train this strategy — the
saved TabDDPM model will be loaded from
``saved_data/NNNC_DDPM_MLX_LSTM/GANs/tab_ddpm/`` 
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_CGP_MLX_LSTM import NNNC_CGP_MLX_LSTM
from Framework.BaseStrategy import GANType


class NNNC_DDPM_MLX_LSTM(NNNC_CGP_MLX_LSTM):
    gan_type = GANType.TAB_DDPM

    gan_target_ratio = 0.5

    # v2 pipeline: TAB_DDPM is naturally v2 (diffusion + internal z-score
    # + linear output). Strategy reads from saved_data/GANs_PostScale/tab_ddpm/.
    # CreateTabDDPM must have use_post_gan_scaling = True too.
    use_post_gan_scaling = True

    # Tier-1 GAN sampling overrides — applied post-load, no retrain needed.
    # More denoising steps (200 vs saved 50) + mild classifier-free
    # guidance to nudge samples toward class modes without collapsing the
    # tails. guidance=2.0 was tried and collapsed σ_ratio to ~0.52, so
    # dialled back to 1.3. Both knobs honoured via
    # BaseNNStrategy._apply_gan_inference_overrides at GAN load time.
    gan_inference_sample_steps = 200
    gan_inference_guidance_scale = 1.3

    # Step 1 of the GAN improvement plan (density-based rejection) was
    # tested at reject_pct ∈ {0.10, 0.20} and n_components ∈ {4, 8}.
    # Every configuration compressed σ_ratio without compensating wins on
    # joint correlations — GMMs underweight tails in 25-D space, so
    # real-distribution tail samples got flagged as low-density. Filter
    # disabled; falling through to the Tier-1 sampling overrides.
    gan_synth_density_reject_pct = 0.0

    buy_params = { **NNNC_CGP_MLX_LSTM.buy_params,
        "prediction_threshold": 0.65
        }

"""
NNNC_DDPM_MLX — NNNC MLX-LSTM classifier with TabDDPM augmentation.

MLX LSTM backbone, NNNC family

Train the GAN first with CreateTabDDPM, then train this strategy — the
saved TabDDPM model will be loaded from
``saved_data/NNNC_DDPM_MLX/GANs/tab_ddpm/`` 
"""

import sys
from pathlib import Path

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNNC_MLX import NNNC_MLX
from Framework.BaseStrategy import GANType


class NNNC_DDPM_MLX(NNNC_MLX):


    buy_params = { **NNNC_MLX.buy_params,
        "prediction_threshold": 0.6
        }

    # Trend filter on NNNC_MLX was an experiment specific to the H=24
    # gbb labeler path; production DDPM uses its own augmentation and
    # doesn't need (and shouldn't get) the upstream filter.
    entry_trend_filter_enable = False

    gan_type = GANType.TAB_DDPM

    gan_target_ratio = 0.5

    # Extend the base calendar passthrough with the features TabDDPM
    # consistently compresses on the clean-data retrain. The marginal
    # diagnostic at g=1.0 shows synth means pulled toward 0 on these
    # heavy-tailed / bimodal volatility & momentum features; copying
    # real values for class-matched source rows sidesteps the
    # MLP-regression-to-mean failure mode without disturbing the
    # other features the GAN does fit well.
    gan_passthrough_columns = [
        # Heavy-tailed volatility features the MLP-regression GAN
        # compresses on the conditional mean. The calendar features
        # that used to live here have been removed from include_list
        # entirely; vwap_ratio and macd_norm were split into pos/neg
        # unimodal pairs in include_list, which sidesteps the failure
        # mode at the source.
        "atr_norm", "spread_ma",
    ]

    # v2 pipeline: TAB_DDPM is naturally v2 (diffusion + internal z-score
    # + linear output). Strategy reads from saved_data/GANs_PostScale/tab_ddpm/.
    # CreateTabDDPM must have use_post_gan_scaling = True too.
    use_post_gan_scaling = True

    # Tier-1 GAN sampling overrides — applied post-load, no retrain needed.
    # More denoising steps (200 vs saved 50) + classifier-free guidance.
    # guidance=1.8 on the clean-data retrain collapsed bb_width σ_ratio to
    # 0.50 (worse than g=1.3's 0.77); CFG amplifies whatever class-
    # conditional bias the model has, so on features with a biased learned
    # mode (bb_width pinned high for non-hold classes) it commits harder
    # to the bias. Backed off to 1.0 = pure conditional sampling, no
    # amplification. Both knobs honoured via
    # BaseNNStrategy._apply_gan_inference_overrides at GAN load time.
    gan_inference_sample_steps = 200
    gan_inference_guidance_scale = 1.0

    # Step 1 of the GAN improvement plan (density-based rejection) was
    # tested at reject_pct ∈ {0.10, 0.20} and n_components ∈ {4, 8}.
    # Every configuration compressed σ_ratio without compensating wins on
    # joint correlations — GMMs underweight tails in 25-D space, so
    # real-distribution tail samples got flagged as low-density. Filter
    # disabled; falling through to the Tier-1 sampling overrides.
    gan_synth_density_reject_pct = 0.0

    # gan_synth_realsignal_reject_pct = 0.3
    # gan_synth_realsignal_threshold = 0.5
    gan_synth_autoencoder_threshold = 0.005

    gan_run_diagnostics = True

    augment_training_data = True




# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_DDPM — multi-task NNMT strategy with MT-DDPM diffusion augmentation.

Sibling of NNMT_WGAN; the only difference is that ``gan_type`` is set to
``GANType.MT_DDPM`` (both the class attribute and the ``GANInterface``
constructor call in ``preprocess_training_data``).  All balancing logic
is shared via ``GANs.balance.balance_multi_task``, which is GAN-type
agnostic.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy  # noqa: E402
from GANs.GANType import GANType  # noqa: E402
from GANs.GANInterface import GANInterface, GANMetadataMismatchError  # noqa: E402
from GANs.paths import gan_save_path  # noqa: E402


class NNMT_DDPM(NNMTStrategy):

    # GAN augmentation — multi-task DDPM diffusion, applied to 3-D sequential
    # tensors in preprocess_training_data.  We turn off the BaseNN 2-D
    # dispatcher (gan_augment=False) because the augmentation is done
    # later in 3-D space.
    gan_type = GANType.MT_DDPM
    gan_augment = False

    # v2 pipeline: GAN consumes raw features and a tensor-level scaler runs
    # AFTER augmentation. Strategy looks for the model under
    # saved_data/GANs_PostScale/mt_ddpm/ rather than saved_data/GANs/mt_ddpm/.
    # Must match the same flag on CreateMTDDPM (training side).
    use_post_gan_scaling = True

    # Classifier-side overrides flowed through _apply_classifier_overrides.
    # Mirror NNMT_WGAN so DDPM and WGAN comparisons share configuration.
    # Set _CLASSIFIER_MAX_EPOCHS on the subclass to raise the training
    # ceiling (early-stopping still cuts off at plateau).
    # Read by BaseNNMTStrategy._apply_classifier_overrides (None == leave the
    # classifier default in place).
    _CLASSIFIER_TASK_WEIGHTS = None
    _CLASSIFIER_MAX_EPOCHS = None

    # Use only real signals as the basis; the GAN provides synthetic
    # samples below, so layered signal augmentation would double-count.
    augment_training_data = False

    # Features the MT_DDPM systematically mis-reproduces — copy from real
    # class-matched rows during balancing instead of trusting the GAN's
    # output. Passthrough breaks the joint between the passed-through
    # column and other GAN-drawn features within the class (you take col
    # X from real row A but cols Y,Z were drawn for a different X), so
    # it's only worth doing on features whose marginal + per-feature
    # autocorr matter MORE than their cross-feature joints.
    #
    # Safe to passthrough (label-independent features the GAN
    # systematically mis-reproduces):
    #   - adx_scaled — pre-passthrough diagnostic showed catastrophic
    #     mode collapse: real μ=-0.80 σ≈0.15, synth μ=+0.10 σ≈0.06
    #     (6σ wrong-direction shift across most buckets). adx IS used in
    #     entry_adx_threshold but not in the gbb label formula. Randomised
    #     (adx × other features) joints within class is strictly better
    #     than mode-collapsed-to-the-wrong-mean adx.
    #   - aroonosc_scaled / macd_neg / macdhist_norm — high real lag-1
    #     autocorr (0.83-0.91) that the GAN compresses to 0.26-0.60.
    #     Smoothed indicators DDPM treats as iid.
    #   - vwap_pos / vwap_neg — same pattern (0.91→0.50 lag-1).
    #   - atr_norm / spread_ma — heavy-tailed volatility scalars,
    #     defensive carryover from NNNC_DDPM_MLX.
    #
    # NOT passthrough (label-coupled or critical for joint learning):
    #   - guard_metric_pos / guard_metric_neg — directly used in the gbb
    #     label formula and broadly correlated with adx/fastk/cci/
    #     bb_position. Passthrough would randomise the (guard × everything)
    #     joints the classifier needs. Let the downstream AE filter cull
    #     off-manifold synth on this feature.
    gan_passthrough_columns = [
        "adx_scaled",
        "aroonosc_scaled",
        "atr_norm",
        "macd_neg",
        "macdhist_norm",
        "spread_ma",
        "vwap_pos",
        "vwap_neg",
    ]

    # When True, balance_multi_task emits a per-(task, class) fidelity
    # report (mean shift in σ, std ratio, mode-collapse / off-distribution
    # flags, plus a worst-feature drilldown).  Off by default; flip on
    # when training regresses with augmentation and you want to know
    # whether the DDPM is producing usable samples.
    gan_run_diagnostics: bool = True


    gan_target_ratio = 0.5

    # ---------------------------------------------------------------------- #
    # 3-D augmentation hook                                                  #
    # ---------------------------------------------------------------------- #

    def preprocess_training_data(
        self,
        dataframe: DataFrame,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: Dict[str, np.ndarray],
        test_labels: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Balance 3-D sequential training data via MT-DDPM."""
        self._augmented_labels = None
        original_shape = np.shape(train_data)

        if len(train_data) == 0:
            print("    No training data to balance")
            return train_data, test_data, train_labels, test_labels

        try:
            save_path = gan_save_path(
                self.get_storage_location(),
                self.gan_type,
                use_pca=bool(getattr(self, "use_pca_reduction", False)),
                post_gan_scaling=bool(getattr(self, "use_post_gan_scaling", False)),
            )

            # Transform 3-D tensor (batch, seq_len, features) → minmax space.
            train_data_shape = train_data.shape
            num_features = train_data.shape[-1]
            train_2d = train_data.reshape(-1, num_features)
            gan_input = self._format_for_gan_scaler(train_2d)
            train_minmax_2d = self.normalise_for_gan(gan_input)
            if isinstance(train_minmax_2d, pd.DataFrame):
                train_minmax_2d = train_minmax_2d.to_numpy()
            train_minmax = train_minmax_2d.reshape(train_data_shape)

            print("    Balancing training data with MT DDPM (via GANInterface)")
            interface = GANInterface(GANType.MT_DDPM, save_path=save_path)
            try:
                interface.load(expected=self._gan_expected_metadata(dataframe))
                print(f"    Loaded existing MT DDPM model from {save_path}.")
            except GANMetadataMismatchError:
                # Strict validation rejects stale models — propagate so
                # the operator sees the per-key diff.
                raise
            except FileNotFoundError as load_err:
                raise RuntimeError(
                    f"MT DDPM model not found at {save_path}. "
                    f"Run CreateMTDDPM first to train and save the model. "
                    f"Error: {load_err}"
                ) from load_err
            self._apply_gan_inference_overrides(interface)

            aug_x, aug_y = self._balance_iteratively(
                interface=interface,
                train_minmax=train_minmax,
                train_labels=train_labels,
            )

            # Log per-task augmentation summary using the first configured task.
            display_task = None
            if isinstance(self.gan_target_ratio, dict):
                display_task = next(iter(self.gan_target_ratio), None)
            if display_task and display_task in aug_y:
                aug_task_idx = aug_y[display_task].argmax(axis=1)
                aug_classes, aug_counts = np.unique(aug_task_idx, return_counts=True)
                aug_counts_map = dict(zip(aug_classes.tolist(), aug_counts.tolist()))
                print("    MT DDPM augmentation complete")
                print(
                    f"    Augmented train size: {len(aug_x)}  "
                    f"{display_task} class counts: {aug_counts_map}"
                )
            else:
                print("    MT DDPM augmentation complete")
                print(f"    Augmented train size: {len(aug_x)}")
            print(f"    MT DDPM effect: shape {original_shape} -> {np.shape(aug_x)}")

            self._augmented_labels = aug_y

            # Transform aug_x back from minmax space to normalised space.
            aug_shape = aug_x.shape
            aug_2d = aug_x.reshape(-1, num_features)
            aug_input = self._format_for_gan_scaler(aug_2d)
            aug_2d_norm = self.denormalise_from_gan(aug_input)
            if isinstance(aug_2d_norm, pd.DataFrame):
                aug_2d_norm = aug_2d_norm.to_numpy()
            aug_x = aug_2d_norm.reshape(aug_shape)

            return aug_x, test_data, aug_y, test_labels

        except GANMetadataMismatchError:
            raise
        except Exception as exc:
            print("    MT DDPM encountered an error; returning original data")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
            self._augmented_labels = None
            return train_data, test_data, train_labels, test_labels

    # _balance_iteratively / _format_for_gan_scaler are inherited from
    # BaseNNMTStrategy (shared with NNMT_WGAN).

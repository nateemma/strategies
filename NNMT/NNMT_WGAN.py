# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNMT_WGAN — multi-task NNMT strategy with MT-WGAN-GP augmentation.

Uses ``GANInterface(GANType.MT_WGAN)`` for the actual fit/generate; this
class does the 3-D tensor reshape (which the BaseNN dispatcher's 2-D
path can't handle) and delegates the cross-task balanced augmentation to
``GANs.balance.balance_multi_task``.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy  # noqa: E402
from GANs.GANType import GANType  # noqa: E402
from GANs.GANInterface import GANInterface, GANMetadataMismatchError  # noqa: E402
from GANs.paths import gan_save_path  # noqa: E402


class NNMT_WGAN(NNMTStrategy):

    # GAN augmentation — multi-task WGAN-GP, applied to 3-D sequential
    # tensors in preprocess_training_data. For multi-task (dict) labels the
    # BaseNN 2-D dispatcher (enhance_training_data) passes through unchanged
    # regardless of this flag, so gan_augment does not double-augment; the
    # real balancing happens in preprocess_training_data on the 3-D tensor.
    gan_type = GANType.MT_WGAN
    gan_augment = True
    # Active gan_target_ratio is the per-task dict declared below.
    learning_rate = 1e-5
    batch_size = 1024

    # v2 pipeline: GAN consumes raw features and a tensor-level scaler runs
    # AFTER augmentation. Strategy looks for the model under
    # saved_data/GANs_PostScale/mt_wgan/ rather than saved_data/GANs/mt_wgan/.
    # Must match the same flag on CreateMTWGAN (training side); if they
    # disagree, the load path won't find a model.
    use_post_gan_scaling = True

    # Per-instance task-weight override — shifts the multi-task loss budget
    # heavily toward trading (and to a lesser extent profit) so the shared
    # backbone prioritises the heads we actually trade on. Auxiliary tasks
    # still train, just with much smaller gradient pull. Default weights
    # (defined in utils.ClassifierMLXMultiTask) gave the trading head 25%
    # of the loss budget and saw the model produce low-confidence Buy/Sell
    # predictions even on balanced training data. Normalised at runtime
    # by the classifier; raw ratios shown here for readability.
    # _CLASSIFIER_TASK_WEIGHTS = {
    #     "trading":  0.70,
    #     "profit":   0.15,
    #     "regime":   0.0375,
    #     "risk":     0.0375,
    #     "momentum": 0.0375,
    #     "flow":     0.0375,
    # }
    _CLASSIFIER_TASK_WEIGHTS = {
        "trading":  0.3,
        "profit":   0.3,
        "regime":   0.1,
        "risk":     0.1,
        "momentum": 0.1,
        "flow":     0.1,
    }

    buy_params = { **NNMTStrategy.buy_params, 
        "prediction_threshold": 0.5}

    # Per-task augmentation targets.  Accepts:
    #   * float                       — broadcast to every task in train_labels
    #   * Dict[task, float]           — per-task target
    #   * Dict[task, Dict[cls, ratio]] — per-(task, class) override
    # Same shape as ``balance_multi_task.target_ratios``.
    gan_target_ratio: Any = {
        "trading":  0.8,
        "regime":   0.8,
        "risk":     0.8,
        "momentum": 0.8,
        "flow":     0.8,
        "profit":   0.8,
    }

    # Densify the trading-task Buy/Sell labels on real rows by extending
    # each signal 2 bars earlier (see BaseNNStrategy.augment_training_signals).
    # Complementary to GAN augmentation, not redundant: this expands the
    # minority class label coverage on the real data the GAN is trained on,
    # while GAN augmentation produces synthetic rows after the fact.
    augment_training_data = True

    # When True, balance_multi_task emits a per-(task, class) fidelity
    # report (mean shift in σ, std ratio, mode-collapse / off-distribution
    # flags, plus a worst-feature drilldown).  Off by default; flip on
    # when training regresses with augmentation and you want to know
    # whether the GAN is producing usable samples.
    gan_run_diagnostics: bool = True

    # _apply_classifier_overrides is inherited from BaseNNMTStrategy; the
    # _CLASSIFIER_TASK_WEIGHTS attr above flows through it.

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
        """Balance 3-D sequential training data via MT-WGAN.

        v2 path A: train_data arrives RAW (clean_for_tensor space) — the space
        the MT-WGAN was trained on (CreateMTGANBase). The GAN self-z-scores
        internally, so it is fed the raw tensor directly (no normalise_for_gan
        round-trip). The combined real+synth tensor is then normalised by the
        column-aware main_tensor_scaler, matching the space get_predictions
        produces via scale_dataframe at inference time.
        """
        self._augmented_labels = None
        original_shape = np.shape(train_data)

        aug_x, aug_y = train_data, train_labels

        if len(train_data) == 0:
            print("    No training data to balance")
        else:
            try:
                save_path = gan_save_path(
                    self.get_storage_location(),
                    self.gan_type,
                    use_pca=bool(getattr(self, "use_pca_reduction", False)),
                    post_gan_scaling=bool(getattr(self, "use_post_gan_scaling", False)),
                )

                print("    Balancing training data with MT WGAN-GP (via GANInterface)")
                interface = GANInterface(GANType.MT_WGAN, save_path=save_path)
                try:
                    interface.load(expected=self._gan_expected_metadata(dataframe))
                    print(f"    Loaded existing MT WGAN model from {save_path}.")
                except GANMetadataMismatchError:
                    # Strict validation rejects stale models — propagate so
                    # the operator sees the per-key diff.
                    raise
                except FileNotFoundError as load_err:
                    raise RuntimeError(
                        f"MT WGAN model not found at {save_path}. "
                        f"Run CreateMTWGAN first to train and save the model. "
                        f"Error: {load_err}"
                    ) from load_err
                self._apply_gan_inference_overrides(interface)

                # GAN trained on raw; feed the raw tensor directly (it z-scores
                # internally and returns raw-space synth).
                aug_x, aug_y = self._balance_iteratively(
                    interface=interface,
                    train_minmax=train_data,
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
                    print("    MT WGAN-GP augmentation complete")
                    print(
                        f"    Augmented train size: {len(aug_x)}  "
                        f"{display_task} class counts: {aug_counts_map}"
                    )
                else:
                    print("    MT WGAN-GP augmentation complete")
                    print(f"    Augmented train size: {len(aug_x)}")
                print(f"    MT WGAN-GP effect: shape {original_shape} -> {np.shape(aug_x)}")

                self._augmented_labels = aug_y

            except GANMetadataMismatchError:
                raise
            except Exception as exc:
                print("    MT WGAN-GP encountered an error; returning original data")
                print(f"      Error: {exc}")
                print(traceback.format_exc())
                self._augmented_labels = None
                aug_x, aug_y = train_data, train_labels

        # Post-GAN column-aware normalise: raw -> normalised (scale_dataframe space).
        aug_x, test_data = self._apply_post_gan_scaler(aug_x, test_data)

        return aug_x, test_data, aug_y, test_labels

    # _balance_iteratively / _format_for_gan_scaler are inherited from
    # BaseNNMTStrategy (shared with NNMT_DDPM).

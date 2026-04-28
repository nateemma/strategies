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
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from NNMTStrategy import NNMTStrategy  # noqa: E402
from GANs.GANType import GANType  # noqa: E402
from GANs.GANInterface import GANInterface, GANMetadataMismatchError  # noqa: E402
from GANs.paths import gan_save_path  # noqa: E402


class NNMT_WGAN(NNMTStrategy):

    # GAN augmentation — multi-task WGAN-GP, applied to 3-D sequential
    # tensors in preprocess_training_data.  We turn off the BaseNN 2-D
    # dispatcher (gan_augment=False) because the augmentation is done
    # later in 3-D space.
    gan_type = GANType.MT_WGAN
    gan_augment = False

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

    # Use only real signals as the basis; the GAN provides synthetic
    # samples below, so layered signal augmentation would double-count.
    augment_training_data = False

    # When True, balance_multi_task emits a per-(task, class) fidelity
    # report (mean shift in σ, std ratio, mode-collapse / off-distribution
    # flags, plus a worst-feature drilldown).  Off by default; flip on
    # when training regresses with augmentation and you want to know
    # whether the GAN is producing usable samples.
    gan_run_diagnostics: bool = False

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
        """Balance 3-D sequential training data via MT-WGAN."""
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
            print("    MT WGAN-GP encountered an error; returning original data")
            print(f"      Error: {exc}")
            print(traceback.format_exc())
            self._augmented_labels = None
            return train_data, test_data, train_labels, test_labels

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _balance_iteratively(
        self,
        interface,
        train_minmax: np.ndarray,
        train_labels: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Delegate to ``GANs.balance.balance_multi_task``.

        Thin wrapper kept so subclasses can override scheduling without
        touching the augmentation policy itself — the algorithm lives
        in the shared utility because it isn't WGAN-specific.
        """
        from GANs.balance import balance_multi_task  # noqa: E402
        from GANs.passthrough import resolve_column_indices  # noqa: E402

        # Pass feature names through when the scaler knows them — gives
        # the diagnostic worst-feature drilldown human-readable column
        # names instead of f0..fF, and lets the passthrough config
        # specify columns by name.
        feature_names = None
        scaler = getattr(self, "gan_scaler_a", None)
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            try:
                feature_names = list(scaler.feature_names_in_)
            except Exception:
                feature_names = None

        # Translate passthrough column names → integer indices for the
        # 3-D ndarray backend.  Names not present in the scaler's
        # column set are silently dropped so an over-broad config
        # doesn't crash augmentation.
        passthrough_columns = None
        configured = getattr(self, "gan_passthrough_columns", None)
        if configured and feature_names:
            indices = resolve_column_indices(configured, feature_names)
            if indices:
                passthrough_columns = indices

        return balance_multi_task(
            interface=interface,
            data=train_minmax,
            labels=train_labels,
            target_ratios=self.gan_target_ratio,
            log=print,
            debug_log=self.debug_print,
            diagnostics=bool(getattr(self, "gan_run_diagnostics", False)),
            feature_names=feature_names,
            passthrough_columns=passthrough_columns,
        )

    def _format_for_gan_scaler(self, array_2d: np.ndarray):
        if isinstance(array_2d, pd.DataFrame):
            return array_2d
        if hasattr(self.gan_scaler_a, "feature_names_in_"):
            feature_names = list(self.gan_scaler_a.feature_names_in_)
            try:
                return pd.DataFrame(array_2d, columns=feature_names)
            except ValueError:
                pass
        return array_2d

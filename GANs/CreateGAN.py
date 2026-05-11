# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621
# type: ignore
# pylint: disable=import-error
"""
CreateGAN — single-task GAN creator strategy.

Replaces the one-class-per-backend variants (CreateWGAN, CreateCtabGanPlus,
Create_CGP_PCA).  The GAN backend is selected via the ``gan_type`` class
attribute; defaults for that backend are resolved automatically from
``_DEFAULTS_BY_TYPE``.

To create a WGAN:
    class MyWGAN(CreateGAN):
        gan_type = GANType.WGAN

To create a CTAB-GAN+ with PCA reduction:
    class MyCgpPca(CreateGAN):
        gan_type = GANType.CTAB_GAN
        use_pca_reduction = True
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateGANBase import CreateGANBase  # noqa: E402
from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402
from GANs.GANInterface import GANInterface  # noqa: E402

import dataclasses
from Framework.BaseStrategy import StrategyConfig


def _generate_pair_agnostic(
    interface: GANInterface,
    *,
    n: int,
    class_label: int,
) -> pd.DataFrame:
    """Generate samples for a class without selecting a specific pair.

    For pair-conditioned CTAB-GAN+ models, ``generate(class_label=...)``
    requires a pair_label.  This wrapper inspects the loaded model and,
    when pair conditioning is on, builds a condition_vector with the
    requested class and uniform random pairs — matching the training
    data's marginal distribution.  Used by post-training eval and
    augmentation reporting where no specific pair is being targeted.
    """
    model = getattr(interface, "_model", None)
    num_pairs = int(getattr(model, "num_pairs", 0) or 0)
    if num_pairs == 0:
        return interface.generate(n=n, class_label=class_label)

    num_classes = int(getattr(model, "num_classes", 0) or 0)
    class_oh = np.zeros((n, num_classes), dtype=np.float32)
    class_oh[:, class_label] = 1.0
    pair_ids = np.random.randint(0, num_pairs, n)
    pair_oh = np.eye(num_pairs, dtype=np.float32)[pair_ids]
    cond = np.concatenate([class_oh, pair_oh], axis=1)
    return interface.generate(n=n, condition_vector=cond)

class CreateGAN(CreateGANBase, BaseNNStrategy):
    """
    Unified single-task GAN creator.

    One class covers every single-task backend exposed via GANInterface —
    WGAN-GP and CTAB-GAN+ today, plus any future variant registered in
    ``_DEFAULTS_BY_TYPE``.  Backend-specific steps (categorical column
    detection, CTAB quality evaluation, per-class augmentation logging)
    are gated on ``self.gan_type``.
    """

    # Concrete class defaults — the WGAN path is the baseline behaviour.
    gan_type: GANType = GANType.WGAN

    aggregate_pairs = True  # use all pairs for training

    # Per-backend defaults.  The active entry is merged with any
    # caller-supplied override at construction time.
    # ``save_subdir`` removed — every GAN type now writes to
    # ``<storage>/GANs/<gan_type>`` via ``GANs.paths.gan_save_path``.
    _DEFAULTS_BY_TYPE: Dict[GANType, Dict[str, Any]] = {
        GANType.WGAN: {
            "name": "WGAN-GP",
            "description": "WGAN-GP",
            "augmentation_target_ratio": 0.4,
            "noise_std": 0.02,
            "multi_task": False,
        },
        GANType.CTAB_GAN: {
            "name": "CTAB-GAN+",
            "description": "CTAB-GAN+",
            "augmentation_target_ratio": 1.0,
            "multi_task": False,
            # None → auto-detect from one_hot_columns
            "categorical_columns": None,
        },
    }


    strategy_config = dataclasses.replace(BaseNNStrategy.strategy_config, gan_run_diagnostics=True)

    def __init__(self, gan_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(gan_config=gan_config, **kwargs)

    # ------------------------------------------------------------------ #
    # Default-config resolution                                           #
    # ------------------------------------------------------------------ #

    def get_default_gan_config(self) -> Dict[str, Any]:
        """Merge the generic base defaults with the type-specific defaults."""
        base = dict(super().get_default_gan_config())
        base.update(self._DEFAULTS_BY_TYPE.get(self.gan_type, {}))
        return base

    # ------------------------------------------------------------------ #
    # GAN training                                                        #
    # ------------------------------------------------------------------ #

    def run_gan_training(
        self,
        *,
        combined_df: DataFrame,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        config: Dict[str, Any],
        train_pair_ids: Optional[np.ndarray] = None,
        pair_names: Optional[List[str]] = None,
    ) -> None:
        try:
            if len(train_data) == 0:
                print("    No training data to balance")
                return

            train_idx = train_labels.argmax(axis=1)
            classes, counts = np.unique(train_idx, return_counts=True)
            class_counts = dict(zip(classes.tolist(), counts.tolist()))
            print(
                f"    Train set size: {len(train_data)}  Class counts: {class_counts}"
            )
            if pair_names is not None:
                print(
                    f"    Pair conditioning enabled: {len(pair_names)} pairs "
                    f"({', '.join(pair_names[:6])}"
                    f"{'...' if len(pair_names) > 6 else ''})"
                )

            save_path = self.get_gan_save_path(config)
            print(
                f"    Training {config.get('name', self.gan_type.name)} via "
                f"GANInterface (save_path={save_path})"
            )
            self._log_master_thresholds()

            if self.gan_type == GANType.CTAB_GAN:
                self._run_ctab_training(
                    train_data=train_data,
                    train_labels=train_labels,
                    classes=classes,
                    counts=counts,
                    save_path=save_path,
                    config=config,
                    train_pair_ids=train_pair_ids,
                    pair_names=pair_names,
                )
            else:
                # Default path — WGAN and any future single-task backend
                # whose interface.fit() accepts (data_2d, one_hot) without
                # per-class post-training generation logic.
                self._run_simple_training(
                    train_data=train_data,
                    train_labels=train_labels,
                    save_path=save_path,
                )

        except Exception as exc:
            backend = config.get("name", self.gan_type.name)
            print(f"    {backend} encountered an error during training")
            print(f"      Error: {exc}")
            print(traceback.format_exc())

    # ------------------------------------------------------------------ #
    # Backend-specific paths                                              #
    # ------------------------------------------------------------------ #

    def _run_simple_training(
        self,
        *,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        save_path: str,
    ) -> None:
        """WGAN-style training: fit + save, no post-training generation."""
        interface = GANInterface(self.gan_type, save_path=save_path)
        interface.fit(
            train_data.astype("float32"),
            train_labels.astype("float32"),
        )
        interface.save(**self._master_save_kwargs())
        print(f"    {self.gan_type.name} model saved to {save_path}")

    def _run_ctab_training(
        self,
        *,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        classes: np.ndarray,
        counts: np.ndarray,
        save_path: str,
        config: Dict[str, Any],
        train_pair_ids: Optional[np.ndarray] = None,
        pair_names: Optional[List[str]] = None,
    ) -> None:
        """CTAB-GAN+ training: fit + save + quality eval + augmentation report."""

        # Rebuild column names for the GAN-scaled DataFrame
        if self.use_pca_reduction and self.pca_n_components is not None:
            passthrough = [
                c for c in self.pca_passthrough_columns
                if c in (self.pca_feature_columns or []) or c in (self.include_list or [])
            ]
            pca_cols = [f"pca_{i}" for i in range(self.pca_n_components)]
            train_df_columns = passthrough + pca_cols
        elif (
            hasattr(self, "gan_scaler_a")
            and self.gan_scaler_a is not None
            and hasattr(self.gan_scaler_a, "feature_columns")
        ):
            train_df_columns = self.gan_scaler_a.feature_columns
        else:
            raise ValueError(
                "GAN scaler (gan_scaler_a) is not available. "
                "CTAB-GAN+ requires the GAN-scaled dataframe with feature_columns."
            )

        if len(train_df_columns) != train_data.shape[1]:
            raise ValueError(
                f"Column count mismatch: expected {len(train_df_columns)} columns, "
                f"but train_data has {train_data.shape[1]} columns"
            )

        train_df = pd.DataFrame(train_data, columns=train_df_columns)

        # Detect or filter categorical columns
        categorical_columns = config.get("categorical_columns")
        if categorical_columns is None:
            categorical_columns = self._get_categorical_columns(train_df)
            print(f"    Auto-detected categorical columns: {categorical_columns}")
        else:
            categorical_columns = [c for c in categorical_columns if c in train_df.columns]
            print(f"    Using provided categorical columns (filtered): {categorical_columns}")

        # Normalise labels to one-hot
        labels_arr = np.asarray(train_labels)
        if labels_arr.ndim == 1:
            num_classes = int(labels_arr.max()) + 1
            train_labels_processed = np.eye(num_classes, dtype=np.float32)[labels_arr.astype(int)]
        else:
            train_labels_processed = labels_arr.astype(np.float32)

        # Compute augmentation targets (for reporting)
        augmentation_target_ratio = config.get("augmentation_target_ratio", 1.0)
        current_max = int(counts.max()) if counts.size > 0 else 0
        if current_max <= 0:
            print("    No majority class found, skipping CTAB-GAN+")
            return
        target = int(current_max * augmentation_target_ratio)
        num_classes = train_labels_processed.shape[1]
        have_map = {c: int(train_labels_processed[:, c].sum()) for c in range(num_classes)}
        needs_map = {c: max(0, target - have_map.get(c, 0)) for c in range(num_classes)}
        print(
            f"    CTAB-GAN+ target per class: {target} "
            f"(ratio={augmentation_target_ratio})  Planned adds: {needs_map}"
        )
        if all(v <= 0 for v in needs_map.values()):
            print("    Already at or above target; skipping CTAB-GAN+")
            return

        print("    CTAB-GAN+ training starting...")

        interface = GANInterface(GANType.CTAB_GAN, save_path=save_path)
        interface.fit(
            train_df,
            train_labels_processed,
            categorical_columns=categorical_columns,
            pair_labels=train_pair_ids,
            pair_names=pair_names,
        )
        interface.save(**self._master_save_kwargs())
        print(f"    CTAB-GAN+ model saved to {save_path}")
        print(
            f"      Stored thresholds: min_buy_gain={self.MASTER_MIN_BUY_GAIN_THRESHOLD:.4f}, "
            f"min_sell_loss={self.MASTER_MIN_SELL_LOSS_THRESHOLD:.4f}, "
            f"training_type={self.MASTER_TRAINING_TYPE}"
        )

        self._evaluate_ctab_model(interface, train_df)
        self._report_ctab_augmentation(
            interface,
            train_data=train_data,
            train_labels_processed=train_labels_processed,
            needs_map=needs_map,
            train_df_columns=train_df_columns,
        )

    # ------------------------------------------------------------------ #
    # Helpers shared between CTAB single- and multi-task                   #
    # ------------------------------------------------------------------ #

    def _get_categorical_columns(self, dataframe: DataFrame) -> list:
        """Identify categorical columns from the one-hot family."""
        categorical_cols = []
        for base_col in getattr(self, "one_hot_columns", []):
            matching_cols = [
                col for col in dataframe.columns if col.startswith(f"{base_col}_")
            ]
            categorical_cols.extend(matching_cols)
        return list(set([col for col in categorical_cols if col in dataframe.columns]))

    def _evaluate_ctab_model(self, interface: GANInterface, train_df: DataFrame) -> None:
        """Run the CTAB quality metrics and log the summary."""
        eval_sample_size = min(2000, len(train_df))
        print("\n    Evaluating CTAB-GAN+ model...")
        try:
            eval_indices = np.random.choice(len(train_df), eval_sample_size, replace=False)
            generated_gan = _generate_pair_agnostic(
                interface, n=eval_sample_size, class_label=0,
            )

            print("    GAN Space Evaluation (minmax normalized to [-1, 1]):")
            eval_df_gan = train_df.iloc[eval_indices]
            eval_metrics_gan = interface._model.evaluate_with_dataframes(
                eval_df_gan, generated_gan
            )
            overall_gan = eval_metrics_gan.get("overall_score", {})
            for metric, label in [
                ("overall_quality",   "Overall Quality"),
                ("diversity_score",   "Diversity Score"),
                ("correlation_score", "Correlation Score"),
                ("statistical_score", "Statistical Score"),
                ("validity_score",    "Validity Score"),
            ]:
                print(f"      {label:<22} {overall_gan.get(metric, 0.0):.4f}")
            quality = overall_gan.get("overall_quality", 0.0)
            if quality < 0.6:
                print(f"      WARNING: Low overall quality ({quality:.4f})")
            elif quality >= 0.8:
                print(f"      Excellent model quality ({quality:.4f})")
        except Exception as eval_exc:
            print(f"      Evaluation failed: {eval_exc}")
            print(traceback.format_exc())

    def _report_ctab_augmentation(
        self,
        interface: GANInterface,
        *,
        train_data: np.ndarray,
        train_labels_processed: np.ndarray,
        needs_map: Dict[int, int],
        train_df_columns: list,
    ) -> None:
        """Generate the per-class augmentation samples and log the effect."""
        aug_data_list = [train_data]
        aug_labels_list = [train_labels_processed]
        for class_idx, need_count in needs_map.items():
            if need_count <= 0:
                continue
            print(f"    Generating {need_count} samples for class {class_idx}")
            gen_df = _generate_pair_agnostic(
                interface, n=need_count, class_label=int(class_idx),
            )
            generated_array = gen_df[train_df_columns].values.astype(np.float32)
            aug_data_list.append(generated_array)
            class_labels = np.zeros(
                (need_count, train_labels_processed.shape[1]), dtype=np.float32
            )
            class_labels[:, class_idx] = 1.0
            aug_labels_list.append(class_labels)

        if len(aug_data_list) > 1:
            aug_x = np.concatenate(aug_data_list, axis=0)
            aug_y = np.concatenate(aug_labels_list, axis=0)
            aug_idx = aug_y.argmax(axis=1)
            aug_classes, aug_counts = np.unique(aug_idx, return_counts=True)
            aug_class_counts = dict(zip(aug_classes.tolist(), aug_counts.tolist()))
            print("    CTAB-GAN+ training complete")
            print(
                f"    Augmented train size: {len(aug_x)}  Class counts: {aug_class_counts}"
            )
        else:
            print("    CTAB-GAN+ training complete, no augmentation needed")

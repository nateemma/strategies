# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error

"""CreateDiscriminator — train the unified RealnessDiscriminator.

Trained ONCE against the union of synthetic samples from every GAN family
that has a persisted model on disk (TabDDPM, CTAB-GAN+, WGAN). The
trained discriminator is then loaded at NNNC training time by
``balance_single_task`` and used to drop the lowest-realness fraction of
synth before it reaches the classifier.

Inherits the BaseNNStrategy data pipeline via CreateGAN so the real
samples it trains against are normalized the same way the classifier
sees them. ``run_gan_training`` is fully overridden — we don't actually
train a GAN here, we train a discriminator using the real data plus
synth from the already-saved GANs.

Invocation: ``zsh test_strat.sh -n 720 -o 30 GANs CreateDiscriminator``
(same pattern as CreateWGAN / CreateMTDDPM).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)
sys.path.append(str(Path(__file__).parent.parent))

from CreateGAN import CreateGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402
from GANInterface import GANInterface  # noqa: E402
from GANs.paths import gan_save_path  # noqa: E402

from Discriminators.RealnessDiscriminator import RealnessDiscriminator  # noqa: E402
from Discriminators.realness_trainer import (  # noqa: E402
    RealnessTrainConfig,
    train_realness_discriminator,
)


# GAN families we attempt to load synth from. Each entry maps a display
# name to the GANType used by GANInterface. Missing models on disk are
# silently skipped — the discriminator trains against whatever is
# available, including just one if only one is saved.
_DISCRIMINATOR_SOURCES: List[Tuple[str, GANType]] = [
    ("tab_ddpm", GANType.TAB_DDPM),
    ("ctab_gan", GANType.CTAB_GAN),
    ("wgan", GANType.WGAN),
]

# How many synth samples to generate per (GAN, class). Total positives
# per GAN per class. Across 3 GANs × 3 classes = 9 × _SYNTH_PER_CLASS
# positives, downsampled to the real-pool size during training.
_SYNTH_PER_GAN_PER_CLASS: int = 3000


class CreateDiscriminator(CreateGAN):
    """Train the unified RealnessDiscriminator.

    `gan_type` is set to NONE because this strategy isn't training a GAN.
    The base class's data-prep machinery runs regardless and hands us the
    real `train_data` + `train_labels`; we override `run_gan_training` to
    pull synth from disk-resident GANs and fit the discriminator.
    """

    gan_type = GANType.NONE
    use_post_gan_scaling = True
    gan_run_diagnostics = False  # the discriminator runs its own metrics

    # Where the trained discriminator is persisted. Sibling of the
    # GAN_PostScale directory rather than under it — the discriminator
    # is not itself a GAN.
    _DISCRIMINATOR_SUBDIR: str = "Discriminators/realness"

    def get_default_gan_config(self) -> Dict[str, Any]:
        # Reuse the base default config so MASTER threshold plumbing
        # works, but tag the description so logs are unambiguous.
        cfg = dict(super().get_default_gan_config())
        cfg["name"] = "Realness Discriminator"
        cfg["description"] = (
            "Trains the unified real-vs-synth discriminator across all "
            "available GAN families on disk"
        )
        return cfg

    # ---------------------------------------------------------------- #
    # Discriminator save path                                          #
    # ---------------------------------------------------------------- #

    def get_discriminator_save_path(self) -> str:
        return os.path.join(self.get_storage_location(), self._DISCRIMINATOR_SUBDIR)

    # ---------------------------------------------------------------- #
    # Main training override                                           #
    # ---------------------------------------------------------------- #

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
                print("    No training data for discriminator — aborting")
                return

            if train_labels.ndim != 2:
                print(
                    "    train_labels must be one-hot 2D — got shape "
                    f"{train_labels.shape}"
                )
                return

            real_features = np.asarray(train_data, dtype=np.float32)
            real_class_idx = train_labels.argmax(axis=1).astype(np.int64)
            num_classes = int(train_labels.shape[1])

            print(
                f"    Real pool: {len(real_features)} rows, "
                f"F={real_features.shape[1]}, C={num_classes}"
            )

            synth_by_gan = self._collect_synth_from_disk(
                num_features=int(real_features.shape[1]),
                num_classes=num_classes,
                per_class=_SYNTH_PER_GAN_PER_CLASS,
            )
            if not synth_by_gan:
                print(
                    "    No GAN models found on disk — discriminator needs "
                    "at least one source. Train a GAN first (CreateWGAN, "
                    "CreateMTDDPM, CreateCtabGanPlus) then re-run."
                )
                return

            model, metrics = train_realness_discriminator(
                real_features=real_features,
                real_class_idx=real_class_idx,
                synth_by_gan=synth_by_gan,
                num_classes=num_classes,
                config=RealnessTrainConfig(verbose=True),
            )

            save_path = self.get_discriminator_save_path()
            model.save(save_path)
            print(f"    RealnessDiscriminator saved to {save_path}")
            print(
                f"    Final metrics — best_val_bce={metrics.get('best_val_loss', float('nan')):.4f}  "
                f"val_acc={metrics.get('best_val_acc', float('nan')):.4f}  "
                f"epochs_run={metrics.get('epochs_run', '?')}"
            )

        except Exception as exc:
            print("    CreateDiscriminator encountered an error")
            print(f"      Error: {exc}")
            print(traceback.format_exc())

    # ---------------------------------------------------------------- #
    # Helpers                                                           #
    # ---------------------------------------------------------------- #

    def _collect_synth_from_disk(
        self,
        num_features: int,
        num_classes: int,
        per_class: int,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load every GAN found on disk and generate ``per_class`` rows
        per class. Returns a dict of ``gan_name → (features, class_idx)``.

        Missing/load-failing GANs are reported and skipped rather than
        aborting — the discriminator can still train against the GANs
        that DO load.
        """
        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        storage = self.get_storage_location()
        rng = np.random.default_rng(0)

        for label, gan_type in _DISCRIMINATOR_SOURCES:
            save_path = gan_save_path(
                storage_location=storage,
                gan_type=gan_type,
                use_pca=False,
                post_gan_scaling=True,
            )
            if not os.path.isdir(save_path):
                print(f"      [{label}] no saved model at {save_path}; skipping")
                continue

            try:
                iface = GANInterface(gan_type, save_path=save_path)
                iface.load()
            except Exception as exc:
                print(f"      [{label}] load failed: {exc}; skipping")
                continue

            try:
                feats_list = []
                cls_list = []
                for class_idx in range(num_classes):
                    gen = self._generate_for_class(
                        iface=iface,
                        class_idx=class_idx,
                        num_classes=num_classes,
                        n=per_class,
                        rng=rng,
                    )
                    if gen is None or len(gen) == 0:
                        continue
                    feats_list.append(np.asarray(gen, dtype=np.float32))
                    cls_list.append(
                        np.full(len(gen), class_idx, dtype=np.int64)
                    )

                if not feats_list:
                    print(f"      [{label}] no synth generated; skipping")
                    continue

                feats = np.concatenate(feats_list, axis=0)
                cls = np.concatenate(cls_list, axis=0)

                if feats.shape[1] != num_features:
                    print(
                        f"      [{label}] feature width mismatch "
                        f"({feats.shape[1]} vs {num_features}); skipping"
                    )
                    continue

                print(
                    f"      [{label}] generated {len(feats)} synth rows "
                    f"({per_class}/class × {num_classes} classes)"
                )
                out[label] = (feats, cls)

            except Exception as exc:
                print(f"      [{label}] generation failed: {exc}; skipping")
                continue

        return out

    @staticmethod
    def _generate_for_class(
        iface: GANInterface,
        class_idx: int,
        num_classes: int,
        n: int,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        """Dispatch on gan_type to call generate() the right way; return
        a flat 2D ndarray of synthesized features. Returns None on
        failure so the caller can skip this GAN."""
        gan_type = iface.gan_type
        if gan_type == GANType.CTAB_GAN:
            df = iface.generate(n=n, class_label=int(class_idx))
            if isinstance(df, pd.DataFrame):
                return df.to_numpy(dtype=np.float32)
            return np.asarray(df, dtype=np.float32)

        # WGAN, TabDDPM, CGAN: one-hot route.
        one_hot = np.zeros((n, num_classes), dtype=np.float32)
        one_hot[:, class_idx] = 1.0
        gen = iface.generate(n=n, one_hot=one_hot)
        gen = np.asarray(gen)
        if gen.ndim == 3:
            gen = gen[:, 0, :]
        return gen.astype(np.float32)

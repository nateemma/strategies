# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error

"""CreateRealsignalDiscriminator — train per-class real-signal classifiers.

Trains K binary classifiers (one per NNNC class) on real data only:
``y_c = (real_class == c)``. No GAN samples involved. The trained
classifiers are loaded by ``balance_single_task`` and used to drop
synth rows whose features don't look like a real example of the class
they claim to be.

Why this exists alongside CreateDiscriminator:
  * CreateDiscriminator's training set includes GAN-generated negatives,
    so the discriminator learns "GAN-ness" rather than "off-real-ness"
    — a circular signal when used to filter GAN output.
  * CreateRealsignalDiscriminator never sees a GAN sample. Each per-
    class model learns "what does a real BUY/HOLD/SELL row look like"
    and rejects synth that doesn't match.

Invocation: ``zsh test_strat.sh -n 720 -o 30 GANs CreateRealsignalDiscriminator``
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)
sys.path.append(str(Path(__file__).parent.parent))

from CreateGAN import CreateGAN  # noqa: E402
from Framework.BaseStrategy import GANType  # noqa: E402

from Discriminators.realsignal_trainer import (  # noqa: E402
    RealsignalTrainConfig,
    train_realsignal_classifiers,
)


class CreateRealsignalDiscriminator(CreateGAN):
    """Train the per-class real-signal classifiers. Inherits the
    BaseNNStrategy data pipeline via CreateGAN so real samples match
    the classifier's view exactly. ``run_gan_training`` is fully
    overridden — no GAN is trained or loaded."""

    gan_type = GANType.NONE
    use_post_gan_scaling = True
    gan_run_diagnostics = False

    _SAVE_SUBDIR: str = "Discriminators/realsignal"

    def get_default_gan_config(self) -> Dict[str, Any]:
        cfg = dict(super().get_default_gan_config())
        cfg["name"] = "Realsignal Discriminator"
        cfg["description"] = (
            "Trains per-class binary signal classifiers on real data only"
        )
        return cfg

    def get_realsignal_save_root(self) -> str:
        return os.path.join(self.get_storage_location(), self._SAVE_SUBDIR)

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
                print("    No training data — aborting")
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

            save_root = self.get_realsignal_save_root()
            metrics = train_realsignal_classifiers(
                real_features=real_features,
                real_class_idx=real_class_idx,
                num_classes=num_classes,
                save_root=save_root,
                config=RealsignalTrainConfig(verbose=True),
            )

            print(f"    RealsignalClassifiers saved under {save_root}")
            print("    Per-class final metrics:")
            for c, m in sorted(metrics.items()):
                print(
                    f"      class {c}: best_val_bce="
                    f"{m.get('best_val_loss', float('nan')):.4f} "
                    f"val_acc={m.get('best_val_acc', float('nan')):.4f} "
                    f"epochs_run={m.get('epoch', '?')}"
                )

        except Exception as exc:
            print("    CreateRealsignalDiscriminator encountered an error")
            print(f"      Error: {exc}")
            print(traceback.format_exc())

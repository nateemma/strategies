# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
# pylint: disable=import-error

"""CreateAutoencoderFilter — train per-class real-data autoencoders.

For each NNNC class, fits an MLP autoencoder on real same-class
features. No GAN samples involved in training. The trained AEs are
loaded by ``balance_single_task`` and used to score synth rows by
reconstruction MSE — synth that doesn't sit on the real class-c
manifold reconstructs poorly and gets dropped.

Why this exists alongside CreateRealsignalDiscriminator:
  * Realsignal classifier scores by class-membership confidence,
    rewarding prototype-like samples → variance compression.
  * Mahalanobis distance is centroid-symmetric, similarly rewards
    near-centroid samples → variance compression.
  * Autoencoder reconstruction error is *manifold-aware*: a real
    tail sample reconstructs well as long as it sits on the same
    manifold; an off-manifold synth reconstructs poorly regardless
    of where it sits relative to the centroid.

Invocation: ``zsh test_strat.sh -n 720 -o 30 GANs CreateAutoencoderFilter``
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

from Discriminators.autoencoder_trainer import (  # noqa: E402
    AutoencoderTrainConfig,
    train_autoencoders,
)


class CreateAutoencoderFilter(CreateGAN):
    """Train the per-class autoencoders. Reuses BaseNNStrategy data
    pipeline via CreateGAN so the real samples match the classifier's
    view exactly. ``run_gan_training`` is fully overridden — no GAN is
    trained or loaded here."""

    gan_type = GANType.NONE
    use_post_gan_scaling = True
    gan_run_diagnostics = False

    _SAVE_SUBDIR: str = "Discriminators/autoencoder"

    def get_default_gan_config(self) -> Dict[str, Any]:
        cfg = dict(super().get_default_gan_config())
        cfg["name"] = "Autoencoder Filter"
        cfg["description"] = (
            "Trains per-class real-data autoencoders for manifold-based "
            "synth filtering"
        )
        return cfg

    def get_autoencoder_save_root(self) -> str:
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

            save_root = self.get_autoencoder_save_root()
            metrics = train_autoencoders(
                real_features=real_features,
                real_class_idx=real_class_idx,
                num_classes=num_classes,
                save_root=save_root,
                config=AutoencoderTrainConfig(verbose=True),
            )

            print(f"    Autoencoders saved under {save_root}")
            print("    Per-class final metrics:")
            for c, m in sorted(metrics.items()):
                print(
                    f"      class {c}: best_val_mse="
                    f"{m.get('best_val_mse', float('nan')):.6f} "
                    f"epochs_run={m.get('epoch', '?')}"
                )

        except Exception as exc:
            print("    CreateAutoencoderFilter encountered an error")
            print(f"      Error: {exc}")
            print(traceback.format_exc())

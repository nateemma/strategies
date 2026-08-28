# pragma pylint: disable=C0103, C0114, C0115, C0116
# type: ignore
# pylint: disable=import-error
"""
CreateWGAN_NoCond — CONTROL arm builder for the WGAN class-mean A/B.

Identical to ``CreateWGAN`` except the generator's class-mean matching term is
disabled (``cond_loss_weight=0.0``), which reproduces the pre-fix WGAN-MLX
generator loss (commit 43dea2f added the term).  Everything else — data window,
thresholds, save path, epochs — is inherited, so the only difference between the
two arms is the term under test.

The GAN save path is keyed by GANType, NOT by builder class name, so this writes
to the SAME directory as CreateWGAN.  Train one arm, run its backtests, then
train the other — they cannot coexist.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from CreateWGAN import CreateWGAN  # noqa: E402
from GANs.GANInterface import GANInterface  # noqa: E402


class CreateWGAN_NoCond(CreateWGAN):

    # The ablated value. 0.0 disables the class-mean term in
    # WGANMLX.loss_gen (df_wgan_mlx.py:122).
    cond_loss_weight = 0.0

    def _run_simple_training(
        self,
        *,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        save_path: str,
        train_pair_ids: Optional[np.ndarray] = None,
        pair_names: Optional[List[str]] = None,
    ) -> None:
        # Mirrors CreateGAN._run_simple_training, with the one extra fit kwarg.
        interface = GANInterface(self.gan_type, save_path=save_path)
        fit_kwargs: Dict[str, Any] = {"cond_loss_weight": self.cond_loss_weight}
        if train_pair_ids is not None:
            fit_kwargs["pair_labels"] = train_pair_ids
        if pair_names is not None:
            fit_kwargs["pair_names"] = pair_names
        print(f"    [A/B CONTROL] cond_loss_weight={self.cond_loss_weight}")
        interface.fit(
            train_data.astype("float32"),
            train_labels.astype("float32"),
            **fit_kwargs,
        )
        interface.save(**self._master_save_kwargs())
        print(f"    {self.gan_type.name} model saved to {save_path}")

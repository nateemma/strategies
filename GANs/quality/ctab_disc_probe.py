"""CTAB discriminator-blindness probe — is the joint-structure SIGNAL there?

The budget ladder (docs/GAN_SCORECARD.md) showed CTAB-TF never starts learning
joint structure while CTAB-MLX has a clear onset at epochs 5-10, with nine causes
refuted. The remaining fork: does TF's discriminator even PENALISE missing
correlation? If it cannot tell real data from a version with identical marginals
and destroyed joints, the generator is never told anything is wrong and no
generator-side change can help.

Method: score three inputs with the SAME trained discriminator —

    real      the fixture
    shuffled  each column independently permuted WITHIN each class, so every
              class-conditional marginal is preserved exactly and only the
              cross-feature joint structure is destroyed
    fake      the generator's own output

All three are pushed through the model's own encoder so the discriminator sees
them in identical space. The statistic is

    joint_sensitivity = (d_real - d_shuffled) / (d_real - d_fake)

~0 means the discriminator is BLIND to joint structure (it scores a
joint-destroyed sample like real data); ~1 means destroying joints is as
detectable to it as the generator's own output.

ONE PROCESS PER CONFIGURATION — MLX shares a global RNG.

Usage (from repo root):
    PYTHONPATH=.:user_data/strategies .venv/bin/python \
        -m GANs.quality.ctab_disc_probe --backend tf --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def shuffle_within_class(x: np.ndarray, y_idx: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Permute every column independently within each class.

    Preserves each class-conditional marginal EXACTLY (it is a permutation of
    the same values) and destroys cross-feature structure. Shuffling globally
    instead would also break the feature/label relationship, and the
    discriminator is conditional — it would then be detecting the wrong thing.
    """
    out = x.copy()
    for c in np.unique(y_idx):
        rows = np.flatnonzero(y_idx == c)
        for j in range(x.shape[1]):
            out[rows, j] = x[rng.permutation(rows), j]
    return out


def _score_tf(model, enc: np.ndarray, oh: np.ndarray) -> float:
    import tensorflow as tf
    pac = int(model.pac)
    n = (len(enc) // pac) * pac
    if n == 0:
        raise RuntimeError(f"need at least pac={pac} rows, got {len(enc)}")
    d = tf.reshape(tf.constant(enc[:n], dtype=tf.float32),
                   [n // pac, pac * model.total_feature_dim])
    lab = tf.reshape(tf.constant(oh[:n], dtype=tf.float32),
                     [n // pac, pac * model.num_classes])
    return float(np.mean(model.discriminator([d, lab], training=False).numpy()))


def _score_mlx(model, enc: np.ndarray, oh: np.ndarray) -> float:
    import mlx.core as mx
    cond = oh
    want = int(getattr(model, "cond_dim", oh.shape[1]))
    if cond.shape[1] < want:  # pair conditioning appended — pad with zeros
        cond = np.hstack([cond, np.zeros((len(cond), want - cond.shape[1]), "float32")])
    elif cond.shape[1] > want:
        cond = cond[:, :want]
    scores = model.critic(mx.array(enc.astype("float32")),
                          mx.array(cond.astype("float32")))
    return float(np.mean(np.asarray(scores)))


def run(backend: str, epochs: int, seed: int, n_rows: int) -> dict:
    import GANs.backends  # noqa: F401 -- populates the registry
    from GANs.GANBackend import resolve_backend
    from GANs.GANType import GANType
    from GANs.quality.real_fixture import build_real_fixture
    from GANs.quality.scorecard import ADAPTERS

    prefer_mlx = backend == "mlx"
    x, y, _names = build_real_fixture(seed=seed)
    y_idx = y.argmax(1)
    cls = resolve_backend(GANType.CTAB_GAN, prefer_mlx=prefer_mlx)
    if cls.__name__.endswith("MLXBackend") != prefer_mlx:
        raise SystemExit(f"asked for {backend}, resolved {cls.__name__}")

    ad = ADAPTERS["CTAB_GAN"]
    data, labels = ad.to_fit(x, y)
    iface = cls()
    iface.fit(data, labels, epochs=epochs)
    model = iface._model

    rng = np.random.default_rng(seed)
    x_shuf = shuffle_within_class(x, y_idx, rng)

    cols = list(data.columns)
    df_real = pd.DataFrame(x, columns=cols)
    df_shuf = pd.DataFrame(x_shuf, columns=cols)

    # Generator output, decoded, then re-encoded like the other two so the
    # discriminator sees all three inputs in exactly the same space.
    synth_df, synth_oh = ad.generate(iface, n_rows, y)
    if not isinstance(synth_df, pd.DataFrame):
        synth_df = pd.DataFrame(np.asarray(synth_df), columns=cols)

    encode = model._transform_data if backend == "tf" else model._transform
    score = _score_tf if backend == "tf" else _score_mlx

    take = min(n_rows, len(x))
    d_real = score(model, encode(df_real.iloc[:take]), y[:take])
    d_shuf = score(model, encode(df_shuf.iloc[:take]), y[:take])
    d_fake = score(model, encode(synth_df), synth_oh)

    denom = d_real - d_fake
    sens = (d_real - d_shuf) / denom if abs(denom) > 1e-9 else float("nan")

    # Sanity: the shuffle must actually have destroyed joint structure.
    def mean_abs_corr(a):
        m = np.nan_to_num(np.corrcoef(a, rowvar=False))
        return float(np.abs(m[np.triu_indices_from(m, k=1)]).mean())

    return {
        "backend": backend, "epochs": epochs, "seed": seed,
        "d_real": d_real, "d_shuf": d_shuf, "d_fake": d_fake,
        "joint_sensitivity": sens,
        "corr_real": mean_abs_corr(x[:take]),
        "corr_shuf": mean_abs_corr(x_shuf[:take]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("tf", "mlx"), required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rows", type=int, default=2400)
    a = ap.parse_args()
    r = run(a.backend, a.epochs, a.seed, a.n_rows)
    print(
        "RESULT "
        f"backend={r['backend']} epochs={r['epochs']} seed={r['seed']} "
        f"d_real={r['d_real']:+.4f} d_shuf={r['d_shuf']:+.4f} "
        f"d_fake={r['d_fake']:+.4f} joint_sens={r['joint_sensitivity']:+.3f} "
        f"corr_real={r['corr_real']:.3f} corr_shuf={r['corr_shuf']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

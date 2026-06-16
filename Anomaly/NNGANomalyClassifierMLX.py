from __future__ import annotations

# MLX (Apple mlx) port of the TensorFlow GANomaly anomaly-detector classifiers.
#
# This is a CORRECT reimplementation of NNGANomalyClassifier.py: encoder /
# decoder / classifier_head weights actually flow across the three training
# phases (the TF version discarded them, leaving a random frozen encoder).
#
# Architecture (one unified nn.Module per instance):
#   encoder:         (B, S, F) -> z (B, L)        L = ganomaly_latent_dim = 32
#   decoder:         z (B, L)   -> (B, S, F)        final activation tanh
#   classifier_head: z (B, L)   -> (B, 3) softmax   shared across variants
# A SEPARATE discriminator nn.Module lives on the predictor (training only;
# not part of the saved model, so only enc/dec/head weights are persisted).
#
# Variants differ ONLY in encoder body, decoder body, discriminator body.
#
# Float32 throughout (no mixed precision).
import logging
import sys
from enum import Enum
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from Predictors.MLXBaseAnomalyDetector import MLXBaseAnomalyDetector


log = logging.getLogger(__name__)

LEAKY_SLOPE = 0.2
DROPOUT = 0.2


def nearest_power_of_2(n: int) -> int:
    return 2 ** n.bit_length()


# -----------------------------------------------------------------------
# Shared classifier head (z (B, L) -> (B, 3) softmax)
# -----------------------------------------------------------------------


class ClassifierHead(nn.Module):
    """Mirrors the TF classifier head: Dense(elu)->drop->Dense(elu)->drop->out."""

    def __init__(self, latent_size: int):
        super().__init__()
        d1 = max(8, nearest_power_of_2((latent_size // 2) + 1))
        d2 = max(4, d1 // 2)
        self.fc1 = nn.Linear(latent_size, d1)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(d1, d2)
        self.drop2 = nn.Dropout(0.4)
        self.out = nn.Linear(d2, 3)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.drop1(nn.elu(self.fc1(z)))
        x = self.drop2(nn.elu(self.fc2(x)))
        return mx.softmax(self.out(x), axis=-1)


# -----------------------------------------------------------------------
# Unified model: holds encoder + decoder + classifier_head
# -----------------------------------------------------------------------


class GANomalyModel(nn.Module):
    """
    Unified model. __call__(x) -> (reconstruction, trading).
    The encoder/decoder/classifier_head submodules are built by the variant.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier_head = ClassifierHead(latent_size)

    def __call__(self, x: mx.array):
        z = self.encoder(x)
        rec = self.decoder(z)
        trd = self.classifier_head(z)
        return rec, trd


class GANomalyBowTie(nn.Module):
    """
    Paper-faithful GANomaly generator (Akcay et al., ACCV 2018): an
    encoder-decoder-encoder "bow-tie". encoder Gₑ(x)->z, decoder G_d(z)->x̂,
    second encoder E(x̂)->ẑ. The anomaly score is the latent distance ‖z-ẑ‖
    (computed in predict), not the input-space reconstruction error.

    classifier_head is a codebase-specific trading head (not part of GANomaly),
    trained on z. __call__ returns (reconstruction, trading) so the model still
    satisfies the predict() contract; training/scoring use the sub-encoders
    directly.
    """

    def __init__(self, encoder, decoder, encoder2, latent_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder2 = encoder2
        self.classifier_head = ClassifierHead(latent_size)

    def __call__(self, x: mx.array):
        z = self.encoder(x)
        rec = self.decoder(z)
        trd = self.classifier_head(z)
        return rec, trd


# -----------------------------------------------------------------------
# Transformer block (encoder-style: self-attn + FFN, pre-norm residuals)
# -----------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, num_heads: int, dff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.att = nn.MultiHeadAttention(dims, num_heads)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dims)
        self.ff1 = nn.Linear(dims, dff)
        self.ff2 = nn.Linear(dff, dims)
        self.drop2 = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.ln1(x)
        h = self.att(h, h, h)
        x = x + self.drop1(h)
        h = self.ln2(x)
        h = self.ff2(nn.relu(self.ff1(h)))
        x = x + self.drop2(h)
        return x


# =======================================================================
# Variant bodies — encoder / decoder / discriminator
# =======================================================================

# --- Dense -------------------------------------------------------------


class _DenseEncoder(nn.Module):
    def __init__(self, seq_len, num_features, latent_size):
        super().__init__()
        flat = seq_len * num_features
        self.fc1 = nn.Linear(flat, 512)
        self.bn1 = nn.BatchNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm(128)
        self.to_latent = nn.Linear(128, latent_size)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.bn1(nn.leaky_relu(self.fc1(x), LEAKY_SLOPE))
        x = self.bn2(nn.leaky_relu(self.fc2(x), LEAKY_SLOPE))
        x = self.bn3(nn.leaky_relu(self.fc3(x), LEAKY_SLOPE))
        return self.to_latent(x)


class _DenseDecoder(nn.Module):
    def __init__(self, seq_len, num_features, latent_size):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.fc1 = nn.Linear(latent_size, 128)
        self.bn1 = nn.BatchNorm(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm(512)
        self.out = nn.Linear(512, seq_len * num_features)

    def __call__(self, z):
        x = self.bn1(nn.leaky_relu(self.fc1(z), LEAKY_SLOPE))
        x = self.bn2(nn.leaky_relu(self.fc2(x), LEAKY_SLOPE))
        x = self.bn3(nn.leaky_relu(self.fc3(x), LEAKY_SLOPE))
        x = mx.tanh(self.out(x))
        return x.reshape(x.shape[0], self.seq_len, self.num_features)


class _DenseDiscriminator(nn.Module):
    def __init__(self, seq_len, num_features):
        super().__init__()
        flat = seq_len * num_features
        self.fc1 = nn.Linear(flat, 512)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.3)
        self.out = nn.Linear(128, 1)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.drop1(nn.leaky_relu(self.fc1(x), LEAKY_SLOPE))
        x = self.drop2(nn.leaky_relu(self.fc2(x), LEAKY_SLOPE))
        x = self.drop3(nn.leaky_relu(self.fc3(x), LEAKY_SLOPE))
        return mx.sigmoid(self.out(x))


# --- LSTM --------------------------------------------------------------


class _LSTMEncoder(nn.Module):
    def __init__(self, seq_len, num_features, latent_size, units):
        super().__init__()
        self.lstm1 = nn.LSTM(num_features, units)
        self.ln1 = nn.LayerNorm(units)
        self.lstm2 = nn.LSTM(units, units // 2)
        self.ln2 = nn.LayerNorm(units // 2)
        self.to_latent = nn.Linear(units // 2, latent_size)

    def __call__(self, x):
        out, _ = self.lstm1(x)  # (B, S, units)
        out = self.ln1(out)
        out, _ = self.lstm2(out)  # (B, S, units//2)
        out = self.ln2(out)
        last = out[:, -1, :]  # (B, units//2)
        return self.to_latent(last)


class _LSTMDecoder(nn.Module):
    def __init__(self, seq_len, num_features, latent_size, units):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = units // 2
        self.project = nn.Linear(latent_size, seq_len * self.hidden)
        self.lstm1 = nn.LSTM(self.hidden, units // 2)
        self.ln1 = nn.LayerNorm(units // 2)
        self.lstm2 = nn.LSTM(units // 2, units)
        self.ln2 = nn.LayerNorm(units)
        self.out = nn.Linear(units, num_features)

    def __call__(self, z):
        x = self.project(z)  # (B, S*hidden)
        x = x.reshape(z.shape[0], self.seq_len, self.hidden)
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out, _ = self.lstm2(out)
        out = self.ln2(out)
        return mx.tanh(self.out(out))


class _LSTMDiscriminator(nn.Module):
    def __init__(self, seq_len, num_features, units):
        super().__init__()
        self.lstm1 = nn.LSTM(num_features, units)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(units, units // 2)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Linear(units // 2, 128)
        self.drop3 = nn.Dropout(0.3)
        self.out = nn.Linear(128, 1)

    def __call__(self, x):
        return mx.sigmoid(self.out(self.features(x)))

    def features(self, x):
        """Penultimate (pre-logit) activations — the feature layer used for the
        GANomaly feature-matching adversarial loss ‖f(x) - f(x̂)‖."""
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        last = self.drop2(out[:, -1, :])
        return self.drop3(nn.relu(self.fc(last)))


# --- Attention ---------------------------------------------------------


class _AttnBlock(nn.Module):
    """Single attention block (pre-norm attn + FFN), feature-dim preserved."""

    def __init__(self, dims, num_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dims)
        self.att = nn.MultiHeadAttention(dims, num_heads)
        self.ln2 = nn.LayerNorm(dims)
        self.ff1 = nn.Linear(dims, ff_dim)
        self.drop = nn.Dropout(dropout)
        self.ff2 = nn.Linear(ff_dim, dims)

    def __call__(self, x):
        h = self.ln1(x)
        if x.shape[1] > 1:
            h = self.att(h, h, h)
            x = x + h
        h = self.ln2(x)
        h = self.ff2(self.drop(nn.relu(self.ff1(h))))
        return x + h


class _AttnEncoder(nn.Module):
    def __init__(self, seq_len, num_features, latent_size, num_heads, ff_dim, dropout):
        super().__init__()
        self.block = _AttnBlock(num_features, num_heads, ff_dim, dropout)
        self.to_latent = nn.Linear(num_features, latent_size)

    def __call__(self, x):
        x = self.block(x)  # (B, S, F)
        pooled = mx.mean(x, axis=1)  # (B, F)
        return self.to_latent(pooled)


class _AttnDecoder(nn.Module):
    def __init__(self, seq_len, num_features, latent_size, num_heads, ff_dim, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.project = nn.Linear(latent_size, seq_len * num_features)
        self.block = _AttnBlock(num_features, num_heads, ff_dim, dropout)
        self.out = nn.Linear(num_features, num_features)

    def __call__(self, z):
        x = self.project(z).reshape(z.shape[0], self.seq_len, self.num_features)
        x = self.block(x)
        return mx.tanh(self.out(x))


class _AttnDiscriminator(nn.Module):
    def __init__(self, seq_len, num_features, num_heads, ff_dim, dropout):
        super().__init__()
        self.block = _AttnBlock(num_features, num_heads, ff_dim, dropout)
        self.fc = nn.Linear(num_features, 128)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 1)

    def __call__(self, x):
        x = self.block(x)
        pooled = mx.mean(x, axis=1)
        h = self.drop(nn.relu(self.fc(pooled)))
        return mx.sigmoid(self.out(h))


# --- Transformer -------------------------------------------------------


class _TransformerEncoder(nn.Module):
    def __init__(
        self, seq_len, num_features, latent_size, t_size, num_heads, dff, num_layers, dropout
    ):
        super().__init__()
        self.proj = nn.Linear(num_features, t_size)
        self.blocks = [TransformerBlock(t_size, num_heads, dff, dropout) for _ in range(num_layers)]
        self.to_latent = nn.Linear(t_size, latent_size)

    def __call__(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        pooled = mx.mean(x, axis=1)
        return self.to_latent(pooled)


class _TransformerDecoder(nn.Module):
    def __init__(
        self, seq_len, num_features, latent_size, t_size, num_heads, dff, num_layers, dropout
    ):
        super().__init__()
        self.seq_len = seq_len
        self.t_size = t_size
        self.project = nn.Linear(latent_size, seq_len * t_size)
        self.blocks = [TransformerBlock(t_size, num_heads, dff, dropout) for _ in range(num_layers)]
        self.out = nn.Linear(t_size, num_features)

    def __call__(self, z):
        x = self.project(z).reshape(z.shape[0], self.seq_len, self.t_size)
        for b in self.blocks:
            x = b(x)
        return mx.tanh(self.out(x))


class _TransformerDiscriminator(nn.Module):
    def __init__(self, seq_len, num_features, t_size, num_heads, dff, num_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(num_features, t_size)
        self.blocks = [TransformerBlock(t_size, num_heads, dff, dropout) for _ in range(num_layers)]
        self.fc = nn.Linear(t_size, 128)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(128, 1)

    def __call__(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        pooled = mx.mean(x, axis=1)
        h = self.drop(nn.relu(self.fc(pooled)))
        return mx.sigmoid(self.out(h))


# =======================================================================
# Base detector — 3-phase train(), shared by all variants
# =======================================================================


def _bce(pred, target):
    """Binary cross-entropy on already-sigmoid'd predictions."""
    eps = 1e-7
    pred = mx.clip(pred, eps, 1.0 - eps)
    return -mx.mean(target * mx.log(pred) + (1.0 - target) * mx.log(1.0 - pred))


class GANomalyDetectorMLXBase(MLXBaseAnomalyDetector):
    """Shared 3-phase GANomaly training. Variants override _build_components()."""

    clean_data_required = True

    # Hyperparams (mirror TF base)
    ganomaly_latent_dim = 32
    ganomaly_generator_lr = 0.0002
    ganomaly_discriminator_lr = 0.0002
    ganomaly_beta1 = 0.5
    ganomaly_beta2 = 0.999
    ganomaly_batch_size = 2048

    # Per-phase epoch counts (overridable, e.g. for smoke tests)
    gan_epochs = 100
    ae_epochs = 50
    clf_epochs = 100

    # -------------------------------------------------------------------
    # Architecture hooks (variants override _build_components)
    # -------------------------------------------------------------------

    def _build_components(self, seq_len, num_features, latent_size):
        """Return (encoder, decoder, discriminator) nn.Modules."""
        raise NotImplementedError

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        self.seq_len = seq_len
        self.num_features = num_features
        latent = self.ganomaly_latent_dim
        encoder, decoder, discriminator = self._build_components(seq_len, num_features, latent)
        self.discriminator = discriminator
        model = GANomalyModel(encoder, decoder, latent)
        mx.eval(model.parameters())
        mx.eval(self.discriminator.parameters())
        return model

    # -------------------------------------------------------------------
    # 3-phase training
    # -------------------------------------------------------------------

    def train(
        self,
        df_train_norm,
        df_test_norm,
        train_results,
        test_results,
        force_train=False,
        class_weights=None,
    ):
        if not force_train and self.model_exists():
            self.load()
            if self.model is not None:
                self.is_trained = True
                return

        # Convert to tensor
        if self.dataframeUtils.is_dataframe(df_train_norm):
            train_tensor = self.dataframeUtils.df_to_tensor(df_train_norm, self.seq_len, method=0)
        else:
            train_tensor = np.asarray(df_train_norm)
        train_tensor = np.asarray(train_tensor, dtype=np.float32)

        # Split normal (HOLD == class 1) vs signal (SELL 0 / BUY 2)
        if isinstance(train_results, dict) and "trading" in train_results:
            trading_labels = np.asarray(train_results["trading"])
            if trading_labels.ndim > 1:
                class_indices = np.argmax(trading_labels, axis=1)
            else:
                class_indices = trading_labels
            normal_mask = class_indices == 1
            signal_mask = (class_indices == 0) | (class_indices == 2)
            normal_data = train_tensor[normal_mask]
            signal_data = train_tensor[signal_mask]
            signal_labels = trading_labels[signal_mask]
            print(
                f"    Data split: {len(normal_data)} normal samples, "
                f"{len(signal_data)} signal samples"
            )
        else:
            normal_data = train_tensor
            signal_data = train_tensor
            signal_labels = (
                np.asarray(train_results["trading"])
                if isinstance(train_results, dict)
                else np.asarray(train_results)
            )
            print(f"    WARNING: Could not split, using all {len(normal_data)} samples")

        # Build unified model + discriminator
        self.model = self.create_model(self.seq_len, self.num_features)

        # Phase 3 trains the head on ALL classes (Hold/Buy/Sell) so it models
        # Hold natively instead of only Buy/Sell (which collapses Hold to 0).
        if isinstance(train_results, dict) and "trading" in train_results:
            full_labels = np.asarray(train_results["trading"])
        else:
            full_labels = signal_labels

        # Variant-specific training core. Base = the legacy 3-phase flow; the
        # LSTM variant overrides this with paper-faithful joint bow-tie training.
        self._train_core(normal_data, train_tensor, full_labels)

        self.save()
        self.is_trained = True

    def _train_core(self, normal_data, train_tensor, full_labels):
        """Legacy 3-phase GANomaly training (noise-GAN warm-start, autoencoder,
        classifier head). Variants may override; the base keeps this for the
        Dense/Attention/Transformer bodies that were not converted to the
        paper-faithful encoder-decoder-encoder form."""
        print("    Phase 1: Training GAN (decoder vs discriminator)...")
        self._train_gan_phase(normal_data)

        print("    Phase 2: Training autoencoder (encoder + decoder)...")
        self._train_autoencoder_phase(normal_data)

        print("    Phase 3: Training classifier head (all classes)...")
        self._train_classifier_phase(train_tensor, full_labels)

    # ---- Phase 1: GAN -------------------------------------------------

    def _train_gan_phase(self, normal_data):
        if len(normal_data) == 0:
            print("      No normal data; skipping GAN phase")
            return

        latent = self.ganomaly_latent_dim
        bs = min(self.ganomaly_batch_size, len(normal_data))
        decoder = self.model.decoder
        disc = self.discriminator

        d_opt = optim.Adam(
            learning_rate=self.ganomaly_discriminator_lr,
            betas=(self.ganomaly_beta1, self.ganomaly_beta2),
        )
        g_opt = optim.Adam(
            learning_rate=self.ganomaly_generator_lr,
            betas=(self.ganomaly_beta1, self.ganomaly_beta2),
        )

        # Discriminator loss: real -> 0.9, fake(detached) -> 0.0
        def disc_loss_fn(disc_model, real, z):
            fake = mx.stop_gradient(decoder(z))
            loss_real = _bce(disc_model(real), mx.full(real.shape[0], 0.9))
            loss_fake = _bce(disc_model(fake), mx.zeros((fake.shape[0],)))
            return 0.5 * (loss_real + loss_fake)

        # Generator (decoder) loss vs the LIVE discriminator: fake -> 1.0
        def gen_loss_fn(dec_model, z):
            fake = dec_model(z)
            return _bce(disc(fake), mx.ones((fake.shape[0],)))

        disc_grad = nn.value_and_grad(disc, disc_loss_fn)
        gen_grad = nn.value_and_grad(decoder, gen_loss_fn)

        n = len(normal_data)
        batches = max(1, n // bs)
        for epoch in range(self.gan_epochs):
            perm = np.random.permutation(n)
            d_acc, g_acc = 0.0, 0.0
            disc.train()
            decoder.train()
            for bi in range(batches):
                idx = perm[bi * bs : (bi + 1) * bs]
                real = mx.array(normal_data[idx])
                cur = real.shape[0]

                # Discriminator step
                z = mx.random.normal((cur, latent))
                ld, gd = disc_grad(disc, real, z)
                d_opt.update(disc, gd)
                mx.eval(disc.parameters(), d_opt.state, ld)

                # Generator (decoder) step vs live discriminator
                z = mx.random.normal((cur, latent))
                lg, gg = gen_grad(decoder, z)
                g_opt.update(decoder, gg)
                mx.eval(decoder.parameters(), g_opt.state, lg)

                d_acc += ld.item()
                g_acc += lg.item()

            if epoch % 20 == 0 or epoch == self.gan_epochs - 1:
                print(
                    f"      GAN epoch {epoch}/{self.gan_epochs} - "
                    f"D {d_acc / batches:.4f}, G {g_acc / batches:.4f}"
                )

            # Release MLX's Metal buffer cache each epoch; otherwise freed
            # buffers accumulate and the allocator thrashes, inflating memory
            # and slowing training (mirrors df_ctab_gan_mlx.py).
            mx.clear_cache()

    # ---- Phase 2: Autoencoder -----------------------------------------

    def _train_autoencoder_phase(self, normal_data):
        if len(normal_data) == 0:
            print("      No normal data; skipping autoencoder phase")
            return

        encoder = self.model.encoder
        decoder = self.model.decoder

        split = max(1, int(0.8 * len(normal_data)))
        train_split = normal_data[:split]
        val_split = normal_data[split:]

        opt = optim.Adam(learning_rate=self.ganomaly_generator_lr)
        ae_bs = 32

        # Loss as a function of (encoder, decoder) packaged as a tiny module.
        ae = nn.Module()
        ae.encoder = encoder
        ae.decoder = decoder

        def ae_loss_fn(ae_model, x):
            rec = ae_model.decoder(ae_model.encoder(x))
            return mx.mean(mx.square(rec - x))

        ae_grad = nn.value_and_grad(ae, ae_loss_fn)

        n = len(train_split)
        batches = max(1, n // ae_bs)
        best_val = float("inf")
        patience, bad = 10, 0
        for epoch in range(self.ae_epochs):
            perm = np.random.permutation(n)
            ae.train()
            tr_loss = 0.0
            for bi in range(batches):
                idx = perm[bi * ae_bs : (bi + 1) * ae_bs]
                x = mx.array(train_split[idx])
                loss, grads = ae_grad(ae, x)
                opt.update(ae, grads)
                mx.eval(ae.parameters(), opt.state, loss)
                tr_loss += loss.item()

            # Validation (batched: a single forward pass over the whole val
            # split materialises activations for every sample at once).
            if len(val_split) > 0:
                ae.eval()
                vbs = 256
                sse = 0.0
                cnt = 0
                for vs in range(0, len(val_split), vbs):
                    vb = mx.array(val_split[vs : vs + vbs])
                    vrec = decoder(encoder(vb))
                    bsse = mx.sum(mx.square(vrec - vb))
                    mx.eval(bsse)
                    sse += bsse.item()
                    cnt += int(vb.size)
                    mx.clear_cache()
                v = sse / max(1, cnt)
            else:
                v = tr_loss / batches

            if v < best_val - 1e-6:
                best_val = v
                bad = 0
            else:
                bad += 1

            if epoch % 10 == 0 or epoch == self.ae_epochs - 1:
                print(
                    f"      AE epoch {epoch}/{self.ae_epochs} - "
                    f"train {tr_loss / batches:.5f}, val {v:.5f}"
                )
            mx.clear_cache()
            if bad >= patience:
                print(f"      AE early stop at epoch {epoch} (val {best_val:.5f})")
                break

    # ---- Phase 3: Classifier head -------------------------------------

    def _train_classifier_phase(self, data, labels):
        if len(data) == 0:
            print("      No data; skipping classifier phase")
            return

        encoder = self.model.encoder
        head = self.model.classifier_head

        if labels.ndim > 1:
            labels_1d = np.argmax(labels, axis=1)
        else:
            labels_1d = labels.astype(int)
        labels_oh = np.eye(3, dtype=np.float32)[labels_1d]

        # Inverse-frequency class weights: the head trains on ALL three classes,
        # and Hold is the dominant class — without weighting the head would
        # collapse to all-Hold. Weights normalised so the mean weight is ~1.
        counts = np.bincount(labels_1d, minlength=3).astype(np.float32)
        inv = np.where(counts > 0, 1.0 / counts, 0.0)
        cls_w = (inv / inv.sum() * 3.0) if inv.sum() > 0 else np.ones(3, np.float32)
        cls_w = cls_w.astype(np.float32)
        w_vec = mx.array(cls_w)
        print(
            f"      Class counts (Sell/Hold/Buy): {counts.astype(int).tolist()}, "
            f"weights: {[round(float(x), 2) for x in cls_w]}"
        )

        # 80/20 split
        n = len(data)
        perm = np.random.permutation(n)
        split = max(1, int(0.8 * n))
        tr_idx, val_idx = perm[:split], perm[split:]
        x_tr, y_tr = data[tr_idx], labels_oh[tr_idx]
        x_val, y_val = data[val_idx], labels_oh[val_idx]

        opt = optim.Adam(learning_rate=1e-3)

        # Class-weighted cross-entropy; only the head is updated (encoder detached).
        def clf_loss_fn(head_model, z_detached, y):
            probs = mx.clip(head_model(z_detached), 1e-7, 1.0)
            ce = -mx.sum(y * mx.log(probs), axis=-1)
            sw = mx.sum(y * w_vec, axis=-1)
            return mx.mean(sw * ce)

        clf_grad = nn.value_and_grad(head, clf_loss_fn)

        bs = min(self.batch_size, len(x_tr))
        batches = max(1, len(x_tr) // bs)
        best_val = float("inf")
        patience, bad = 10, 0
        encoder.eval()  # encoder frozen during phase 3
        for epoch in range(self.clf_epochs):
            p = np.random.permutation(len(x_tr))
            head.train()
            tr_loss = 0.0
            for bi in range(batches):
                idx = p[bi * bs : (bi + 1) * bs]
                x = mx.array(x_tr[idx])
                y = mx.array(y_tr[idx])
                z = mx.stop_gradient(encoder(x))
                loss, grads = clf_grad(head, z, y)
                opt.update(head, grads)
                mx.eval(head.parameters(), opt.state, loss)
                tr_loss += loss.item()

            # Weighted validation loss (same objective as training), batched
            # so the encoder forward pass doesn't run over all val samples at
            # once. Accumulate the summed weighted CE and divide by sample
            # count to reproduce mx.mean(sw * ce) exactly.
            if len(x_val) > 0:
                head.eval()
                num = 0.0
                cnt = 0
                for vs in range(0, len(x_val), bs):
                    xb = mx.array(x_val[vs : vs + bs])
                    yb = mx.array(y_val[vs : vs + bs])
                    z = mx.stop_gradient(encoder(xb))
                    probs = mx.clip(head(z), 1e-7, 1.0)
                    ce = -mx.sum(yb * mx.log(probs), axis=-1)
                    sw = mx.sum(yb * w_vec, axis=-1)
                    s = mx.sum(sw * ce)
                    mx.eval(s)
                    num += s.item()
                    cnt += int(xb.shape[0])
                    mx.clear_cache()
                v = num / max(1, cnt)
            else:
                v = tr_loss / batches

            if v < best_val - 1e-6:
                best_val = v
                bad = 0
            else:
                bad += 1

            if epoch % 20 == 0 or epoch == self.clf_epochs - 1:
                print(
                    f"      CLF epoch {epoch}/{self.clf_epochs} - "
                    f"train {tr_loss / batches:.5f}, val {v:.5f}"
                )
            mx.clear_cache()
            if bad >= patience:
                print(f"      CLF early stop at epoch {epoch} (val {best_val:.5f})")
                break


# =======================================================================
# Concrete variants
# =======================================================================


class GANomalyClassifierMLX_Dense(GANomalyDetectorMLXBase):
    def _build_components(self, seq_len, num_features, latent_size):
        enc = _DenseEncoder(seq_len, num_features, latent_size)
        dec = _DenseDecoder(seq_len, num_features, latent_size)
        disc = _DenseDiscriminator(seq_len, num_features)
        return enc, dec, disc


class GANomalyClassifierMLX_LSTM(GANomalyDetectorMLXBase):
    """Paper-faithful GANomaly (Akcay et al., 2018) on an LSTM backbone:
    encoder-decoder-encoder bow-tie, joint adversarial + contextual + encoder
    training, and a latent ‖z-ẑ‖ anomaly score. (The other variants in this
    file still use the legacy 3-phase autoencoder form.)"""

    lstm_units = 128

    # Joint-loss weights. Akcay et al. use w_adv=1, w_con=50, w_enc=1.
    # adv_weight=0 disables the adversarial term → a non-GAN bow-tie ablation
    # (used by NNAnomaly_MLX).
    adv_weight = 1.0
    con_weight = 50.0
    enc_weight = 1.0

    def _build_components(self, seq_len, num_features, latent_size):
        u = self.lstm_units
        enc = _LSTMEncoder(seq_len, num_features, latent_size, u)
        dec = _LSTMDecoder(seq_len, num_features, latent_size, u)
        disc = _LSTMDiscriminator(seq_len, num_features, u)
        return enc, dec, disc

    def create_model(self, seq_len: int, num_features: int) -> nn.Module:
        self.seq_len = seq_len
        self.num_features = num_features
        latent = self.ganomaly_latent_dim
        u = self.lstm_units
        enc = _LSTMEncoder(seq_len, num_features, latent, u)
        dec = _LSTMDecoder(seq_len, num_features, latent, u)
        enc2 = _LSTMEncoder(seq_len, num_features, latent, u)  # second encoder E
        self.discriminator = _LSTMDiscriminator(seq_len, num_features, u)
        model = GANomalyBowTie(enc, dec, enc2, latent)
        mx.eval(model.parameters())
        mx.eval(self.discriminator.parameters())
        return model

    def _train_core(self, normal_data, train_tensor, full_labels):
        print("    GANomaly joint training (adv feature-match + L1 con + L2 enc)...")
        self._train_ganomaly_joint(normal_data)
        print("    Training classifier head (all classes)...")
        self._train_classifier_phase(train_tensor, full_labels)

    def _train_ganomaly_joint(self, normal_data):
        """Joint bow-tie training. D discriminates real x vs reconstruction
        x̂=G(x); G minimises feature-matching adversarial + L1 contextual +
        L2 encoder loss. With adv_weight=0 the D updates and adv term are
        skipped (pure bow-tie autoencoder + latent-consistency)."""
        if len(normal_data) == 0:
            print("      No normal data; skipping GANomaly training")
            return

        enc = self.model.encoder
        dec = self.model.decoder
        enc2 = self.model.encoder2
        disc = self.discriminator
        use_adv = self.adv_weight > 0.0

        # Wrap the three sub-encoders so a single grad covers all generator params.
        gen = nn.Module()
        gen.encoder = enc
        gen.decoder = dec
        gen.encoder2 = enc2

        betas = (self.ganomaly_beta1, self.ganomaly_beta2)
        g_opt = optim.Adam(learning_rate=self.ganomaly_generator_lr, betas=betas)
        d_opt = optim.Adam(learning_rate=self.ganomaly_discriminator_lr, betas=betas)
        w_adv, w_con, w_enc = self.adv_weight, self.con_weight, self.enc_weight

        # D: real -> 0.9 (label smoothing), reconstruction (detached) -> 0.0
        def d_loss_fn(disc_model, real):
            x_hat = mx.stop_gradient(dec(enc(real)))
            loss_real = _bce(disc_model(real), mx.full(real.shape[0], 0.9))
            loss_fake = _bce(disc_model(x_hat), mx.zeros((real.shape[0],)))
            return 0.5 * (loss_real + loss_fake)

        # G: feature-matching adversarial + L1 contextual + L2 encoder loss.
        def g_loss_fn(gen_model, real):
            z = gen_model.encoder(real)
            x_hat = gen_model.decoder(z)
            z_hat = gen_model.encoder2(x_hat)
            l_con = mx.mean(mx.abs(real - x_hat))
            l_enc = mx.mean(mx.square(z - z_hat))
            if use_adv:
                f_real = mx.stop_gradient(disc.features(real))
                f_fake = disc.features(x_hat)
                l_adv = mx.mean(mx.square(f_fake - f_real))
            else:
                l_adv = mx.zeros(())
            return w_adv * l_adv + w_con * l_con + w_enc * l_enc

        d_grad = nn.value_and_grad(disc, d_loss_fn)
        g_grad = nn.value_and_grad(gen, g_loss_fn)

        n = len(normal_data)
        bs = min(self.ganomaly_batch_size, n)
        batches = max(1, n // bs)
        for epoch in range(self.gan_epochs):
            perm = np.random.permutation(n)
            disc.train()
            self.model.train()
            d_acc, g_acc = 0.0, 0.0
            for bi in range(batches):
                idx = perm[bi * bs : (bi + 1) * bs]
                real = mx.array(normal_data[idx])

                if use_adv:
                    ld, gd = d_grad(disc, real)
                    d_opt.update(disc, gd)
                    mx.eval(disc.parameters(), d_opt.state, ld)
                    d_acc += ld.item()

                lg, gg = g_grad(gen, real)
                g_opt.update(gen, gg)
                mx.eval(self.model.parameters(), g_opt.state, lg)
                g_acc += lg.item()

            if epoch % 20 == 0 or epoch == self.gan_epochs - 1:
                print(
                    f"      epoch {epoch}/{self.gan_epochs} - "
                    f"D {d_acc / batches:.4f}, G {g_acc / batches:.4f}"
                )
            mx.clear_cache()

    def predict(self, data):
        """GANomaly inference. Returns the (N,S,F) reconstruction and (N,3)
        trading probs as before, plus the paper's latent anomaly score
        ``anomaly_score`` (N,) = mean|z - ẑ|, which the strategy uses in place
        of the input-reconstruction MSE."""
        if self.model is None:
            self.model = self.load()
        if self.model is None:
            log.error("    ERR: no model for predictions")
            n = np.shape(data)[0]
            return {
                "reconstruction": np.zeros((n, self.seq_len, self.num_features), np.float32),
                "trading": np.zeros((n, 3), np.float32),
                "anomaly_score": np.zeros((n,), np.float32),
            }

        if self.dataframeUtils.is_dataframe(data):
            df_tensor = self.dataframeUtils.df_to_tensor(data, self.seq_len)
        else:
            df_tensor = data
        df_tensor = np.asarray(df_tensor, dtype=np.float32)

        self.model.eval()
        n = df_tensor.shape[0]
        bs = max(1, int(getattr(self, "batch_size", 256) or 256))
        rec_parts, trd_parts, score_parts = [], [], []
        for start in range(0, n, bs):
            xb = mx.array(df_tensor[start : start + bs])
            z = self.model.encoder(xb)
            rec_b = self.model.decoder(z)
            z_hat = self.model.encoder2(rec_b)
            trd_b = self.model.classifier_head(z)
            score_b = mx.mean(mx.abs(z - z_hat), axis=-1)
            mx.eval(rec_b, trd_b, score_b)
            rec_parts.append(np.array(rec_b, dtype=np.float32))
            trd_parts.append(np.array(trd_b, dtype=np.float32))
            score_parts.append(np.array(score_b, dtype=np.float32))
            mx.clear_cache()

        if rec_parts:
            rec = np.concatenate(rec_parts, axis=0)
            trd = np.concatenate(trd_parts, axis=0)
            score = np.concatenate(score_parts, axis=0)
        else:
            rec = np.zeros((0, self.seq_len, self.num_features), np.float32)
            trd = np.zeros((0, 3), np.float32)
            score = np.zeros((0,), np.float32)

        return {"reconstruction": rec, "trading": trd, "anomaly_score": score}


class GANomalyClassifierMLX_Attention(GANomalyDetectorMLXBase):
    num_heads = 4
    ff_dim = 128
    attn_dropout = 0.2

    def _build_components(self, seq_len, num_features, latent_size):
        # MLX MultiHeadAttention requires dims divisible by num_heads.
        nh = self.num_heads
        while num_features % nh != 0 and nh > 1:
            nh -= 1
        enc = _AttnEncoder(seq_len, num_features, latent_size, nh, self.ff_dim, self.attn_dropout)
        dec = _AttnDecoder(seq_len, num_features, latent_size, nh, self.ff_dim, self.attn_dropout)
        disc = _AttnDiscriminator(seq_len, num_features, nh, self.ff_dim, self.attn_dropout)
        return enc, dec, disc


class GANomalyClassifierMLX_Transformer(GANomalyDetectorMLXBase):
    dff = 256
    num_layers = 4
    trans_dropout = 0.1

    def _build_components(self, seq_len, num_features, latent_size):
        t_size = max(32, nearest_power_of_2(num_features // 2) + 1)
        num_heads = max(1, min(8, t_size // 8))
        # Ensure t_size divisible by num_heads (MLX MHA requirement)
        while t_size % num_heads != 0:
            num_heads -= 1
        enc = _TransformerEncoder(
            seq_len,
            num_features,
            latent_size,
            t_size,
            num_heads,
            self.dff,
            self.num_layers,
            self.trans_dropout,
        )
        dec = _TransformerDecoder(
            seq_len,
            num_features,
            latent_size,
            t_size,
            num_heads,
            self.dff,
            self.num_layers,
            self.trans_dropout,
        )
        disc = _TransformerDiscriminator(
            seq_len,
            num_features,
            t_size,
            num_heads,
            self.dff,
            self.num_layers,
            self.trans_dropout,
        )
        return enc, dec, disc


# =======================================================================
# Factory + enum
# =======================================================================


class ClassifierTypeMLX(Enum):
    Dense = GANomalyClassifierMLX_Dense
    LSTM = GANomalyClassifierMLX_LSTM
    Attention = GANomalyClassifierMLX_Attention
    Transformer = GANomalyClassifierMLX_Transformer


def create_classifier_mlx(classifier_type, pair, nfeatures, seq_len, tag=""):
    """Create an MLX GANomaly classifier of the specified type."""
    classifier_name = str(classifier_type).split(".")[-1]
    classifier = classifier_type.value(pair, seq_len, nfeatures, tag=tag)
    return classifier, classifier_name

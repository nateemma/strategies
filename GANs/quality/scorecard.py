"""Comparative scorecard across every registered (GANType, backend) pair.

Answers the question the existing suite cannot: not "did this variant clear a
threshold" but "which variants produce good output, and where does each fail".

Design notes (spec 2026-08-26-gan-parity-design.md):
  * BOTH axes in one table. Fidelity alone is the mistake this codebase already
    made once -- TabDDPM fidelity was fixed and augmentation still matched
    no-GAN (GAN_TODO §5).
  * NEVER raises. A variant that crashes, produces nothing, or emits NaN becomes
    a ROW recording that, because a table missing its broken entries is exactly
    the table that hides the problem.
  * TF and MLX are peers -- every type with both backends yields both rows.

Run:
    PYTHONPATH=.:user_data/strategies .venv/bin/python \
        -m GANs.quality.scorecard --out docs/GAN_SCORECARD.md
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from GANs.GANType import GANType
from GANs.quality.manifold import clip_band_fraction, nn_distance_ratio
from GANs.quality.utility_probe import delta_val_mcc

N_CLASSES = 3


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
def make_fixture(n: int = 600, n_features: int = 8, seed: int = 42):
    """Class-discriminative 2-D fixture + one-hot labels.

    Deliberately learnable: if the fixture had no class structure the utility
    probe could not distinguish a good GAN from a bad one.
    """
    rng = np.random.default_rng(seed)
    means = np.array([
        [0.6, -0.5, 0.4, -0.3, 0.5, -0.4, 0.3, -0.5],
        [-0.5, 0.6, -0.4, 0.5, -0.3, 0.4, -0.5, 0.6],
        [0.0, 0.0, 0.6, -0.6, 0.0, 0.0, 0.6, -0.6],
    ], dtype="float32")[:, :n_features]
    idx = rng.integers(0, N_CLASSES, n)
    x = (means[idx] + rng.normal(scale=0.45, size=(n, n_features))).astype("float32")
    y = np.zeros((n, N_CLASSES), dtype="float32")
    y[np.arange(n), idx] = 1.0
    return x, y


# ---------------------------------------------------------------------------
# Per-type adapters: shape of fit inputs + how generate() is conditioned
# ---------------------------------------------------------------------------
@dataclass
class Adapter:
    to_fit: Callable[[np.ndarray, np.ndarray], tuple]
    generate: Callable[[Any, int, np.ndarray], Any]
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)


def _one_hot_gen(iface, n, y):
    oh = np.zeros((n, N_CLASSES), dtype="float32")
    oh[:, 1] = 1.0
    return iface.generate(n, one_hot=oh), oh


def _task_gen(iface, n, y):
    out = iface.generate(n, task_labels={"trading": y[:n]})
    return out, y[:n]


def _class_label_gen(iface, n, y):
    oh = np.zeros((n, N_CLASSES), dtype="float32")
    oh[:, 1] = 1.0
    return iface.generate(n, class_label=1), oh


def _as_df(x, y):
    import pandas as pd
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    return df, y


def _as_df_mt(x, y):
    """MT CTAB takes a DataFrame but REQUIRES dict labels, unlike single-task."""
    df, _ = _as_df(x, y)
    return df, {"trading": y}


def _as_3d(x, y):
    return x[:, None, :], y


def _as_mt(x, y):
    return x[:, None, :], {"trading": y}


ADAPTERS: Dict[str, Adapter] = {
    "WGAN":        Adapter(lambda x, y: (x, y), _one_hot_gen, {"epochs": 8}),
    "TAB_DDPM":    Adapter(lambda x, y: (x, y), _one_hot_gen,
                           {"epochs": 15, "d_model": 32, "d_layers": (32, 32),
                            "num_timesteps": 100, "num_sample_steps": 20,
                            "batch_size": 128, "verbose": False}),
    "CGAN":        Adapter(_as_3d, _one_hot_gen, {"epochs": 8}),
    "MT_WGAN":     Adapter(_as_mt, _task_gen, {"epochs": 8}),
    "MT_DDPM":     Adapter(_as_mt, _task_gen, {"epochs": 15, "verbose": False}),
    "CTAB_GAN":    Adapter(_as_df, _class_label_gen, {"epochs": 8}),
    "MT_CTAB_GAN": Adapter(_as_df_mt, _task_gen, {"epochs": 8}),
}


def _to_2d(a: Any) -> Optional[np.ndarray]:
    """Coerce whatever a backend returned into (n, F), or None."""
    if a is None:
        return None
    if isinstance(a, tuple):
        a = a[0]
    try:
        import pandas as pd
        if isinstance(a, pd.DataFrame):
            a = a.to_numpy()
    except Exception:
        pass
    a = np.asarray(a, dtype=float)
    if a.ndim == 0 or a.size == 0:
        return None
    return a.reshape(a.shape[0], -1) if a.ndim > 2 else a


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def score_one(gan_type: GANType, prefer_mlx: bool, x, y, *, seed=42) -> Dict[str, Any]:
    import GANs.backends  # noqa: F401 -- populates the registry
    from GANs.GANBackend import resolve_backend

    row: Dict[str, Any] = {
        "type": gan_type.name, "backend": "MLX" if prefer_mlx else "TF",
        "available": False, "fit_s": None, "n_synth": 0,
        "worst_dmu": None, "sigma_ratio": None, "max_dcorr": None,
        "clip_band": None, "nn_ratio": None, "delta_mcc": None,
        "status": "", "note": "",
    }
    try:
        cls = resolve_backend(gan_type, prefer_mlx=prefer_mlx)
    except Exception as e:
        row["status"] = "unavailable"
        row["note"] = str(e)[:60]
        return row
    if (cls.__name__.endswith("MLXBackend")) != prefer_mlx:
        row["status"] = "no-such-backend"
        row["note"] = f"resolved to {cls.__name__}"
        return row
    row["available"] = True

    ad = ADAPTERS.get(gan_type.name)
    if ad is None:
        row["status"] = "no-adapter"
        return row

    try:
        data, labels = ad.to_fit(x, y)
        iface = cls()
        t0 = time.time()
        iface.fit(data, labels, **ad.fit_kwargs)
        row["fit_s"] = round(time.time() - t0, 1)
    except Exception as e:
        row["status"] = "fit-failed"
        row["note"] = f"{type(e).__name__}: {str(e)[:50]}"
        return row

    try:
        raw, synth_y = ad.generate(iface, min(300, len(x)), y)
        synth = _to_2d(raw)
    except Exception as e:
        row["status"] = "generate-failed"
        row["note"] = f"{type(e).__name__}: {str(e)[:50]}"
        return row

    if synth is None:
        row["status"] = "no-output"
        return row
    row["n_synth"] = int(len(synth))
    if not np.isfinite(synth).all():
        row["status"] = "non-finite"
        row["note"] = f"{(~np.isfinite(synth)).mean():.1%} bad"
        return row

    real2d = x.reshape(len(x), -1)
    if synth.shape[1] != real2d.shape[1]:
        row["status"] = "shape-mismatch"
        row["note"] = f"{synth.shape[1]} vs {real2d.shape[1]}"
        return row

    # fidelity
    mu_r, sd_r = real2d.mean(0), real2d.std(0)
    sd_r = np.where(sd_r > 1e-12, sd_r, 1.0)
    row["worst_dmu"] = round(float(np.abs((synth.mean(0) - mu_r) / sd_r).max()), 3)
    row["sigma_ratio"] = round(float(np.median(synth.std(0) / sd_r)), 3)
    try:
        dc = np.corrcoef(synth, rowvar=False) - np.corrcoef(real2d, rowvar=False)
        row["max_dcorr"] = round(float(np.nanmax(np.abs(dc))), 3)
    except Exception:
        pass

    # manifold
    row["clip_band"] = round(clip_band_fraction((synth - mu_r) / sd_r, 4.0), 4)
    row["nn_ratio"] = round(nn_distance_ratio(real2d, synth), 3)

    # utility
    u = delta_val_mcc(real2d, y, synth, synth_y, seed=seed)
    row["delta_mcc"] = None if u["delta"] is None else round(u["delta"], 4)
    row["status"] = "ok" if u["delta"] is not None else "no-utility"
    row["note"] = row["note"] or u.get("reason", "")
    return row


def build_scorecard(x=None, y=None, *, types=None, seed=42) -> List[Dict[str, Any]]:
    if x is None:
        x, y = make_fixture(seed=seed)
    wanted = types or [t for t in GANType if t.name not in ("NONE", "BOTH")]
    rows = []
    for t in wanted:
        for prefer in (False, True):          # TF first: it is a peer, not an afterthought
            try:
                rows.append(score_one(t, prefer, x, y, seed=seed))
            except Exception as e:            # belt and braces -- never abort the sweep
                rows.append({"type": t.name, "backend": "MLX" if prefer else "TF",
                             "status": "driver-error", "note": f"{type(e).__name__}",
                             "available": False})
                traceback.print_exc()
    return rows


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    cols = [("type", "type"), ("backend", "be"), ("status", "status"),
            ("fit_s", "fit s"), ("n_synth", "n"), ("worst_dmu", "worst Δμ/σ"),
            ("sigma_ratio", "σ_syn/σ_real"), ("max_dcorr", "max Δcorr"),
            ("clip_band", "clip band"), ("nn_ratio", "NN ratio"),
            ("delta_mcc", "Δval_mcc"), ("note", "note")]
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(
            "" if r.get(k) is None else str(r.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rows = build_scorecard()
    md = render_markdown(rows)
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write("# GAN scorecard\n\nGenerated by `GANs/quality/scorecard.py`.\n\n")
            f.write(md + "\n")

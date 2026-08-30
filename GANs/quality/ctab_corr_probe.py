"""CTAB joint-structure probe — WHERE does TF diverge from MLX?

The scorecard localized the remaining CTAB gap to the generator: CTAB-MLX
reproduces the real correlation structure (mean|off-diag corr| 0.265 vs real
0.306, corr-of-corr +0.976) while CTAB-TF emits nearly INDEPENDENT features
(0.059).  Eight hypotheses formed by reading code were refuted, so the D2
playbook applies: instrument, don't hypothesise.

This fits ONE (backend, epochs) configuration and reports the correlation
structure of its output, so a budget ladder shows whether TF starts
decorrelated and stays flat (never learns) or starts structured and actively
decorrelates (training destroys it).  That is the fork the fix depends on.

ONE PROCESS PER CONFIGURATION — MLX shares a global RNG, so configs run in a
single process each sit at a different point in the stream and are not
comparable (method note recorded in docs/GAN_SCORECARD.md).

Usage (from repo root):
    PYTHONPATH=.:user_data/strategies .venv/bin/python \
        -m GANs.quality.ctab_corr_probe --backend tf --epochs 10
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import numpy as np


def _offdiag(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


def corr_stats(real: np.ndarray, synth: np.ndarray) -> dict:
    """Correlation structure of synth, measured against real."""
    cr = np.corrcoef(real, rowvar=False)
    cs = np.corrcoef(synth, rowvar=False)
    cr = np.nan_to_num(cr)
    cs = np.nan_to_num(cs)
    vr, vs = _offdiag(cr), _offdiag(cs)
    # corr-of-corr: does synth rank the same pairs as correlated?
    if vs.std() > 1e-9 and vr.std() > 1e-9:
        coc = float(np.corrcoef(vr, vs)[0, 1])
    else:
        coc = float("nan")
    return {
        "mean_abs_corr": float(np.abs(vs).mean()),
        "real_mean_abs_corr": float(np.abs(vr).mean()),
        "corr_of_corr": coc,
        "mean_dcorr": float(np.abs(vs - vr).mean()),
        "max_dcorr": float(np.abs(vs - vr).max()),
    }


def run(backend: str, epochs: int, seed: int, n_synth: int,
        pac: int | None = None, corr_weight: float | None = None,
        gumbel_noise: bool | None = None,
        corr_warmup: int | None = None,
        d_clip: str | None = None, g_clip: str | None = None,
        gen_norm: str | None = None, gen_layers: str | None = None,
        disc_layers: str | None = None,
        gan_type: str = "CTAB_GAN") -> dict:
    import GANs.backends  # noqa: F401 -- populates the registry
    from GANs.GANBackend import resolve_backend
    from GANs.GANType import GANType
    from GANs.quality.real_fixture import build_real_fixture
    from GANs.quality.scorecard import ADAPTERS, _to_2d

    prefer_mlx = backend == "mlx"
    x, y, _names = build_real_fixture(seed=seed)
    cls = resolve_backend(getattr(GANType, gan_type), prefer_mlx=prefer_mlx)
    if cls.__name__.endswith("MLXBackend") != prefer_mlx:
        raise SystemExit(f"asked for {backend}, resolved {cls.__name__}")

    ad = ADAPTERS[gan_type]
    data, labels = ad.to_fit(x, y)
    iface = cls()
    fit_kw = {"epochs": epochs}
    if pac is not None:
        # TF only -- the MLX CTAB backend has no packing at all (grep: 0 hits).
        fit_kw["pac"] = pac
    if corr_weight is not None:
        # TF only -- explicit off-diagonal correlation matching, the candidate
        # fix for the discriminator's joint blindness (docs/GAN_SCORECARD.md).
        fit_kw["corr_loss_weight"] = corr_weight
    if gumbel_noise is not None:
        # TF only -- MLX has no Gumbel noise at all (it uses a plain
        # temperature-sharpened softmax), which is the parity question here.
        fit_kw["gumbel_noise"] = gumbel_noise
    if corr_warmup is not None:
        fit_kw["corr_loss_warmup_epochs"] = corr_warmup
    for key, val in (("d_grad_clip_norm", d_clip), ("g_grad_clip_norm", g_clip)):
        if val is not None:
            fit_kw[key] = None if val == "none" else float(val)
    if gen_norm is not None:
        fit_kw["generator_norm"] = gen_norm
    if gen_layers is not None:
        fit_kw["generator_layers"] = [int(v) for v in gen_layers.split(",")]
    if disc_layers is not None:
        fit_kw["discriminator_layers"] = [int(v) for v in disc_layers.split(",")]
    t0 = time.time()
    iface.fit(data, labels, **fit_kw)
    fit_s = time.time() - t0

    raw, _ = ad.generate(iface, n_synth, y)
    synth = _to_2d(raw)
    real2d = x.reshape(len(x), -1)

    out = corr_stats(real2d, synth)
    sd_r = real2d.std(0)
    sd_r = np.where(sd_r > 1e-12, sd_r, 1.0)
    out.update({
        "backend": backend,
        "epochs": epochs,
        "seed": seed,
        "pac": pac,
        "gan_type": gan_type,
        "corr_weight": corr_weight,
        "gumbel_noise": gumbel_noise,
        "corr_warmup": corr_warmup,
        "d_clip": d_clip,
        "g_clip": g_clip,
        "gen_norm": gen_norm,
        "gen_layers": gen_layers,
        "disc_layers": disc_layers,
        "n_synth": int(len(synth)),
        "fit_s": round(fit_s, 1),
        "sigma_ratio": float(np.median(synth.std(0) / sd_r)),
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("tf", "mlx"), required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pac", type=int, default=None,
                    help="TF only: PacGAN packing factor (default: model's 10).")
    ap.add_argument("--corr-weight", type=float, default=None,
                    help="TF only: weight on the off-diagonal correlation "
                         "matching term (default: the model's 0.0 = off).")
    ap.add_argument("--no-gumbel-noise", dest="gumbel_noise",
                    action="store_false", default=None,
                    help="TF only: drop the independent per-column Gumbel "
                         "draw on mode selection, matching MLX.")
    ap.add_argument("--corr-warmup", type=int, default=None,
                    help="TF only: ramp the correlation weight from 0 to full "
                         "over N epochs (default: no ramp).")
    ap.add_argument("--d-clip", default=None,
                    help="TF only: D gradient clip norm, or 'none' to disable.")
    ap.add_argument("--g-clip", default=None,
                    help="TF only: G gradient clip norm, or 'none' to disable.")
    ap.add_argument("--gan-type", choices=("CTAB_GAN", "MT_CTAB_GAN"),
                    default="CTAB_GAN")
    ap.add_argument("--gen-norm", choices=("batch", "layer", "none"),
                    default=None, help="TF only: generator normalisation.")
    ap.add_argument("--gen-layers", default=None,
                    help="TF only: comma-separated generator widths, "
                         "e.g. 256,512,256 to match MLX.")
    ap.add_argument("--disc-layers", default=None,
                    help="TF only: comma-separated discriminator widths, "
                         "e.g. 256,512 to match MLX.")
    ap.add_argument("--n-synth", type=int, default=2400,
                    help="Large enough for a stable 24x24 correlation estimate; "
                         "the scorecard's 300 is too few for this metric.")
    a = ap.parse_args()
    r = run(a.backend, a.epochs, a.seed, a.n_synth, a.pac, a.corr_weight,
            a.gumbel_noise, a.corr_warmup, a.d_clip, a.g_clip,
            a.gen_norm, a.gen_layers, a.disc_layers, a.gan_type)
    print(
        "RESULT "
        f"backend={r['backend']} type={r['gan_type']} "
        f"epochs={r['epochs']} seed={r['seed']} "
        f"pac={r['pac']} cw={r['corr_weight']} gn={r['gumbel_noise']} wu={r['corr_warmup']} dc={r['d_clip']} gc={r['g_clip']} gnorm={r['gen_norm']} gl={r['gen_layers']} dl={r['disc_layers']} "
        f"mean_abs_corr={r['mean_abs_corr']:.3f} "
        f"real={r['real_mean_abs_corr']:.3f} "
        f"corr_of_corr={r['corr_of_corr']:+.3f} "
        f"mean_dcorr={r['mean_dcorr']:.3f} max_dcorr={r['max_dcorr']:.3f} "
        f"sigma={r['sigma_ratio']:.3f} fit_s={r['fit_s']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

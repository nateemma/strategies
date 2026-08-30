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
        gan_type: str = "CTAB_GAN",
        extra: dict | None = None,
        imbalance: float | None = None) -> dict:
    import GANs.backends  # noqa: F401 -- populates the registry
    from GANs.GANBackend import resolve_backend
    from GANs.GANType import GANType
    from GANs.quality.real_fixture import build_real_fixture
    from GANs.quality.scorecard import ADAPTERS, _to_2d

    prefer_mlx = backend == "mlx"
    x, y, _names = build_real_fixture(seed=seed)
    if imbalance is not None:
        # The stock fixture is PERFECTLY balanced (1333/1333/1334), so both
        # samplers produce equivalent batches on it and a class-balance effect
        # is undetectable.  Production trading labels are heavily skewed
        # (signal frequency ~5%), so subsample the minority classes to make the
        # comparison meaningful.
        lab = y.argmax(1) if y.ndim > 1 else y
        rng = np.random.default_rng(seed)
        keep = []
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            n = len(idx) if c == 0 else max(1, int(round(len(idx) * imbalance)))
            keep.append(rng.choice(idx, size=n, replace=False))
        keep = np.sort(np.concatenate(keep))
        x, y = x[keep], y[keep]
        counts = np.bincount((y.argmax(1) if y.ndim > 1 else y).astype(int))
        # The MT adapter generates with y[:n], so n cannot exceed the rows left.
        n_synth = min(n_synth, len(y))
        print(
            f"    imbalanced fixture: class counts {counts.tolist()}, "
            f"n_synth={n_synth}"
        )
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
    if extra:
        fit_kw.update(extra)
    t0 = time.time()
    iface.fit(data, labels, **fit_kw)
    fit_s = time.time() - t0

    raw, _ = ad.generate(iface, n_synth, y)
    synth = _to_2d(raw)
    real2d = x.reshape(len(x), -1)

    # Clip-saturation guard.  A generator whose samples pile at its own
    # z-score clip produces sigma ~= clip by ARITHMETIC, not by training, and
    # every fidelity number from it is meaningless.  Measured on the DDPM
    # variants trained on this fixture: 99.8% of values at the boundary
    # against a production-trained model's 0.2%.  Reported so the harness
    # says so instead of silently emitting garbage.
    clip_sat = None
    try:
        _m = getattr(iface, "_model", None)
        _clip = getattr(type(_m), "_ZSCORE_CLIP", None)
        _mu = getattr(_m, "feature_mean", None)
        _sd = getattr(_m, "feature_std", None)
        if _clip and _mu is not None and _sd is not None:
            _mu = np.asarray(_mu).reshape(-1)
            _sd = np.asarray(_sd).reshape(-1)
            _sd = np.where(_sd > 1e-12, _sd, 1.0)
            if synth.shape[1] == _mu.shape[0]:
                _az = np.abs((synth - _mu) / _sd)
                clip_sat = float(np.mean(_az >= 0.95 * _clip))
    except Exception:
        clip_sat = None
    if clip_sat is not None and clip_sat > 0.20:
        print(
            f"    *** WARNING: {100*clip_sat:.1f}% of generated values sit at "
            f"the model's +/-{_clip} z-score clip. Samples are pinned at the "
            f"boundary, so sigma is arithmetic and EVERY fidelity number below "
            f"is meaningless. Do not report these as model defects.",
            flush=True,
        )

    out = corr_stats(real2d, synth)
    sd_r = real2d.std(0)
    sd_r = np.where(sd_r > 1e-12, sd_r, 1.0)
    out.update({
        "backend": backend,
        "epochs": epochs,
        "seed": seed,
        "pac": pac,
        "gan_type": gan_type,
        "imbalance": imbalance,
        "extra": extra,
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
        "clip_sat": clip_sat,
    })
    return out


def _parse_set(items) -> dict | None:
    """Parse repeated --set KEY=VALUE into a ctor-kwargs dict."""
    if not items:
        return None
    import ast
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--set expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v  # bare string
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
    ap.add_argument("--imbalance", type=float, default=None,
                    help="Subsample non-zero classes to this fraction, e.g. "
                         "0.15, to make a class-balance effect detectable.")
    ap.add_argument("--set", dest="extra", action="append", default=None,
                    metavar="KEY=VALUE",
                    help="Pass an arbitrary ctor kwarg through, repeatable. "
                         "Value is parsed as a Python literal "
                         "(e.g. --set generator_loss_weight=0.0 "
                         "--set restore_best_weights=False).")
    ap.add_argument("--gan-type", choices=("CTAB_GAN", "MT_CTAB_GAN", "CGAN", "WGAN",
                             "MT_WGAN", "TAB_DDPM", "MT_DDPM"),
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
            a.gen_norm, a.gen_layers, a.disc_layers, a.gan_type,
            _parse_set(a.extra), a.imbalance)
    print(
        "RESULT "
        f"backend={r['backend']} type={r['gan_type']} "
        f"epochs={r['epochs']} seed={r['seed']} "
        f"pac={r['pac']} cw={r['corr_weight']} gn={r['gumbel_noise']} wu={r['corr_warmup']} dc={r['d_clip']} gc={r['g_clip']} gnorm={r['gen_norm']} gl={r['gen_layers']} dl={r['disc_layers']} set={r['extra']} imb={r['imbalance']} "
        f"mean_abs_corr={r['mean_abs_corr']:.3f} "
        f"real={r['real_mean_abs_corr']:.3f} "
        f"corr_of_corr={r['corr_of_corr']:+.3f} "
        f"mean_dcorr={r['mean_dcorr']:.3f} max_dcorr={r['max_dcorr']:.3f} "
        f"sigma={r['sigma_ratio']:.3f} "
        + (f"clipsat={100*r['clip_sat']:.1f}% " if r.get("clip_sat") is not None else "")
        + f"fit_s={r['fit_s']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

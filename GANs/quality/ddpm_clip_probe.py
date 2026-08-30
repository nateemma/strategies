"""DDPM clip-saturation probe — is sigma ~4 a training failure or arithmetic?

Usage (from repo root):
    PYTHONPATH=.:user_data/strategies .venv/bin/python \
        -m GANs.quality.ddpm_clip_probe

Confirm the mechanism: do TabDDPM's samples SATURATE the +/-4 sigma clip?

GAN_TODO section 2 describes exactly this for MT_DDPM ("z-scores saturate the
+/-4 sigma clip -> synth sigma ~4-5x real"). If TAB_DDPM's generated samples sit
at the clip boundary in the model's OWN z-space, sigma ~4 is arithmetic, not a
training failure -- and no amount of capacity or data would fix it.
"""
import numpy as np, GANs.backends
from GANs.GANBackend import resolve_backend
from GANs.GANType import GANType
from GANs.quality.real_fixture import build_real_fixture
from GANs.quality.scorecard import ADAPTERS, _to_2d

x, y, _ = build_real_fixture(seed=42)
ad = ADAPTERS["TAB_DDPM"]
data, labels = ad.to_fit(x, y)
ifc = resolve_backend(GANType.TAB_DDPM, prefer_mlx=True)()
ifc.fit(data, labels, epochs=300, d_model=256, d_layers=[256, 256],
        num_timesteps=1000, num_sample_steps=50, batch_size=1024, verbose=False)
raw, _ = ad.generate(ifc, len(y), y)
synth = _to_2d(raw); real = x.reshape(len(x), -1)
m = ifc._model
CLIP = getattr(type(m), "_ZSCORE_CLIP", 4.0)
mu = getattr(m, "feature_mean", None); sd = getattr(m, "feature_std", None)
if mu is None:
    mu, sd = real.mean(0), real.std(0)
mu = np.asarray(mu).reshape(-1); sd = np.asarray(sd).reshape(-1)
sd = np.where(sd > 1e-12, sd, 1.0)

zs = (synth - mu) / sd
zr = (real - mu) / sd
print(f"\nmodel _ZSCORE_CLIP = +/-{CLIP}\n")
print(f"{'':14} {'std':>8} {'|z| mean':>9} {'|z| p99':>9} {'|z|>=0.95*clip':>16}")
for tag, z in (("REAL  (z-space)", zr), ("SYNTH (z-space)", zs)):
    az = np.abs(z)
    print(f"{tag:14} {z.std():8.3f} {az.mean():9.3f} {np.percentile(az,99):9.3f} "
          f"{100*np.mean(az >= 0.95*CLIP):15.1f}%")
print(f"\nsynth std / real std (median per-feature): "
      f"{np.median(synth.std(0)/np.where(real.std(0)>1e-12, real.std(0),1.0)):.3f}")
frac = float(np.mean(np.abs(zs) >= 0.95*CLIP))
print("\nVERDICT:", "CLIP SATURATION CONFIRMED — samples pile at the boundary"
      if frac > 0.20 else "NOT clip saturation — samples are not at the boundary")

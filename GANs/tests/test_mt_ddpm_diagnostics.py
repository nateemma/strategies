"""Test diagnostic output for MT_DDPM."""

import io
import contextlib
import numpy as np

from GANs.df_mt_ddpm_mlx import MTDDPMMLX


def test_diagnostics_print_and_dont_crash():
    rng = np.random.default_rng(5)
    seq_len, F, C = 8, 6, 3
    N = 128
    # Structured: sinusoidal across time gives non-trivial autocorrelation.
    t_grid = np.linspace(0, 2 * np.pi, seq_len, dtype=np.float32)
    data = np.sin(t_grid)[None, :, None] + 0.1 * rng.normal(0, 1, size=(N, seq_len, F)).astype(np.float32)
    labels = {"trading": np.eye(C, dtype=np.float32)[rng.integers(0, C, size=N)]}

    model = MTDDPMMLX(
        seq_len=seq_len,
        num_features=F,
        task_label_dims={"trading": C},
        d_model=32,
        d_layers=2,
        num_timesteps=50,
        num_sample_steps=10,
        epochs=2,
        batch_size=32,
        verbose=True,
    )
    model.fit(data, labels)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.print_diagnostics(data, labels)

    output = buf.getvalue()
    assert "marginal" in output.lower() or "mean" in output.lower()
    assert "autocorr" in output.lower() or "lag" in output.lower()

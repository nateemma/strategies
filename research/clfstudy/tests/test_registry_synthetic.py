# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""Liveness test for every `ClassifierTypeMLX` member (T022).

Builds each member through `registry.build`, trains 2 epochs on a tiny synthetic
tensor with a weak 3-class signal, predicts on a held-out slice, and asserts the
output is a well-formed softmax with a non-zero trainable parameter count.

New in feature 018 (T023): RIDGE and ROCKET get an exact parameter-count check.

New in feature 018 (T034): the five pure single-family types (`LSTM_FAM`,
`GRU_FAM`, `TCN_FAM`, `TRANSFORMER_FAM`, `MLP_FAM`) get a size-axis
parameter-count-differs check and a structural-axis (bidirectional, depth,
pool, residual, dilation_base, pos) build-and-train-1-epoch liveness check
(research R7).

New in feature 018 (T044-T048): `TSMIXER`, `ITRANSFORMER`, `XLSTM`,
`TRANSFORMER_MODERN`, `PATCHTST` (research R13 / Part B refs 5, 6, 7, 8, 12,
13, 20). `PATCHTST` needs `seq_len=64` (production `seq_len=16` is below its
`4*patch_len` minimum) -- handled via `SEQ_LEN_OVERRIDE` in the main
build/train/predict parametrisation, plus a dedicated test that it raises
`ValueError` at `seq_len=16`.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from NNNC.NNNClassifierMLX import ClassifierTypeMLX, _RoPEMultiHeadAttention
from research.clfstudy.registry import ExperimentConfig, build, param_count

SEQ_LEN = 16
N_FEATURES = 24
N_CLASSES = 3
N_ROWS = 400
N_VAL = 100

#: members that need a different `seq_len` than the rest of the parametrised
#: build/train/predict sweep (T048: PATCHTST's `4*patch_len` minimum is above
#: the production seq_len=16 used everywhere else).
SEQ_LEN_OVERRIDE: dict[str, int] = {"PATCHTST": 64}


def _synthetic_data(seed: int = 0, seq_len: int = SEQ_LEN, n_rows: int = N_ROWS):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, seq_len, N_FEATURES)).astype(np.float32)

    # Weak signal: label = argmax of 3 random linear projections of the last timestep.
    proj = rng.normal(size=(N_FEATURES, N_CLASSES)).astype(np.float32)
    logits = X[:, -1, :] @ proj
    y = np.argmax(logits, axis=1)
    y_onehot = np.eye(N_CLASSES, dtype=np.float32)[y]

    X_train, X_val = X[:-N_VAL], X[-N_VAL:]
    y_train, y_val = y_onehot[:-N_VAL], y_onehot[-N_VAL:]
    return X_train, X_val, y_train, y_val


CLASS_WEIGHTS = [1.0, 1.0, 1.0]

# Members that fail for reasons unrelated to T014/T023 (registry + RIDGE/ROCKET).
# Reason strings are the observed failure; re-check after the family/backbone that
# owns each member is next touched (contracts/registry.md task list).
XFAIL_REASONS: dict[str, str] = {}


def _member_param(member: ClassifierTypeMLX):
    reason = XFAIL_REASONS.get(member.name)
    if reason:
        return pytest.param(member, marks=pytest.mark.xfail(reason=reason, strict=False))
    return member


@pytest.mark.parametrize(
    "member", [_member_param(m) for m in ClassifierTypeMLX], ids=[m.name for m in ClassifierTypeMLX]
)
def test_build_train_predict(member, tmp_path):
    seq_len = SEQ_LEN_OVERRIDE.get(member.name, SEQ_LEN)
    cfg = ExperimentConfig(
        arch=member.name,
        hp={},
        input_set="ind24",
        seq_len=seq_len,
        split_set="v1",
        train={"max_epochs": 2, "patience": 5, "batch_size": 64},
    )
    cfg.validate()

    clf = build(cfg, nf=N_FEATURES, seq_len=seq_len, seed=0, checkpoint_dir=tmp_path)

    X_train, X_val, y_train, y_val = _synthetic_data(seq_len=seq_len)
    clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)

    preds = clf.predict(X_val)

    assert preds.shape == (N_VAL, N_CLASSES)
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-4)

    n_params = param_count(clf)
    assert n_params > 0

    if member.name == "RIDGE":
        assert n_params == seq_len * N_FEATURES * N_CLASSES + N_CLASSES
    elif member.name == "ROCKET":
        assert n_params == 2 * 1000 * N_CLASSES + N_CLASSES


# -----------------------------------------------------------------------
# T034: size-axis and structural-axis liveness for the pure family types
# -----------------------------------------------------------------------

#: family arch -> (hp key, values) for the size axis (research R7).
FAMILY_SIZE_AXES: dict[str, tuple[str, list[int]]] = {
    "LSTM_FAM": ("hidden", [16, 64, 256]),
    "GRU_FAM": ("hidden", [16, 64, 256]),
    "TCN_FAM": ("channels", [16, 64, 256]),
    "TRANSFORMER_FAM": ("d_model", [32, 64, 128]),
    "MLP_FAM": ("width", [64, 256, 1024]),
}


@pytest.mark.parametrize(
    "arch", list(FAMILY_SIZE_AXES), ids=list(FAMILY_SIZE_AXES)
)
def test_family_size_axis_param_counts_differ(arch, tmp_path):
    """R7: trainable parameter counts must differ across each family's size axis."""
    key, values = FAMILY_SIZE_AXES[arch]
    counts = []
    for v in values:
        cfg = ExperimentConfig(
            arch=arch,
            hp={key: v},
            input_set="ind24",
            seq_len=SEQ_LEN,
            split_set="v1",
            train={"max_epochs": 1, "patience": 5, "batch_size": 64},
        )
        cfg.validate()
        clf = build(
            cfg, nf=N_FEATURES, seq_len=SEQ_LEN, seed=0, checkpoint_dir=tmp_path / str(v)
        )

        X_train, X_val, y_train, y_val = _synthetic_data()
        clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)
        counts.append(param_count(clf))

    assert len(set(counts)) == len(counts), f"{arch} param counts did not differ: {counts}"


#: (arch, hp) pairs for the structural axes named in T034 — each must build and
#: train 1 epoch without error.
STRUCTURAL_AXES: list[tuple[str, dict]] = [
    ("LSTM_FAM", {"bidirectional": True}),
    ("LSTM_FAM", {"depth": 2}),
    ("LSTM_FAM", {"pool": "mean"}),
    ("GRU_FAM", {"bidirectional": True}),
    ("GRU_FAM", {"depth": 2}),
    ("GRU_FAM", {"pool": "mean"}),
    ("TCN_FAM", {"depth": 2}),
    ("TCN_FAM", {"residual": True}),
    ("TCN_FAM", {"dilation_base": 2}),
    ("TRANSFORMER_FAM", {"pos": "learned"}),
    ("TRANSFORMER_FAM", {"pos": "none"}),
    ("TRANSFORMER_FAM", {"pos": "rope"}),
    ("MLP_FAM", {"depth": 2}),
]


@pytest.mark.parametrize(
    "arch,hp",
    STRUCTURAL_AXES,
    ids=[f"{a}-{next(iter(hp))}={next(iter(hp.values()))}" for a, hp in STRUCTURAL_AXES],
)
def test_family_structural_axes_build_train(arch, hp, tmp_path):
    """R7: structural-axis hp combinations (bidirectional, depth, pool, residual,
    dilation_base, pos) all build and train 1 epoch on the tiny synthetic tensor."""
    cfg = ExperimentConfig(
        arch=arch,
        hp=hp,
        input_set="ind24",
        seq_len=SEQ_LEN,
        split_set="v1",
        train={"max_epochs": 1, "patience": 5, "batch_size": 64},
    )
    cfg.validate()
    clf = build(cfg, nf=N_FEATURES, seq_len=SEQ_LEN, seed=0, checkpoint_dir=tmp_path)

    X_train, X_val, y_train, y_val = _synthetic_data()
    clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)

    preds = clf.predict(X_val)
    assert preds.shape == (N_VAL, N_CLASSES)
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-4)
    assert param_count(clf) > 0


# -----------------------------------------------------------------------
# _RoPEMultiHeadAttention unit tests (rotary position embedding on the
# per-head q/k projections, used by TRANSFORMER_FAM's pos="rope")
# -----------------------------------------------------------------------


def test_rope_mha_output_shape():
    B, T, d_model, heads = 3, 10, 32, 4
    attn = _RoPEMultiHeadAttention(d_model, heads)
    x = mx.random.normal(shape=(B, T, d_model))
    out = attn(x, x, x)
    assert out.shape == (B, T, d_model)


def test_rope_mha_relative_position_invariance():
    """Attention scores between positions i and j must depend only on (i-j).

    With identity-initialised q/k projections (single head), compute the raw
    score matrix for a window of content and for the same window slid forward
    by one bar (drop the oldest bar, append a new one). Since every window is
    RoPE-encoded from position 0, the (T-1)x(T-1) sub-block of overlapping
    content — original rows/cols [1:], shifted rows/cols [:-1] — must match:
    the same pair of bars sits at the same relative offset in both windows.
    """
    d_model, heads = 4, 1
    attn = _RoPEMultiHeadAttention(d_model, heads)
    identity = mx.eye(d_model)
    zero_bias = mx.zeros((d_model,))
    attn.q_proj.weight = identity
    attn.q_proj.bias = zero_bias
    attn.k_proj.weight = identity
    attn.k_proj.bias = zero_bias

    T = 6
    mx.random.seed(0)
    content = mx.random.normal(shape=(T + 1, d_model))

    def score_matrix(start: int) -> mx.array:
        x = content[start : start + T][None]  # (1, T, d_model)
        q = attn._split_heads(attn.q_proj(x), T)
        k = attn._split_heads(attn.k_proj(x), T)
        q = attn.rope(q)
        k = attn.rope(k)
        return q[0, 0] @ k[0, 0].T  # (T, T), single head

    s_orig = score_matrix(0)
    s_shifted = score_matrix(1)

    sub_orig = s_orig[1:, 1:]
    sub_shifted = s_shifted[:-1, :-1]
    assert mx.allclose(sub_orig, sub_shifted, atol=1e-4).item()


# -----------------------------------------------------------------------
# T044-T048: phase-6 architecture-specific checks
# -----------------------------------------------------------------------


def test_patchtst_raises_at_seq_len_16(tmp_path):
    """PATCHTST's default patch_len=16 needs seq_len >= 4*16=64; production
    seq_len=16 must raise a clear ValueError rather than build a degenerate
    model."""
    cfg = ExperimentConfig(
        arch="PATCHTST",
        hp={},
        input_set="ind24",
        seq_len=SEQ_LEN,
        split_set="v1",
        train={"max_epochs": 1, "patience": 5, "batch_size": 64},
    )
    cfg.validate()
    clf = build(cfg, nf=N_FEATURES, seq_len=SEQ_LEN, seed=0, checkpoint_dir=tmp_path)

    with pytest.raises(ValueError, match="seq_len"):
        clf.create_model(SEQ_LEN, N_FEATURES)


def test_itransformer_exposes_last_attention(tmp_path):
    """ITRANSFORMER must expose `model.last_attention` with shape (F, F) --
    the mean-over-heads/batch attention matrix over feature tokens -- after a
    forward pass, for use as the secondary (attention-based) importance
    signal."""
    seq_len = SEQ_LEN
    cfg = ExperimentConfig(
        arch="ITRANSFORMER",
        hp={},
        input_set="ind24",
        seq_len=seq_len,
        split_set="v1",
        train={"max_epochs": 1, "patience": 5, "batch_size": 64},
    )
    cfg.validate()
    clf = build(cfg, nf=N_FEATURES, seq_len=seq_len, seed=0, checkpoint_dir=tmp_path)

    X_train, X_val, y_train, y_val = _synthetic_data(seq_len=seq_len)
    clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)

    assert clf.model.last_attention is not None
    assert clf.model.last_attention.shape == (N_FEATURES, N_FEATURES)


def test_xlstm_finite_on_large_magnitude_inputs(tmp_path):
    """XLSTM's mLSTM scan (exponential gating + matrix memory) must stay
    finite even when the input magnitude is 50x the normal synthetic scale --
    the log-space stabiliser is the mechanism under test, not just liveness."""
    seq_len = SEQ_LEN
    cfg = ExperimentConfig(
        arch="XLSTM",
        hp={},
        input_set="ind24",
        seq_len=seq_len,
        split_set="v1",
        train={"max_epochs": 1, "patience": 5, "batch_size": 64},
    )
    cfg.validate()
    clf = build(cfg, nf=N_FEATURES, seq_len=seq_len, seed=0, checkpoint_dir=tmp_path)

    X_train, X_val, y_train, y_val = _synthetic_data(seq_len=seq_len)
    clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)

    X_large = (X_val * 50.0).astype(np.float32)
    preds = clf.predict(X_large)
    assert np.isfinite(preds).all()
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-3)


def test_tsmixer_param_counts_differ_across_blocks(tmp_path):
    """R13/T044: TSMixer's parameter count must scale with `blocks` (each
    block adds its own time-mix + feature-mix Linear layers)."""
    counts = []
    for blocks in (1, 3):
        cfg = ExperimentConfig(
            arch="TSMIXER",
            hp={"blocks": blocks},
            input_set="ind24",
            seq_len=SEQ_LEN,
            split_set="v1",
            train={"max_epochs": 1, "patience": 5, "batch_size": 64},
        )
        cfg.validate()
        clf = build(
            cfg, nf=N_FEATURES, seq_len=SEQ_LEN, seed=0, checkpoint_dir=tmp_path / str(blocks)
        )
        X_train, X_val, y_train, y_val = _synthetic_data()
        clf.train(X_train, X_val, y_train, y_val, class_weights=CLASS_WEIGHTS)
        counts.append(param_count(clf))
    assert len(set(counts)) == len(counts), f"TSMIXER param counts did not differ: {counts}"

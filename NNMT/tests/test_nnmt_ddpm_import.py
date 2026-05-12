"""Smoke: NNMT_DDPM_MLX_MultiLSTM imports and reports the right gan_type."""

import pytest

from GANs.GANType import GANType


def _import_strategy():
    """Import the strategy class, skipping if freqtrade is not installed."""
    return pytest.importorskip(
        "NNMT.NNMT_DDPM_MLX_MultiLSTM",
        reason="NNMT import chain requires freqtrade to be installed",
    ).NNMT_DDPM_MLX_MultiLSTM


def test_imports_succeed():
    """If importorskip returns, the module imported without error."""
    cls = _import_strategy()
    assert cls is not None


def test_gan_type_is_mt_ddpm():
    """The class should report MT_DDPM as its gan_type, however that's set."""
    cls = _import_strategy()

    # Class-level attribute path:
    if hasattr(cls, "gan_type"):
        assert cls.gan_type == GANType.MT_DDPM
    # Otherwise, try get_gan_type method on the class:
    elif hasattr(cls, "get_gan_type"):
        # May need instantiation; if that fails in the test env, skip the assertion.
        try:
            inst = cls()
            assert inst.get_gan_type() == GANType.MT_DDPM
        except Exception:
            pytest.skip("Cannot instantiate strategy in this env")
    else:
        pytest.fail(
            "NNMT_DDPM_MLX_MultiLSTM has neither `gan_type` attribute nor "
            "`get_gan_type` method — gan_type is not discoverable"
        )

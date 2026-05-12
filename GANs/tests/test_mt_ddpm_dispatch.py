"""Test that GANInterface dispatches MT_DDPM correctly."""

from GANs.GANInterface import GANInterface, GAN_BACKED_TYPES
from GANs.GANType import GANType


def test_mt_ddpm_in_gan_backed_types():
    assert GANType.MT_DDPM in GAN_BACKED_TYPES


def test_mt_ddpm_defaults_present():
    cfg = dict(GANInterface._DEFAULTS.get(GANType.MT_DDPM, {}))
    # Defaults should include the standard MT_DDPM training params.
    assert "epochs" in cfg or "batch_size" in cfg, f"defaults missing: {cfg}"


def test_create_gan_has_mt_ddpm_defaults():
    import pytest
    CreateGAN = pytest.importorskip(
        "GANs.CreateGAN",
        reason="CreateGAN import chain requires freqtrade to be installed",
    ).CreateGAN
    # _DEFAULTS_BY_TYPE is a class attribute on CreateGAN — verify MT_DDPM key exists.
    assert GANType.MT_DDPM in CreateGAN._DEFAULTS_BY_TYPE

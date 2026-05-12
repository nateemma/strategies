"""Smoke test: GANType.MT_DDPM exists and is distinct."""

from GANs.GANType import GANType


def test_mt_ddpm_member_exists():
    assert hasattr(GANType, "MT_DDPM")


def test_mt_ddpm_is_distinct():
    members = {m for m in GANType}
    # MT_DDPM is distinct from all other members
    assert GANType.MT_DDPM in members
    assert GANType.MT_DDPM is not GANType.TAB_DDPM
    assert GANType.MT_DDPM is not GANType.MT_WGAN

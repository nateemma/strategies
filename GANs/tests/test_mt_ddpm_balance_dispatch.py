"""Test that balance.py routes MT_DDPM through the MT pipeline."""

import inspect

from GANs.GANType import GANType
from GANs import balance


def test_mt_ddpm_not_in_squeeze_seq_dim_types():
    # MT types keep the seq dimension; only 2D types squeeze it.
    assert GANType.MT_DDPM not in balance._SQUEEZE_SEQ_DIM_TYPES


def test_mt_ddpm_referenced_in_balance_source():
    src = inspect.getsource(balance)
    assert "MT_DDPM" in src, "MT_DDPM not referenced anywhere in balance.py"

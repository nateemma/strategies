"""040 T020 + T026 (research R4, R6): ARO's operating point on NNMT reuses NNMT_MLX's weights and keeps
every parent key; the four model variants have the inputs R6 lists, distinct model keys, and causal
raw8 inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "NNMT"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from Framework import ModelKey as MK  # noqa: E402
from NNMT.NNMT_LR import NNMT_LR  # noqa: E402
from NNMT.NNMT_LR_Raw import NNMT_LR_Raw  # noqa: E402
from NNMT.NNMT_LR_RI import NNMT_LR_RI  # noqa: E402
from NNMT.NNMT_MLX import NNMT_MLX  # noqa: E402
from NNMT.NNMT_MLX_OP import NNMT_MLX_OP  # noqa: E402
from NNMT.NNMT_Raw import NNMT_Raw  # noqa: E402
from NNPredict._inputs import R8  # noqa: E402

CONFIG = {"exchange": {"pair_whitelist": ["SOL/USDT", "LINK/USDT"]}, "timeframe": "15m"}


def _key(cls):
    pinned = type(cls.__name__ + "_f", (cls,), {"MODEL_TRAIN_TIMERANGE": "20210101-20230101"})
    return MK.key_hash(MK.resolve(pinned, CONFIG))


def test_operating_point_is_aros_and_keeps_every_parent_key():
    aro = json.loads((STRAT / "NNNC" / "NNNC_A_Raw_Open.json").read_text())["params"]["buy"]
    bp = NNMT_MLX_OP.buy_params
    for k, v in aro.items():
        assert bp[k] == v, k
    assert set(NNMT_MLX.buy_params) <= set(bp)
    assert bp["profit_prediction_threshold"] == NNMT_MLX.buy_params["profit_prediction_threshold"]
    assert NNMT_MLX_OP.sell_params == NNMT_MLX.sell_params  # NNMT's exits unchanged


def test_operating_point_reuses_the_weights():
    assert _key(NNMT_MLX_OP) == _key(NNMT_MLX)


def test_variant_keys_are_distinct():
    keys = {c.__name__: _key(c) for c in (NNMT_MLX, NNMT_Raw, NNMT_LR, NNMT_LR_Raw, NNMT_LR_RI)}
    assert len(set(keys.values())) == 5, keys


def _inputs(cls, df):
    s = cls.__new__(cls)
    s.include_list = list(cls.include_list)
    s.pre_normalized_columns = list(getattr(cls, "pre_normalized_columns", []))
    # only the mixin's own column builder (the indicator chain is not under test here)
    from NNPredict._inputs import raw8_columns

    for c, v in raw8_columns(df).items():
        df[c] = v.to_numpy()
    s.include_list = s.include_list + [c for c in R8 if c not in s.include_list]
    return s.model_inputs()


def _ohlcv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c,
                         "volume": rng.uniform(1, 10, n)})


def test_inputs_are_what_r6_lists():
    df = _ohlcv()
    assert _inputs(NNMT_Raw, df.copy()) == R8
    assert _inputs(NNMT_LR_Raw, df.copy()) == R8
    ri = _inputs(NNMT_LR_RI, df.copy())
    assert ri[: len(NNMT_MLX.include_list)] == list(NNMT_MLX.include_list) and ri[-len(R8):] == R8
    assert NNMT_LR.classifier_type.name == NNMT_LR_Raw.classifier_type.name == NNMT_LR_RI.classifier_type.name == "LSTM_RAW"
    assert NNMT_Raw.classifier_type == NNMT_MLX.classifier_type


@pytest.mark.parametrize("cut", [300, 450, 599])
def test_raw8_inputs_are_causal(cut):
    from NNPredict._inputs import raw8_columns

    df = _ohlcv()
    full = pd.DataFrame(raw8_columns(df))
    part = pd.DataFrame(raw8_columns(df.iloc[:cut].copy()))
    pd.testing.assert_frame_equal(full.iloc[:cut].reset_index(drop=True), part.reset_index(drop=True),
                                  rtol=1e-6)

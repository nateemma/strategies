"""Shared fixtures: the in-scope strategies loaded once, and a real SOL slice."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def loaded():
    from research.indicator_screen import harvest as H

    H._quiet()
    cfg = H.base_config()
    scope, strats = H.load_scope(cfg)
    return {"cfg": cfg, "scope": scope, "strats": {type(s).__name__: s for s in strats},
            "aro": H.load_aro(), "H": H}


@pytest.fixture(scope="session")
def sol(loaded):
    """SOL/USDT 2025-01 onward: 20,000 bars (long enough for every filter to bind somewhere)."""
    o = loaded["H"].load_ohlcv(loaded["cfg"], "SOL/USDT")
    i = int(o["date"].searchsorted(o["date"].iloc[0].replace(year=2025)))
    return o.iloc[i: i + 20000].reset_index(drop=True)


@pytest.fixture(scope="session")
def base(loaded, sol):
    return loaded["H"].populate_path(loaded["strats"]["CCI"], sol, "SOL/USDT")

"""038 truncation test for the learned-tier inputs (FR-012; research R7): recompute every L and LW input
on data truncated at a cut and require every value up to the cut to be identical. LW inputs that fail
are excluded from set LW and named.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.truncation
"""
from __future__ import annotations

import numpy as np

from research.forecast import common as C
from research.forecast import features as X
from research.forecast import frames as F

PAIRS = ("SOL/USDT", "AAVE/USDT")
N_CUTS = 5


def check(fr, cuts) -> dict[str, bool]:
    full = X.set_L(fr).join(X.set_W(fr))
    ok = {c: True for c in full.columns}
    for cut in cuts:
        part = fr.iloc[: cut + 1]
        trunc = X.set_L(part).join(X.set_W(part))
        for c in full.columns:
            a, b = full[c].to_numpy()[: cut + 1], trunc[c].to_numpy()
            if not np.array_equal(a, b, equal_nan=True):
                ok[c] = False
    return ok


def main(argv=None) -> int:
    rng = np.random.default_rng(0)
    res: dict[str, bool] = {}
    for pair in PAIRS:
        fr = F.read(pair, ["date", "lc", "volume"])
        cuts = sorted(rng.integers(1000, len(fr) - 1000, N_CUTS).tolist())
        for c, ok in check(fr, cuts).items():
            res[c] = res.get(c, True) and ok
    out = {"pairs": PAIRS, "cuts_per_pair": N_CUTS,
           "inputs": {c: v for c, v in res.items() if c in X.LW_COLS},
           "L_inputs": {c: v for c, v in res.items() if c not in X.LW_COLS}}
    C.write_json(C.HERE / "truncation.json", out)
    bad = [c for c, v in res.items() if not v]
    print("truncation:", "all inputs identical up to every cut" if not bad else f"FAIL {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

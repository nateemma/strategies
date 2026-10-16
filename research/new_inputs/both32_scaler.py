"""035 follow-up both32: build `saved_data/main_scaler_both32.pkl` from the two fitted scalers.

RobustScaler is per column, so the composite is exact: ARO's 6 columns keep `main_scaler`'s centre
and scale, the 8 raw8 columns keep `main_scaler_fwdmae_raw8`'s. Nothing is refitted and neither
source file is touched. Refuses to overwrite an existing composite that differs.

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.both32_scaler
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np


STRAT = Path(__file__).resolve().parents[2]
SAVED = STRAT / "saved_data"
SOURCES = ("main_scaler", "main_scaler_fwdmae_raw8")
NAME = "main_scaler_both32"


def compose(scalers) -> object:
    out = copy.deepcopy(scalers[0])
    out.feature_names_in_ = np.concatenate([s.feature_names_in_ for s in scalers]).astype(object)
    out.center_ = np.concatenate([s.center_ for s in scalers])
    out.scale_ = np.concatenate([s.scale_ for s in scalers])
    out.n_features_in_ = len(out.feature_names_in_)
    if len(set(out.feature_names_in_)) != out.n_features_in_:
        raise ValueError("source scalers share a column")
    return out


def main() -> int:
    srcs = [pickle.loads((SAVED / f"{n}.pkl").read_bytes()) for n in SOURCES]
    comp = compose(srcs)
    path = SAVED / f"{NAME}.pkl"
    if path.is_file():
        old = pickle.loads(path.read_bytes())
        same = (list(old.feature_names_in_) == list(comp.feature_names_in_)
                and np.array_equal(old.center_, comp.center_)
                and np.array_equal(old.scale_, comp.scale_))
        if not same:
            raise SystemExit(f"{path} exists and differs; refusing to overwrite")
        print("composite already present and identical")
        return 0
    path.write_bytes(pickle.dumps(comp))
    print(path, list(comp.feature_names_in_))
    return 0


if __name__ == "__main__":
    sys.exit(main())

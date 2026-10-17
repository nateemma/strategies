"""ATR-barrier study, arm B inputs: the 037 composite's 7 member columns (vol/composite.json, chosen on
early + E0 only) per pair from the 037 cache (causal: every column passed the truncation test), each
normalised as 035's z inputs -- (x - trailing 2880-bar mean) / trailing std, min_periods 960, clipped
to +-5, / 5 -> [-1, 1]; no fitted artefact. Written to user_data/data/tb_inputs/<PAIR>.parquet with
build_meta.json (table_sha = hash of the member list and the normalisation).

    PYTHONPATH=. ../../.venv/bin/python -m research.tb_model.inputs
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from research.cmp_sizer.signals import members
from research.indicator_screen import harvest as H
from research.indicator_screen.panel import screen_pairs
from research.new_inputs import inputs as I

OUT = H.ROOT / "user_data" / "data" / "tb_inputs"


def colname(c: str) -> str:
    return "tbi_" + re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")


def table_sha(cols: list[str]) -> str:
    spec = {"members": cols, "z": [I.Z_WINDOW, I.Z_MIN, I.Z_CLIP]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def znorm(x: pd.Series) -> np.ndarray:
    m = x.rolling(I.Z_WINDOW, min_periods=I.Z_MIN).mean()
    s = x.rolling(I.Z_WINDOW, min_periods=I.Z_MIN).std()
    return (((x - m) / s).clip(-I.Z_CLIP, I.Z_CLIP) / I.Z_CLIP).to_numpy(float)


def main() -> int:
    cols, _ = members()
    OUT.mkdir(parents=True, exist_ok=True)
    for pair in screen_pairs():
        f = pd.read_feather(H.CACHE / f"{pair.replace('/', '_')}.feather", columns=["date", *cols])
        out = pd.DataFrame({"date": pd.to_datetime(f["date"], utc=True)})
        for c in cols:
            out[colname(c)] = znorm(f[c].astype(float))
        out.to_parquet(OUT / f"{pair.replace('/', '_')}.parquet", index=False)
        print(pair, len(out), "rows", flush=True)
    meta = {"table_sha": table_sha(cols), "members": cols, "columns": [colname(c) for c in cols],
            "z": [I.Z_WINDOW, I.Z_MIN, I.Z_CLIP], "source": "037 cache (user_data/data/indicator_screen)"}
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta["table_sha"], meta["columns"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

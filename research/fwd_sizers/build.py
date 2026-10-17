"""Forward confirmation of cs_cmp / cs_mix (preregistration.json): the composite on 2026-09-07..10-12.

1. The 7 composite member columns, by the 037 harvest's own code (harvest_frame with the scope's first
   class as the base and the member classes; extras ratio_frame / raw8_frame), cast to float32 as the
   cache stores them, on candles from DATA_START through FWD_END. Instrument: equal to the 037 cache
   on every bar before 2026-09-07.
2. f14 frozen: each member z-scored with f14's mean / sd (research/cmp_sizer/signals: tradeable bars
   before f14's validation tail, from the 037 cache), signed, missing -> 0, averaged; written for
   gbb candidate bars in the forward window as user_data/data/cmp_sizer/HGB_f15.parquet.
3. cs_mix: f14's percentile knots copied as user_data/data/mix_sizer/MIX_f15.json.

    PYTHONPATH=. ../../.venv/bin/python -m research.fwd_sizers.build
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from research.cmp_sizer import signals as CS
from research.hgb_sizer import signals as HS
from research.indicator_screen import discover as D
from research.indicator_screen import extras as E
from research.indicator_screen import harvest as H
from research.indicator_screen import windows as W
from research.indicator_screen.panel import Panel
from research.mix_sizer.knots import OUT as MIX

HERE = Path(__file__).resolve().parent
FWD_START, FWD_END = "2026-09-07", "2026-10-12"
FOLD = "f15"


def load(cfg, pair: str) -> pd.DataFrame:
    from freqtrade.configuration import TimeRange
    from freqtrade.data import history

    tr = TimeRange.parse_timerange(f"{W.DATA_START.replace('-', '')}-{FWD_END.replace('-', '')}")
    df = history.load_pair_history(pair=pair, timeframe="15m", datadir=cfg["datadir"], timerange=tr,
                                   fill_up_missing=True, data_format=cfg.get("dataformat_ohlcv", "feather"),
                                   candle_type=cfg.get("candle_type_def", "spot"))
    return df[df["date"] < pd.Timestamp(FWD_END, tz="UTC")].reset_index(drop=True)


def member_frame(cfg, strats, pair: str, cols: list[str]) -> pd.DataFrame:
    f, _ = H.harvest_frame(load(cfg, pair), strats, None, pair, keep_defects=True)
    srcs = [c[: -len("/close")] for c in cols if c.endswith("/close")]
    f = pd.concat([f, E.raw8_frame(f), E.ratio_frame(f, srcs)], axis=1)
    f = f[["date", "base:gbb_candidate", *cols]].copy()
    f[cols] = f[cols].astype(np.float32)        # as the 037 cache stores values (extras come out float64)
    return f


def run() -> dict:
    H._quiet()
    cols, sign = CS.members()
    classes = sorted({c.split(":")[0] for c in cols if not c.startswith("raw8:")})
    cfg = H.base_config()
    first = D.discover()["classes"][0]["class"]
    strats = [H.load_strategy(cfg, c) for c in [first, *[c for c in classes if c != first]]]

    panel = Panel(tradeable_only=True)
    f14 = next(f for f in W.folds() if f.id == "f14")
    tails = HS.tail_starts(panel, f14, HS.raw_dates(panel.pairs))
    nr = CS.norm_rows(panel, f14, tails)
    d = panel.columns(cols)
    X = np.column_stack([d[c] for c in cols]).astype(float)
    mu, sd = np.nanmean(X[nr], 0), np.nanstd(X[nr], 0)

    inst, rows = {}, []
    for pair in panel.pairs:
        mf = member_frame(cfg, strats, pair, cols)
        ref = pd.read_feather(H.CACHE / f"{pair.replace('/', '_')}.feather", columns=["date", *cols])
        old = mf[mf["date"] < pd.Timestamp(FWD_START, tz="UTC")].reset_index(drop=True)
        assert len(old) == len(ref), (pair, len(old), len(ref))
        inst[pair] = {c: int(np.sum(~np.isclose(old[c].to_numpy(float), ref[c].to_numpy(float),
                                                rtol=0, atol=0, equal_nan=True))) for c in cols}
        fw = mf[(mf["date"] >= pd.Timestamp(FWD_START, tz="UTC")) & (mf["base:gbb_candidate"] > 0)]
        Z = np.nan_to_num((fw[cols].to_numpy(float) - mu) / np.where(sd > 0, sd, 1) * sign)
        rows.append(pd.DataFrame({"date": pd.to_datetime(fw["date"], utc=True), "pair": pair, "p_buy": Z.mean(1)}))
        print(pair, "mismatches", sum(inst[pair].values()), "forward candidates", len(fw), flush=True)
    mism = sum(v for r in inst.values() for v in r.values())
    tab = pd.concat(rows, ignore_index=True)
    tab.to_parquet(CS.OUT / f"HGB_{FOLD}.parquet", index=False)
    shutil.copyfile(MIX / "MIX_f14.json", MIX / f"MIX_{FOLD}.json")
    out = {"instrument_mismatches": mism, "per_pair": inst, "forward_candidate_bars": len(tab),
           "f14_mean": mu.tolist(), "f14_sd": sd.tolist(), "members": cols}
    (HERE / "build.json").write_text(json.dumps(out, indent=1))
    print("TOTAL mismatches", mism)
    return out


if __name__ == "__main__":
    raise SystemExit(0 if run()["instrument_mismatches"] == 0 else 1)

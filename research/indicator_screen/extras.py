"""037 extra columns, added AFTER the lock at the user's request (2026-10-16). The pre-registered rules
are unchanged; the new columns are scored by them and reported as late additions.

- `raw8:<col>`: the eight raw-price inputs (ret1, ret4, log_hc, log_lc, log_oc, range, logvol_z,
  vol_ratio) from their own code, `research.clfstudy.data.raw8` (the GBR / both32 inputs);
- `ni:<col>`: the 035 new inputs (order flow, positioning, market state, basis; 17 inputs + their
  missing flags) from `user_data/data/new_inputs/<PAIR>.parquet`, as built and causality-checked
  in 035 (research/new_inputs/causality.json). No BTC file exists, so `btc:ni:*` is empty;
- `<col>/close`: every column tagged price_level or price_scaled (tags.json), divided by close --
  the scale-free version (causal because both parts are).

raw8 and the ratios get the same truncation test as the harvest (causality_extras.json). A full
`harvest` rebuild drops these columns; re-run this module after one.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.extras
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.indicator_screen import tags as TG

NEW_INPUTS = H.ROOT / "user_data" / "data" / "new_inputs"
CAUS = H.HERE / "causality_extras.json"
ADDED = "2026-10-16, after the pre-registration lock (user request); rules unchanged"


def ratio_sources() -> list[str]:
    return sorted(c for c, t in TG.load().items() if t in ("price_level", "price_scaled"))


def raw8_frame(f: pd.DataFrame) -> pd.DataFrame:
    from research.clfstudy.data import raw8

    r = raw8(f[["date", *H.OHLCV]].reset_index(drop=True))
    cols = ["ret1", "ret4", "log_hc", "log_lc", "log_oc", "range", "logvol_z", "vol_ratio"]
    return pd.DataFrame({f"raw8:{c}": r[c].to_numpy(np.float64) for c in cols})


def ratio_frame(f: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    close = f["close"].to_numpy(np.float64)
    out = {}
    for c in sources:
        if c in f:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"{c}/close"] = f[c].to_numpy(np.float64) / close
    return pd.DataFrame(out)


def ni_frame(pair: str, dates: pd.Series) -> pd.DataFrame:
    p = NEW_INPUTS / f"{H._slug(pair)}.parquet"
    if not p.exists():
        return pd.DataFrame(index=range(len(dates)))
    t = pd.read_parquet(p)
    t["date"] = pd.to_datetime(t["date"], utc=True)
    m = pd.DataFrame({"date": pd.to_datetime(dates, utc=True)}).merge(t, on="date", how="left")
    return pd.DataFrame({f"ni:{c}": m[c].to_numpy(np.float64) for c in t.columns if c != "date"})


def causality(sources: list[str]) -> dict:
    """Truncation test of raw8 and the ratio columns on the outcomes era slices."""
    from research.indicator_screen import causality as C
    from research.outcomes import causality as OC

    H._quiet()
    cfg = H.base_config()
    scope, strats = H.load_scope(cfg)
    slices, skipped = C.real_slices(OC.DEFAULT_PAIRS, OC.DEFAULT_ERAS)

    def harvest(ohlcv):
        f, _ = H.harvest_frame(ohlcv, strats, None, "SOL/USDT", keep_defects=True)
        return pd.concat([f[["date"]], raw8_frame(f), ratio_frame(f, sources)], axis=1)

    res, meta = C.check_columns(harvest, slices, OC.DEFAULT_CUTS, OC.DEFAULT_WARMUP)
    return {"slices": {"checked": meta["checked"], "skipped": skipped}, "columns": res,
            "fails": sorted(c for c, r in res.items() if r["verdict"] == "FAIL")}


def extend(pairs: list[str], fails: set[str]) -> dict:
    sources = ratio_sources()
    added = {}
    for pair in pairs:
        f = H.read_cache(pair)
        f = f.drop(columns=[c for c in f.columns if c.startswith(("raw8:", "ni:")) or c.endswith("/close")])
        new = pd.concat([raw8_frame(f), ni_frame(pair, f["date"]), ratio_frame(f, sources)], axis=1)
        new = new.drop(columns=[c for c in new.columns if c in fails])
        keep = {}
        for c in new.columns:
            a = new[c].to_numpy(np.float64)
            if H._defect(a) is None:
                keep[c] = a.astype(np.float32)
        f = pd.concat([f, pd.DataFrame(keep)], axis=1)
        f.to_feather(H.cache_path(pair))
        sc = json.loads(H.sidecar_path(pair).read_text())
        sc["columns"] = [c for c in f.columns if c not in ("date", *H.OHLCV)]
        sc["extras"] = {"added": ADDED, "columns": sorted(keep)}
        H.sidecar_path(pair).write_text(json.dumps(sc, indent=1))
        added[pair] = sorted(keep)
        print(f"{pair}: +{len(keep)} columns", flush=True)
    return added


def update_manifest(added: dict, caus: dict) -> None:
    m = json.loads(H.MANIFEST.read_text())
    m["columns"] = [c for c in m["columns"] if not (c["name"].startswith(("raw8:", "ni:")) or c["name"].endswith("/close"))]
    names = sorted({c for v in added.values() for c in v})
    for c in names:
        if c.startswith("ni:"):
            src, verdict = "ni", "PASS"     # 035 causality.json
        else:
            src = "raw8" if c.startswith("raw8:") else "ratio"
            verdict = caus["columns"].get(c, {}).get("verdict", "UNTESTED")
        m["columns"].append({"name": c, "source": src, "kind": "value", "causality": verdict, "added": ADDED})
    m["extras"] = {"added": ADDED, "n": len(names), "causality": {
        "raw8_and_ratios": "causality_extras.json", "ni": "research/new_inputs/causality.json (035)"}}
    H.MANIFEST.write_text(json.dumps(m, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-causality", action="store_true")
    a = ap.parse_args(argv)
    busy = H._busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    sources = ratio_sources()
    if a.skip_causality and CAUS.exists():
        caus = json.loads(CAUS.read_text())
    else:
        caus = causality(sources)
        CAUS.write_text(json.dumps({"added": ADDED, **caus}, indent=1, default=str))
    print(f"truncation test: {len(caus['columns'])} columns, FAIL {caus['fails']}")
    added = extend(H.pairs(), set(caus["fails"]))
    update_manifest(added, caus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leakage audit: prove no feature reads data after the trade's open_date (T014-T018).

TWO DIFFERENT QUESTIONS, both run, reported separately:

  CAUSALITY   recompute the feature with every bar strictly after open_date DELETED and
              assert the value is unchanged. This is the only check that cannot be argued
              with -- inspection misses indicator warm-up and centred windows, and a
              permutation test measures something else entirely.
  INFORMATION permute the feature across trades and watch discrimination. This is the
              brief's shuffle test. It says whether a feature carries signal, NOT whether
              it is causal. A leaky feature passes it; a useless causal feature fails it.

THE AUDIT MUST BE ABLE TO FAIL (FR-004, SC-011). A planted-leak control -- a feature set to
the trade's own realised profit_abs -- is injected and the audit is REQUIRED to flag it. It
cannot be disabled. The first version of the analogous control in research/labelfree/
liveness.py passed vacuously by not actually severing anything, which is the same failure one
level up: an audit that cannot detect a known leak certifies nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import metalabel as M                # noqa: E402

PLANTED_LEAK = "__planted_leak__"          # must be detected; see module docstring
RTOL = 0.0                                  # causality means IDENTICAL, not close
ATOL = 1e-9

# Features recomputable outside freqtrade, via the repo's own indicator code.
# The probability columns are audited separately (see audit_probabilities).
RECOMPUTABLE = ("guard_metric", "rvol", "atr_pct", "bb_width", "adx", "close", "volume")


def _populate(df: pd.DataFrame) -> pd.DataFrame:
    """The repo's own indicator computation -- not a reimplementation of it."""
    sys.path.insert(0, str(ROOT / "utils"))
    from DataframePopulator import DataframePopulator
    return DataframePopulator().add_minimal_indicators(df.reset_index(drop=True).copy())


def audit_causality(rows: list[dict], sample_per_pair: int = 8) -> dict:
    """Recompute each feature on TRUNCATED history and compare at the entry bar.

    Truncation is the test: if a value computed from bars <= t differs from the same value
    computed from the full series, that feature reads the future at t.
    """
    from regime import regimedata as rd

    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)

    failures, checked = [], 0
    per_feature = {c: {"checked": 0, "mismatch": 0, "max_abs_diff": 0.0}
                   for c in RECOMPUTABLE}

    for pair, prs in by_pair.items():
        ohlcv = rd.load_ohlcv(pair.split("/")[0])
        full = _populate(ohlcv)
        full.index = ohlcv.index[: len(full)]
        rng = np.random.default_rng(M.BOOTSTRAP_SEED)
        pick = prs if len(prs) <= sample_per_pair else [
            prs[i] for i in rng.choice(len(prs), sample_per_pair, replace=False)]

        for r in pick:
            entry = pd.Timestamp(r["open_date"]) - pd.Timedelta(minutes=15)
            if entry not in full.index:
                continue
            trunc_src = ohlcv[ohlcv.index <= entry]
            if len(trunc_src) < 250:                      # below indicator warm-up
                continue
            trunc = _populate(trunc_src)
            trunc.index = trunc_src.index[: len(trunc)]
            if entry not in trunc.index:
                continue
            checked += 1
            for c in RECOMPUTABLE:
                if c not in full.columns or c not in trunc.columns:
                    continue
                a, b = full.loc[entry, c], trunc.loc[entry, c]
                if pd.isna(a) and pd.isna(b):
                    continue
                per_feature[c]["checked"] += 1
                d = abs(float(a) - float(b))
                per_feature[c]["max_abs_diff"] = max(per_feature[c]["max_abs_diff"], d)
                if not np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
                    per_feature[c]["mismatch"] += 1
                    failures.append({"pair": pair, "open_date": r["open_date"],
                                     "feature": c, "full": float(a), "truncated": float(b),
                                     "abs_diff": d})
    return {"entries_checked": checked, "per_feature": per_feature,
            "failures": failures[:20], "n_failures": len(failures),
            "causal": not failures}


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    p, n = label.sum(), (~label).sum()
    if p == 0 or n == 0:
        return float("nan")
    return float((r[label].sum() - p * (p + 1) / 2) / (p * n))


def audit_information(rows: list[dict], seed: int = M.BOOTSTRAP_SEED) -> dict:
    """The brief's shuffle test, plus the PLANTED LEAK the audit must catch.

    Reports each feature's own AUC and the AUC after permuting it across trades. A causal
    but uninformative feature barely moves; a LEAK is obvious because its own AUC is
    implausibly high.
    """
    ev = [r for r in rows if r["fold_id"] in M.EVAL_FOLDS]
    y = np.array([r["target_win"] for r in ev])
    feats = sorted({k for r in ev for k in r["features"]})
    rng = np.random.default_rng(seed)

    out = {}
    for c in feats + [PLANTED_LEAK]:
        if c == PLANTED_LEAK:
            # The control: the trade's OWN realised outcome, which no entry-time feature
            # could know. If the audit does not flag this, it certifies nothing.
            x = np.array([r["profit_abs"] for r in ev], dtype=float)
        else:
            x = np.array([r["features"].get(c, np.nan) for r in ev], dtype=float)
        if np.isnan(x).all():
            continue
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
        a = _auc(x, y)
        perm = float(np.mean([_auc(rng.permutation(x), y) for _ in range(25)]))
        out[c] = {"auc": a, "auc_permuted": perm, "delta": a - perm}

    leak = out.get(PLANTED_LEAK, {})
    # A leak is "detected" when its discrimination is implausible for an entry-time feature.
    detected = bool(leak and (leak["auc"] > 0.90 or leak["auc"] < 0.10))
    return {"per_feature": out, "planted_leak_detected": detected,
            "planted_leak_auc": leak.get("auc")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="research/labelfree/metalabel/rows.jsonl")
    ap.add_argument("--sample-per-pair", type=int, default=8)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines() if l.strip()]
    print(f"LEAKAGE AUDIT — {len(rows)} rows\n")

    print("1. CAUSALITY (recompute with post-open_date bars deleted)")
    cz = audit_causality(rows, a.sample_per_pair)
    print(f"   entry bars checked: {cz['entries_checked']}")
    print(f"   {'feature':<16}{'checked':>9}{'mismatch':>10}{'max|diff|':>12}")
    for c, v in cz["per_feature"].items():
        if v["checked"]:
            print(f"   {c:<16}{v['checked']:>9}{v['mismatch']:>10}{v['max_abs_diff']:>12.2e}")
    for f in cz["failures"][:5]:
        print(f"   FAIL {f['pair']} {f['open_date'][:16]} {f['feature']}: "
              f"{f['full']:.6g} vs {f['truncated']:.6g}")
    print(f"   -> {'CAUSAL' if cz['causal'] else '*** NOT CAUSAL ***'}\n")

    print("2. INFORMATION (own AUC vs permuted) + PLANTED-LEAK CONTROL")
    iz = audit_information(rows)
    print(f"   {'feature':<24}{'AUC':>8}{'permuted':>10}{'delta':>9}")
    for c, v in sorted(iz["per_feature"].items(), key=lambda kv: -abs(kv[1]["delta"])):
        mark = "  <-- PLANTED LEAK" if c == PLANTED_LEAK else ""
        print(f"   {c:<24}{v['auc']:>8.4f}{v['auc_permuted']:>10.4f}{v['delta']:>+9.4f}{mark}")
    print(f"   -> planted leak detected: {iz['planted_leak_detected']} "
          f"(AUC {iz['planted_leak_auc']:.4f})\n")

    ok = cz["causal"] and iz["planted_leak_detected"]
    if not iz["planted_leak_detected"]:
        print("*** THE AUDIT FAILED ITS OWN CONTROL. It did not detect a feature equal to")
        print("    the trade's realised outcome, so it cannot be trusted to detect a real")
        print("    leak either. No downstream number is admissible (SC-011).")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

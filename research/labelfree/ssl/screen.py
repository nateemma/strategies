"""The screen: does the embedding beat the hand-built features? (T006-T010, T027-T033)

THE BAR IS MEASURED, NEVER QUOTED. rho ~ 0.15 is cited throughout this repo as "the OHLCV
information ceiling". Measured across five folds and six horizons it is a HORIZON-3-TO-5 number:

    H        3        6       12       24       48       96
    median  0.1465   0.1003   0.0684   0.0556   0.0499   0.0254

Predictability decays with forward horizon, so a constant is the wrong bar at every horizon but
one. Quoting 0.15 at H=48 -- where the hand-built features themselves reach 0.083 -- would have
set a bar THE BASELINE CANNOT CLEAR, and an embedding failing it would fail for a reason
unrelated to the hypothesis. Each screen therefore computes its own bar with the same ridge on
the same split.

THREE-WAY SPLIT, because there are two fittable objects and the obvious design leaks through the
second. The encoder is unsupervised, so it is tempting to fit the ridge on the same range -- but
the ridge IS supervised, and one fitted and scored on the same bars beats any bar. That is how a
null becomes a false positive.

    fold training range   ->  A  encoder   (no labels)
                          ->  B  ridge     (supervised; labels from B only)
    fold test range       ->  C  score     <- the only number reported

PRIMARY IS FIXED BEFORE ANY SCREEN RUNS and written into the record BEFORE scoring, so the record
itself shows it was not chosen afterwards. Four screens against one threshold would be four
chances to clear a bar meant to be cleared once -- the failure brief 05 measured at PBO 0.7128.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.ssl import inputs as IN                 # noqa: E402

# Pre-registered before any screen runs (FR-007b, SC-001a).
#
# AMENDED 2026-09-15, after the bar was measured and BEFORE any encoder was trained: the primary
# is a CONJUNCTION over both horizons. The corrected per-pair temporal bar at H=48 is 0.0499
# median with a MINIMUM OF -0.0169 -- the hand-built features are themselves barely
# distinguishable from noise there, so "beat the hand-built features at H=48" is close to "beat
# zero", which a null could clear by accident. H=6 has a bar that discriminates (0.1003 median,
# 0.0715 minimum). Requiring both keeps the declared horizon honest and removes the accidental
# pass. It RAISES the gate; a change in the other direction would not have been made here.
PRIMARY_INPUT = "engineered"
PRIMARY_HORIZONS = (48, 6)
INPUTS = ("engineered", "ohlcv")
HORIZONS = (6, 48)

# The measured ceiling under a CAUSALLY ALIGNED target, for the reproduction test (T009). NOT a
# bar -- each screen recomputes its own.
#
# THIS CONSTANT HAS BEEN WRONG TWICE, so treat a mismatch as a signal, not a nuisance:
#   gen 1  {3: 0.1717 ... 48: 0.0826}  split by PAIR while claiming to split by time
#   gen 2  {3: 0.1465 ... 48: 0.0499}  target started at open[t], including bar t's own move
#   gen 3  below                       target open[t+1] -> open[t+1+H]
# If the bar stops reproducing these, DO NOT adjust the constant to match -- check the target's
# alignment first. Both earlier generations looked plausible and both were leaking.
# Full history: research/labelfree/CEILING_CORRECTIONS.md
#
# R1's first numbers were higher and WRONG. That probe stacked every pair and then sliced the
# array, so "the first 60%" was THE FIRST 6 PAIRS rather than the first 60% of time — a
# contemporaneous cross-sectional split, not forward-in-time prediction. Measured on f07: at H=6
# it makes no difference (0.1601 vs 0.1602), but at H=48 it nearly doubles rho (0.0926 vs
# 0.0490), because a contemporaneous split lets market-wide moves leak between the fit and score
# sets and those dominate at longer horizons.
MEASURED_CEILING = {3: 0.0352, 6: 0.0461, 12: 0.0396, 24: 0.0328, 48: 0.0238, 96: 0.0401}

RIDGE_ALPHA = 1.0
BOOT_N = 2000
BOOT_SEED = 0


def splits(n: int, a: float = 0.7) -> tuple[slice, slice]:
    """Training range -> A (encoder) | B (ridge), in time order."""
    cut = int(a * n)
    return slice(0, cut), slice(cut, n)


def purge(X: np.ndarray, y: np.ndarray, horizon: int, embargo: int = 96):
    """Drop the last `horizon + embargo` rows of a split.

    A label at the end of a split reaches `horizon` bars past its boundary, so without this the
    ridge's training labels overlap the scoring range (FR-003). Applied at EVERY boundary, not
    only A|B.
    """
    k = horizon + embargo
    return (X[:-k], y[:-k]) if len(X) > k else (X[:0], y[:0])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if a.size < 8:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _fit_score(Xb, yb, Xc, yc) -> float:
    """Ridge fitted on B ONLY, scored on C ONLY."""
    from sklearn.linear_model import Ridge
    if len(Xb) < 50 or len(Xc) < 50:
        return float("nan")
    r = Ridge(alpha=RIDGE_ALPHA).fit(Xb, yb)
    return _spearman(r.predict(Xc), yc)


def _boot_ci(Xb, yb, Xc, yc, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Interval on the C-side rho by resampling C — the bar is compared with an interval, never
    bare (FR-009)."""
    from sklearn.linear_model import Ridge
    if len(Xb) < 50 or len(Xc) < 50:
        return (float("nan"), float("nan"))
    pred = Ridge(alpha=RIDGE_ALPHA).fit(Xb, yb).predict(Xc)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(yc), size=(n, len(yc)))
    vals = np.array([_spearman(pred[i], yc[i]) for i in idx[:200]])   # 200 is ample for a CI
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def stack(per_pair: dict, horizon: int, sl: slice):
    """Pool pairs over one split, aligning each pair's features to its own forward return."""
    A, B = [], []
    for X, px in per_pair.values():
        n = min(len(X), len(px))
        fwd = IN.forward_return(px[:n], horizon)
        if fwd.size < 100:
            continue
        Xi, yi = X[:len(fwd)], fwd
        Xi, yi = Xi[sl], yi[sl]
        A.append(Xi); B.append(yi)
    if not A:
        return np.zeros((0, 1)), np.zeros(0)
    X = np.vstack(A); y = np.concatenate(B)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return X[m], y[m]


ENCODER_MAX_BARS = 40_000        # cap per pair on split A; TS2Vec samples random crops anyway


def load_split(fold, loader, which: str) -> dict:
    """Per-pair (X, px) over the fold's TRAINING or TEST range."""
    s, e = ((fold.train_start, fold.train_end) if which == "train"
            else (fold.test_start, fold.test_end))
    out = {}
    for p in sorted(fold.pair_universe):
        try:
            X, px = loader(p, s, e)
            if len(X) > 600:
                out[p] = (X, px)
        except Exception:
            pass
    return out


def handbuilt_bar(fold, horizon: int) -> dict:
    """THE BAR: the same ridge on the same split as the embedding it is compared against.

    Ridge fitted on the fold's TRAINING range (split B), scored on its TEST range (split C) --
    the same geometry the embedding screen uses, so the two numbers are comparable.
    """
    tr = load_split(fold, IN.engineered, "train")
    te = load_split(fold, IN.engineered, "test")
    if not tr or not te:
        return {"rho": float("nan"), "ci": [float("nan")] * 2, "n_pairs": 0}
    ntr = max(min(len(X), len(px)) for X, px in tr.values())
    _sa, sb = splits(ntr)
    Xb, yb = stack(tr, horizon, sb)                       # ridge on TRAINING split B
    Xc, yc = stack(te, horizon, slice(0, None))           # scored on TEST
    Xb, yb = purge(Xb, yb, horizon)
    per = tr
    rho = _fit_score(Xb, yb, Xc, yc)
    lo, hi = _boot_ci(Xb, yb, Xc, yc)
    return {"rho": rho, "ci": [lo, hi], "n_pairs": len(per),
            "n_fit": int(len(Xb)), "n_score": int(len(Xc))}


def baseline_table(fold_ids, horizons=(3, 6, 12, 24, 48, 96)) -> dict:
    """T009's reproduction check — the bar across folds and horizons."""
    from research.labelfree.folds import folds as _folds
    by = {f.fold_id: f for f in _folds()}
    out = {}
    for H in horizons:
        vals = []
        for fid in fold_ids:
            r = handbuilt_bar(by[fid], H)
            if np.isfinite(r["rho"]):
                vals.append(r["rho"])
        if vals:
            out[H] = {"median": round(float(np.median(vals)), 4),
                      "min": round(float(np.min(vals)), 4),
                      "max": round(float(np.max(vals)), 4),
                      "n_folds": len(vals)}
    return out


def embedding_screen(fold, input_name: str, horizon: int, seed: int = 7,
                     epochs: int = 20) -> dict:
    """One (input, horizon) screen: encoder on A, ridge on B, score on C.

    The bar, the embedding and the control are all computed by the SAME ridge on the SAME split,
    so the three numbers are comparable.
    """
    from research.labelfree.ssl import control as CT
    from research.labelfree.ssl import ts2vec as TV

    load = IN.LOADERS[input_name]
    tr = load_split(fold, load, "train")
    te = load_split(fold, load, "test")
    if len(tr) < 3 or len(te) < 3:
        return {"error": "too few pairs"}

    ntr = max(min(len(X), len(px)) for X, px in tr.values())
    sa, sb = splits(ntr)                                  # A encoder | B ridge, both in TRAINING

    # Standardisation stats from SPLIT A ONLY, applied everywhere. Fitting them on a series that
    # includes the scoring range leaks its location and scale into every embedding.
    stats = {pair: IN.standardise(X[sa][-ENCODER_MAX_BARS:])[1] for pair, (X, _) in tr.items()}
    Xa = np.vstack([IN.standardise(X[sa][-ENCODER_MAX_BARS:], stats[pair])[0]
                    for pair, (X, _) in tr.items()])
    model = TV.fit(Xa, seed=seed, epochs=epochs)

    def embed_all(src):
        e, c = {}, {}
        for pair, (X, px) in src.items():
            st = stats.get(pair) or IN.standardise(X[:len(X) // 2])[1]
            z = TV.embed(model, IN.standardise(X, st)[0])
            e[pair] = (z, px)
            c[pair] = (CT.gaussian_block(len(z), z.shape[1], seed=seed), px)
        return e, c

    emb_tr, ctl_tr = embed_all(tr)
    emb_te, ctl_te = embed_all(te)

    out = {"input": input_name, "horizon": horizon,
           "n_pairs_train": len(tr), "n_pairs_test": len(te),
           "width": int(next(iter(emb_te.values()))[0].shape[1])}
    for name, s_tr, s_te in (("embedding", emb_tr, emb_te), ("control", ctl_tr, ctl_te)):
        Xb, yb = stack(s_tr, horizon, sb)                 # ridge on TRAINING split B
        Xc, yc = stack(s_te, horizon, slice(0, None))     # scored on TEST
        Xb, yb = purge(Xb, yb, horizon)
        out[f"rho_{name}"] = round(_fit_score(Xb, yb, Xc, yc), 4)
        lo, hi = _boot_ci(Xb, yb, Xc, yc)
        out[f"ci_{name}"] = [round(lo, 4), round(hi, 4)]
    # THE BAR THAT ACTUALLY ISOLATES THE ENCODER: a ridge on this screen's OWN raw input,
    # same split, same ridge. Comparing embedding(ohlcv) against handbuilt(engineered) conflates
    # TWO differences -- the encoder AND the input -- and measuring it showed the conflation is
    # not hypothetical: ohlcv beat the engineered bar at H=6 entirely because raw log-return
    # channels are a better input (f07: raw ohlcv 0.2298 vs engineered 0.1566), while the
    # EMBEDDING of ohlcv scored 0.2186 -- BELOW its own input. The encoder added nothing, and the
    # engineered-feature bar would have credited it for the input's advantage.
    Xb, yb = stack(tr, horizon, sb)
    Xc, yc = stack(te, horizon, slice(0, None))
    Xb, yb = purge(Xb, yb, horizon)
    out["rho_raw_input"] = round(_fit_score(Xb, yb, Xc, yc), 4)
    out["encoder_adds"] = round(out["rho_embedding"] - out["rho_raw_input"], 4)

    bar = handbuilt_bar(fold, horizon)
    out["rho_handbuilt"] = round(bar["rho"], 4)
    out["ci_handbuilt"] = [round(v, 4) for v in bar["ci"]]
    # The screen's real question is whether the ENCODER earns its place, so the primary
    # comparison is against its own raw input. The hand-built bar is retained as context.
    out["beats_own_input"] = bool(out["encoder_adds"] > 0)
    out["beats_bar"] = bool(out["rho_embedding"] > out["rho_handbuilt"]
                            and out["ci_embedding"][0] > out["rho_handbuilt"])
    out["_emb"] = {k: v[0] for k, v in emb_te.items()}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="f05..f14")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "screen.json"))
    a = ap.parse_args()

    from research.labelfree.baselines import parse_span
    fids = parse_span(a.folds)

    if a.baseline_only:
        tab = baseline_table(fids)
        print("the hand-built bar, measured (NOT quoted):")
        print("  %-6s %9s %9s %9s %8s %10s" % ("H", "median", "min", "max", "folds", "R1 says"))
        for H, v in sorted(tab.items()):
            print("  %-6d %9.4f %9.4f %9.4f %8d %10.4f"
                  % (H, v["median"], v["min"], v["max"], v["n_folds"],
                     MEASURED_CEILING.get(H, float('nan'))))
        # Record WHICH folds produced this. Without it a legitimate subset run is
        # indistinguishable from the bar drifting, and the reproduction test cannot tell them
        # apart — it failed exactly that way on 2026-09-15.
        Path(a.out).with_name("baseline.json").write_text(json.dumps(
            {"folds": fids, "n_folds": len(fids), "by_horizon": tab}, indent=2))
        return 0

    from research.labelfree.folds import folds as _folds
    from research.labelfree.ssl import pairagnostic as PAG
    by = {f.fold_id: f for f in _folds()}

    # SC-001a: the primary is written into the record BEFORE anything is scored.
    rec = {"primary": {"input": PRIMARY_INPUT, "horizons": list(PRIMARY_HORIZONS),
                       "declared": "before any combination was scored"},
           "folds": fids, "results": {}}
    Path(a.out).write_text(json.dumps(rec, indent=2))

    for inp in INPUTS:
        for H in HORIZONS:
            per_fold, pag = [], None
            for fid in fids:
                r = embedding_screen(by[fid], inp, H)
                if "error" in r:
                    continue
                if pag is None:
                    pag = PAG.check(r.pop("_emb"))
                else:
                    r.pop("_emb", None)
                per_fold.append(r)
                print(f"  {inp:<11} H={H:<3} {fid}  emb {r['rho_embedding']:+.4f}  "
                      f"raw {r['rho_raw_input']:+.4f}  adds {r['encoder_adds']:+.4f}  "
                      f"bar {r['rho_handbuilt']:+.4f}  ctl {r['rho_control']:+.4f}", flush=True)
            if not per_fold:
                continue
            med = lambda k: round(float(np.median([x[k] for x in per_fold])), 4)   # noqa: E731
            rec["results"][f"{inp}@{H}"] = {
                "rho_embedding": med("rho_embedding"),
                "rho_handbuilt": med("rho_handbuilt"),
                "rho_control": med("rho_control"),
                "rho_raw_input": med("rho_raw_input"),
                "encoder_adds": med("encoder_adds"),
                "encoder_adds_in_folds": int(sum(x["beats_own_input"] for x in per_fold)),
                "beats_bar_in_folds": int(sum(x["beats_bar"] for x in per_fold)),
                "n_folds": len(per_fold),
                "pair_agnostic": pag,
                "per_fold": per_fold}
            Path(a.out).write_text(json.dumps(rec, indent=2))

    prim = [rec["results"].get(f"{PRIMARY_INPUT}@{h}") for h in PRIMARY_HORIZONS]
    # The encoder must BOTH beat the hand-built bar AND add something over its own raw input.
    passed = all(p and p["rho_embedding"] > p["rho_handbuilt"] and p["encoder_adds"] > 0
                 for p in prim)
    rec["primary_passed"] = bool(passed)
    Path(a.out).write_text(json.dumps(rec, indent=2))

    print("\n%-18s %11s %11s %11s %11s %11s"
          % ("combination", "embedding", "raw input", "adds", "bar", "control"))
    for k, v in rec["results"].items():
        mark = "  <- PRIMARY" if k.startswith(PRIMARY_INPUT) and \
            int(k.split("@")[1]) in PRIMARY_HORIZONS else ""
        print("%-18s %11.4f %11.4f %11.4f %11.4f %11.4f%s"
              % (k, v["rho_embedding"], v["rho_raw_input"], v["encoder_adds"],
                 v["rho_handbuilt"], v["rho_control"], mark))
    print(f"\nPRIMARY ({PRIMARY_INPUT} at H={'+'.join(map(str, PRIMARY_HORIZONS))}): "
          f"{'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

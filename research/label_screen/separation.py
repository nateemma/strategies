"""034 S1 (T010): does a model trained on another label separate width-book trades better than the
gbb_path control? EARLY half (f00-f07) only -- there is no option to read the late half.

Instrument = 033's (research/width_relative/separation.py): the width book (gbb rule + bb_width >=
the previous fold's count-matched cut, width_twin.json) is simulated ONCE per fold from the
control's tables with the 24 h grace, so every model re-weights the IDENTICAL trades. At each
trade's signal bar (entry bar - 1):

  control (pre-registered)  agree = its own entry decision (enter_long of its table);
  control at share          agree = p_buy > its 0.5-share threshold (thresholds_control_share.json);
  candidate                 agree = p_buy > its sz threshold (plan_ls_<tag>_sz);
                            tables LS_<TAG>_fNN (GFM: the existing GFM_fNN).

No BUY-argmax condition on the at-share rows (AGREE_RULE "p_buy", amendment 2026-10-14): balanced
three-class labels rarely make BUY the argmax, so the decision rule cannot reach the 0.5 share.

Per model: agree / disagree n and pp per trade, the gap, AUC of p_buy against pp > 0, and the
folds where the agree share is 0 or 1 (degenerate). S1 per candidate (FR-011): gap above BOTH the
pre-registered 0.072 and the control-at-share gap, with at most MAX_DEGENERATE degenerate folds.

    NNNC_FS_ROUND=2 PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.separation
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.label_screen import candidates as C  # noqa: E402


HERE = Path(__file__).resolve().parent
TWIN = STRAT / "research" / "model_calibration" / "width_twin.json"
SIG = STRAT / "research" / "nnnc_framework" / "signals"
FOLDS_DIR = STRAT / "evaluation" / "folds"
EARLY = tuple(f"f{i:02d}" for i in range(8))
PRE_REGISTERED_GAP = 0.072
MAX_DEGENERATE = 2


def auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Rank AUC (ties averaged): P(score of a positive > score of a negative)."""
    from scipy.stats import rankdata

    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def split_stats(pp: np.ndarray, agree: np.ndarray, score: np.ndarray | None = None) -> dict:
    pp, agree = np.asarray(pp, float), np.asarray(agree, bool)
    a, d = pp[agree], pp[~agree]
    r = {"trades": int(len(pp)),
         "agree": {"n": int(len(a)), "pp_per_trade": round(float(a.mean()), 4) if len(a) else None},
         "disagree": {"n": int(len(d)), "pp_per_trade": round(float(d.mean()), 4) if len(d) else None}}
    ag, dg = r["agree"]["pp_per_trade"], r["disagree"]["pp_per_trade"]
    r["gap"] = None if ag is None or dg is None else round(ag - dg, 4)
    r["auc"] = None if score is None else auc(score, pp > 0)
    return r


def degenerate(df: pd.DataFrame, col: str) -> list[str]:
    s = df.groupby("fold")[col].mean()
    return sorted(s[(s == 0) | (s == 1)].index)


def s1_pass(gap: float | None, refs: dict, n_degenerate: int) -> bool:
    return (gap is not None and n_degenerate <= MAX_DEGENERATE
            and all(r is not None and gap > r for r in refs.values()))


def agree_of(t: pd.DataFrame, threshold: float) -> np.ndarray:
    return np.nan_to_num(t["p_buy"].to_numpy(float), nan=0.0) > threshold


def _table(tag: str, fid: str) -> pd.DataFrame | None:
    p = SIG / (f"GFM_{fid}.parquet" if tag == "GFM" else f"LS_{tag}_{fid}.parquet")
    return pd.read_parquet(p) if p.is_file() else None


def main(argv=None) -> int:
    from research.exit_coupling import feasibility as F
    from research.model_calibration.width_sizing import _signal_rows
    from research.model_calibration.width_twin import _gates
    from research.nnnc_framework import entries as E

    ctl_share = json.loads((HERE / "thresholds_control_share.json").read_text())
    base = E.Model("ARO")
    base.params = dataclasses.replace(base.params, stoploss_grace_hours=24.0)
    folds, cuts = sorted(base.folds), json.loads(TWIN.read_text())["models"]["ARO"]["cuts"]
    cands = [c for c in C.CANDIDATES if c.tag != "CONTROL"]
    thr = {c.tag: json.loads((FOLDS_DIR / f"plan_{C.plan_tag(c, 'sz')}.json").read_text())
           .get("thresholds", {}) for c in cands}
    rows, missing = [], {c.tag: [] for c in cands}
    for i, f in enumerate(folds):
        if f not in EARLY:
            continue
        t = base.tables[f]
        g = _gates(t, E.fold_params(base.prefix, f))
        cut = cuts[folds[i - 1]] if i else cuts[f]
        d = base.simulate(f, g & (t["bb_width"].to_numpy(float) >= cut))
        if d.empty:
            continue
        cur = _signal_rows(t, d)
        sh = _signal_rows(t.assign(enter_long=agree_of(t, float(ctl_share[f]["threshold"])).astype(int)), d)
        r = d.assign(fold=f, agree_ctl=cur["agree"].to_numpy(), p_ctl=cur["p_buy"].to_numpy(),
                     agree_ctl_share=sh["agree"].to_numpy())
        for c in cands:
            tc = _table(c.tag, f)
            if tc is None or f not in thr[c.tag]:
                missing[c.tag].append(f)
                continue
            s = _signal_rows(tc.assign(enter_long=agree_of(tc, float(thr[c.tag][f])).astype(int)), d)
            r[f"agree_{c.tag}"] = s["agree"].to_numpy()
            r[f"p_{c.tag}"] = s["p_buy"].to_numpy()
        rows.append(r)
    allr = pd.concat(rows, ignore_index=True)
    ctl = split_stats(allr["pp"], allr["agree_ctl"], allr["p_ctl"])
    ctl_sh = split_stats(allr["pp"], allr["agree_ctl_share"], allr["p_ctl"])
    refs = {"pre_registered": PRE_REGISTERED_GAP, "control_at_share": ctl_sh["gap"]}
    out = {"label": "034 S1 (pre-registered; early half only)", "folds": sorted(set(allr["fold"])),
           "control": ctl, "control_at_share": ctl_sh, "references": refs, "candidates": {}}
    for c in cands:
        if missing[c.tag]:
            out["candidates"][c.tag] = {"status": "INCONCLUSIVE",
                                        "reason": f"tables/thresholds missing {missing[c.tag]}"}
            continue
        st = split_stats(allr["pp"], allr[f"agree_{c.tag}"], allr[f"p_{c.tag}"])
        deg = degenerate(allr, f"agree_{c.tag}")
        st |= {"degenerate_folds": deg, "agree_share": round(float(allr[f"agree_{c.tag}"].mean()), 4),
               "S1_pass": s1_pass(st["gap"], refs, len(deg))}
        out["candidates"][c.tag] = st
        print(c.tag, json.dumps(st), flush=True)
    print("control", json.dumps(ctl), "\ncontrol_at_share", json.dumps(ctl_sh), flush=True)
    (HERE / "separation.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

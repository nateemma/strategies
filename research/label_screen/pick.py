"""034 T009: per-fold thresholds for the label screen, ONE fold per process.

  --role sz  the sizer's agree threshold: 033's picker (research/width_relative/pick.py) on the
             candidate's `plan_ls_<tag>_sz.json` -- first bar of each width-book candidate run in
             the fold's validation tail, the (1 - AGREE_SHARE) quantile of p_buy
             (AGREE_RULE "p_buy": no BUY-argmax condition; amendment 2026-10-14), 4 decimals, fold class loaded with ft_bot_start(). `--tag CONTROL`
             scores the control the same way (the S1 reference) and writes only to
             thresholds_control_share.json, never to a control plan.
  --role es  the entry-source threshold: evaluation/pick_threshold.py's rate pick on the fold's
             training window at the control's entry rate 0.014735 (032 Q1; how the control's own
             thresholds were picked), stored at 4 decimals in `plan_ls_<tag>_es.json`.

The sizer fold class is generated (no threshold) when missing, as run_folds would.
`--verify` re-loads the fold class with the stored threshold: for sz, every width-book entry tag
must equal (p_buy > threshold) and the loaded threshold must equal the stored one.

    PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.pick --tag FMAE --fold f00 --role sz [--verify]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.label_screen import candidates as C  # noqa: E402


HERE = Path(__file__).resolve().parent
FOLDS = STRAT / "evaluation" / "folds"
ENTRY_RATE = 0.014735
CONTROL_SHARE = HERE / "thresholds_control_share.json"


def plan_path(tag: str, role: str) -> Path:
    return FOLDS / f"plan_{C.plan_tag(C.BY_TAG[tag], role)}.json"


def check_role(tag: str, role: str) -> None:
    c = C.BY_TAG[tag]
    if role == "es" and "es" not in c.roles:
        raise SystemExit(f"{tag} has no entry-source role")
    if tag == "CONTROL" and role != "sz":
        raise SystemExit("CONTROL is picked only as the S1 sizer reference (--role sz)")


def store(tag: str, role: str, fid: str, threshold: float, extra: dict) -> Path:
    t = round(float(threshold), 4)
    if tag == "CONTROL":
        d = json.loads(CONTROL_SHARE.read_text()) if CONTROL_SHARE.is_file() else {}
        d[fid] = {"threshold": t, **extra}
        CONTROL_SHARE.write_text(json.dumps(d, indent=1, sort_keys=True))
        return CONTROL_SHARE
    p = plan_path(tag, role)
    plan = json.loads(p.read_text())
    plan.setdefault("thresholds", {})[fid] = t
    plan.setdefault("threshold_picks", {})[fid] = extra
    p.write_text(json.dumps(plan, indent=1) + "\n")
    return p


def _ensure_fold_class(plan: dict, fid: str, seed: int = 1) -> None:
    import run_folds as rf

    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    if (STRAT / "NNNC" / f"{f['strategy_class']}.py").is_file():
        return
    snap = Path(plan["params_snapshot"])
    snap = snap if snap.is_absolute() else STRAT.parents[1] / snap
    rf.generate_fold(f, plan["base_strategy"], plan["family"], json.loads(snap.read_text()),
                     threshold=(plan.get("thresholds") or {}).get(fid), seed=seed)


def p_buy(probs: np.ndarray) -> np.ndarray:
    """The sizer's agree score (AGREE_RULE "p_buy"): p_buy, NaN rows -> 0."""
    return np.nan_to_num(np.asarray(probs, dtype=float)[:, 2], nan=0.0)


def pick_sz(tag: str, fid: str) -> tuple[float, dict]:
    from research.width_relative import pick as WP

    plan = json.loads(plan_path(tag, "sz").read_text())
    _ensure_fold_class(plan, fid)
    frames, strat = WP._frames(plan, fid, 0.0)
    pb, effs = [], []
    for s in frames.values():
        rs = WP.run_starts(s["enter_long"].fillna(0).to_numpy() == 1)
        probs = s[list(WP.PROB)].to_numpy()
        pb.append(p_buy(probs)[rs])
        effs.append(WP.effective(probs)[rs])
    score = np.concatenate(pb) if pb else np.empty(0)
    eff = np.concatenate(effs) if effs else np.empty(0)
    share = float(strat.AGREE_SHARE)
    t, achieved = WP.threshold_for_share(score, share)
    return t, {"target_share": share, "achieved_share": round(achieved, 4),
               "run_starts": int(len(score)), "buy_argmax_share": round(float(np.mean(eff > 0)), 4)}


def verify_sz(tag: str, fid: str) -> bool:
    from research.width_relative import pick as WP

    plan = json.loads(plan_path(tag, "sz").read_text())
    t = (json.loads(CONTROL_SHARE.read_text())[fid]["threshold"] if tag == "CONTROL"
         else float(plan["thresholds"][fid]))
    import run_folds as rf

    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    snap = Path(plan["params_snapshot"])
    snap = snap if snap.is_absolute() else STRAT.parents[1] / snap
    rf.generate_fold(f, plan["base_strategy"], plan["family"], json.loads(snap.read_text()),
                     threshold=t)  # the class a backtest will load
    frames, strat = WP._frames(plan, fid, None)
    loaded = float(strat.prediction_threshold.value)
    bad = n = 0
    for s in frames.values():
        enter = s["enter_long"].fillna(0).to_numpy() == 1
        score = p_buy(s[list(WP.PROB)].to_numpy())
        agree = s["enter_tag"].fillna("").astype(str).str.endswith(":agree").to_numpy()
        n += int(enter.sum())
        bad += int(np.sum(enter & (agree != (score > t))))
    print(f"{tag} {fid}: verify sz threshold {t} loaded {loaded}: entry bars {n}, tag mismatches {bad}")
    return bad == 0 and abs(loaded - t) < 1e-9


def pick_es(tag: str, fid: str) -> tuple[float, dict]:
    import pick_threshold as P

    plan = json.loads(plan_path(tag, "es").read_text())
    got = P.cmd_pick(plan, fid, ENTRY_RATE)
    if fid not in got:
        raise SystemExit(f"{tag} {fid}: no trained model -- cannot pick")
    t = got[fid]
    return t, {"target_rate": ENTRY_RATE}


def verify_es(tag: str, fid: str) -> bool:
    import pick_threshold as P

    plan = json.loads(plan_path(tag, "es").read_text())
    t = float(plan["thresholds"][fid])
    rate = P.cmd_control_rate(plan, fid, t)[fid]
    ok = abs(rate - ENTRY_RATE) <= 0.05 * ENTRY_RATE
    print(f"{tag} {fid}: verify es threshold {t}: rate {rate:.5f} vs {ENTRY_RATE} "
          f"({'OK' if ok else 'OUT OF BAND'})")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(C.BY_TAG))
    ap.add_argument("--fold", required=True, help="exactly one fold id")
    ap.add_argument("--role", required=True, choices=("sz", "es"))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if "," in a.fold:
        raise SystemExit("one fold per process (a multi-fold pick was reaped for low memory)")
    check_role(a.tag, a.role)
    if a.verify:
        ok = verify_sz(a.tag, a.fold) if a.role == "sz" else verify_es(a.tag, a.fold)
        return 0 if ok else 1
    t, extra = pick_sz(a.tag, a.fold) if a.role == "sz" else pick_es(a.tag, a.fold)
    p = store(a.tag, a.role, a.fold, t, extra)
    print(f"{a.tag} {a.fold} {a.role}: threshold {round(t, 4)} {json.dumps(extra)} -> {p.name}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

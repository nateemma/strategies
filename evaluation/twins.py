"""Twin check and model-contribution compare (feature 023 US3, FR-008/FR-013/FR-014, contract C5).

A TWIN is a SimpleTrigger arm whose framework is identical to an NNNC arm's, so the ONLY thing
between them is the model. Two gates come before any number is reported, and `compare` refuses
until both have passed:

  1. FINGERPRINT (FR-013). Every fold's `framework_sha` must be equal across the NNNC arm, the
     twin and the equivalence arm. Records written before the fingerprint existed get it
     computed now from the fold's class and params JSON.               -> exit 2 on mismatch
  2. EQUIVALENCE (FR-008). The equivalence arm (`NNNC_TriggerGate`: the NNNC arm's real code
     path with the trigger as its entry) must reproduce the twin's trades exactly, fold for
     fold. Equal fingerprints say the SETTINGS match; this says the CODE PATHS do.
                                                                         -> exit 3 on mismatch

`compare` then reports the model's contribution as NNNC-arm minus twin: the paired verdict
(`compare_pooled.compare_data`, the repository's one implementation), its early/late halves, the
shared-trade fraction that limits the paired test, and the matched-count selection test, which
does not need shared trades. Every number is PROVISIONAL under quality gate 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STRAT = ROOT / "user_data/strategies"
sys.path.insert(0, str(STRAT))

from evaluation import bootstrap  # noqa: E402


bootstrap.BT_RESULTS = ROOT / "user_data/backtest_results"  # cwd-independent archive path

from evaluation.compare_pooled import compare_data  # noqa: E402
from evaluation.halves import PROVISIONAL, split  # noqa: E402
from evaluation.matched_count import matched_count_pooled  # noqa: E402
from evaluation.pool import pool  # noqa: E402


ARMS = STRAT / "regime/arms"
OUT = STRAT / "evaluation/twins"
DIRECTIONAL = ("IMPROVED", "WORSENED", "UNCHANGED")


def _record(tag: str, fid: str) -> dict | None:
    p = ARMS / f"{tag}_{fid}_{fid.upper()}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _sha(rec: dict, config: str) -> str | None:
    if rec.get("framework_sha"):
        return rec["framework_sha"]
    from evaluation.fingerprint import resolve_by_name

    return resolve_by_name(rec["strategy"], ROOT / config)[0]


def check(plan: dict, nnnc_tag: str, simple_tag: str, equivalence_tag: str) -> tuple[int, dict]:
    folds = [f["fold_id"] for f in plan["folds"]]
    rows, sha_bad, eq_bad, missing = [], [], [], []
    for fid in folds:
        recs = {
            k: _record(t, fid)
            for k, t in (
                ("nnnc", nnnc_tag),
                ("simple", simple_tag),
                ("equivalence", equivalence_tag),
            )
        }
        absent = [k for k, r in recs.items() if not r or r.get("returncode") != 0 or r.get("error")]
        if absent:
            missing.append({"fold": fid, "absent_or_failed": absent})
            continue
        shas = {k: _sha(r, plan["config"]) for k, r in recs.items()}
        hashes = {k: r.get("trades_hash") for k, r in recs.items()}
        row = {
            "fold": fid,
            "framework_sha": shas,
            "trades_hash": hashes,
            "n_trades": {k: r.get("n_trades") for k, r in recs.items()},
        }
        if len(set(shas.values())) != 1:
            sha_bad.append(fid)
        if hashes["equivalence"] != hashes["simple"]:
            eq_bad.append(fid)
        rows.append(row)
    if missing:
        code, verdict = 4, "INCOMPLETE"
    elif sha_bad:
        code, verdict = 2, "FINGERPRINT_MISMATCH"
    elif eq_bad:
        code, verdict = 3, "EQUIVALENCE_FAILED"
    else:
        code, verdict = 0, "PASS"
    return code, {
        "verdict": verdict,
        "passed": code == 0,
        "nnnc_tag": nnnc_tag,
        "simple_tag": simple_tag,
        "equivalence_tag": equivalence_tag,
        "fingerprint_mismatch_folds": sha_bad,
        "equivalence_mismatch_folds": eq_bad,
        "missing": missing,
        "folds": rows,
    }


def compare(nnnc_plan: dict, simple_plan: dict, check_artifact: dict | None) -> dict:
    if not check_artifact or not check_artifact.get("passed"):
        raise PermissionError(
            "twin compare refused: no passing `twins check` artefact -- the "
            "frameworks are not proven identical, so a delta would mix model "
            "and framework effects"
        )
    nn, st = pool(nnnc_plan), pool(simple_plan)
    d = compare_data(nn, st)
    bs = d.get("bootstrap") or {}
    shared = bs.get("shared", 0)
    out = {
        "label": PROVISIONAL,
        "arm": "NNNC (model)",
        "control": "SimpleTrigger twin (no model)",
        "nnnc": {
            "trades": nn["n_trades_pooled"],
            "status": nn["status"],
            "sum_pp": sum(f["return_pct"] or 0 for f in nn["per_fold"]),
            "positive_folds": sum(1 for f in nn["per_fold"] if (f["return_pct"] or 0) > 0),
        },
        "twin": {
            "trades": st["n_trades_pooled"],
            "status": st["status"],
            "sum_pp": sum(f["return_pct"] or 0 for f in st["per_fold"]),
            "positive_folds": sum(1 for f in st["per_fold"] if (f["return_pct"] or 0) > 0),
        },
        "verdict": d["verdict"],
        "directional_verdict": d["verdict"] if d["verdict"] in DIRECTIONAL else None,
        "delta_pp": d.get("delta_pp"),
        "delta_pp_yr": d.get("delta_pp_yr"),
        "ci_pp_yr": d.get("ci_pp_yr"),
        "p": bs.get("p"),
        "shared_trades": shared,
        "shared_fraction_of_nnnc": shared / nn["n_trades_pooled"]
        if nn["n_trades_pooled"]
        else None,
        "shared_fraction_of_twin": shared / st["n_trades_pooled"]
        if st["n_trades_pooled"]
        else None,
        "halves": split(nn, st),
        "matched_count": matched_count_pooled(nn, st),
    }
    return out


def _side_line(label: str, side: dict) -> str:
    return (
        f"- {label}: {side['trades']} trades, {side['sum_pp']:+.2f}pp per-fold sum, "
        f"{side['positive_folds']} positive folds"
    )


def _md(res: dict) -> str:
    verdict = f"- Paired verdict: **{res['verdict']}**"
    if res.get("ci_pp_yr"):
        lo, hi = res["ci_pp_yr"]
        verdict += (
            f" (delta {res['delta_pp']:+.2f}pp, CI {lo:+.2f}..{hi:+.2f}pp/yr, p={res['p']:.3f})"
        )
    fn = res["shared_fraction_of_nnnc"] or 0
    ft = res["shared_fraction_of_twin"] or 0
    lines = [
        f"# Model contribution: {res['arm']} vs {res['control']}",
        "",
        f"**{res['label']}**",
        "",
        _side_line("NNNC", res["nnnc"]),
        _side_line("Twin", res["twin"]),
        verdict,
        f"- Shared trades: {res['shared_trades']} ({fn:.1%} of NNNC, {ft:.1%} of twin)",
        "",
        "| matched count | n | observed pp | null pp | z | p |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("pooled", "early", "late"):
        r = res["matched_count"].get(name)
        if r:
            null = f"{r['null_mean_pp']:+.2f} ± {r['null_sd_pp']:.2f}"
            lines.append(
                f"| {name} | {r['n_selected']} | {r['observed_pp']:+.2f} | {null} | "
                f"{r['z']:+.2f} | {r['p_one_sided']:.5f} |"
            )
    h = res["halves"]
    lines += ["", "| half | NNNC pp | twin pp | paired verdict |", "|---|---|---|---|"]
    for name in ("early", "late"):
        a, c = h[name]["arm"], h[name]["ctl"]
        lines.append(
            f"| {name} | {a['sum_pp']:+.2f} ({a['positive_folds']}/{a['folds']}) | "
            f"{c['sum_pp']:+.2f} ({c['positive_folds']}/{c['folds']}) | {h[name]['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--plan", required=True, help="the NNNC arm's fold plan (fold ids, config)")
    c.add_argument("--nnnc-tag", required=True)
    c.add_argument("--simple-tag", required=True)
    c.add_argument("--equivalence-tag", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--nnnc-plan", required=True)
    m.add_argument("--simple-plan", required=True)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "check":
        plan = json.loads(Path(a.plan).read_text())
        code, res = check(plan, a.nnnc_tag, a.simple_tag, a.equivalence_tag)
        (OUT / f"{a.simple_tag}.check.json").write_text(json.dumps(res, indent=2) + "\n")
        print(f"twins check: {res['verdict']}  (exit {code})")
        for k in ("fingerprint_mismatch_folds", "equivalence_mismatch_folds", "missing"):
            if res[k]:
                print(f"  {k}: {res[k]}")
        return code
    nnnc_plan = json.loads(Path(a.nnnc_plan).read_text())
    simple_plan = json.loads(Path(a.simple_plan).read_text())
    simple_tag = simple_plan["folds"][0]["arm_tag"].rsplit("_", 1)[0]
    ca = OUT / f"{simple_tag}.check.json"
    try:
        res = compare(nnnc_plan, simple_plan, json.loads(ca.read_text()) if ca.is_file() else None)
    except PermissionError as exc:
        print(f"REFUSED: {exc}")
        return 5
    (OUT / f"{simple_tag}.compare.json").write_text(json.dumps(res, indent=2) + "\n")
    (OUT / f"{simple_tag}.compare.md").write_text(_md(res))
    print(_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())

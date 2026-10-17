"""037 report: research/indicator_screen/report.md from every screen output (FR-022, FR-025, SC-004).

Deterministic: reads only committed JSON outputs, sorts everything, prints fixed precision.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = 15
SCREEN = ("**SCREEN (descriptive).** Association with a label or a regime decides nothing about "
          "acceptance (Principle III). Every candidate below needs a backtest in a later feature "
          "before any P&L claim (FR-025).")


def _load(name, d=HERE):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def _f(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def _persist(flags):
    e = flags.get("e0_agrees")
    return "E0 agrees" if e is True else "E0 disagrees" if e is False else "distant era not tested"


def _names(items):
    return ", ".join(f"`{x}`" for x in items) if items else "none"


def _top_table(results, target, pop, key="median_strength", n=TOP):
    rows = sorted((r for r in results if r["target"] == target and r["population"] == pop
                   and r.get(key) is not None), key=lambda r: (-r[key], r["column"]))[:n]
    out = ["| column | median AUC | early | late | E0 | same-dir share | group | ρ guard | ρ width |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        h = r["halves"]
        rg, rw = r.get("rho_guard") or {}, r.get("rho_width") or {}
        out.append(f"| `{r['column']}` | {_f(r['median_auc'])} | {_f(h['early']['median_auc'])} | "
                   f"{_f(h['late']['median_auc'])} | {_f(h['e0']['median_auc'])} | "
                   f"{_f(r['same_direction_share'], 2)} | {r.get('redundancy_group', '–')} | "
                   f"{_f(rg.get(pop), 2)} | {_f(rw.get(pop), 2)} |")
    return out


def grouped(cols, groups: dict) -> list[str]:
    """One entry per redundancy group (|rho| >= 0.8): 'representative (+k in group)'."""
    by = {}
    for c in sorted(cols):
        g = groups.get(c[4:] if c.startswith("btc:") else c)
        by.setdefault((g, c.startswith("btc:")) if g is not None else c, []).append(c)
    out = []
    for members in by.values():
        rep = min(members, key=lambda c: (c.count(":") > 1 and not c.startswith("btc:"), len(c), c))
        out.append(rep + (f" (+{len(members) - 1} in group)" if len(members) > 1 else ""))
    return sorted(out)


def sc004(single_labels, single_regime, pairs, models) -> dict:
    """The SC-004 answer lists. A flag counts only with E0 agreement (Principle II)."""
    res = (single_labels or {}).get("results", [])
    groups = {r["column"]: r.get("redundancy_group") for r in res}
    a = sorted({r["column"] for r in res if r["population"] == "candidate"
                and r["candidate_flags"].get("beats_width_both_halves") and r["candidate_flags"].get("e0_agrees")})
    b = sorted({r["column"] for r in res if r["population"] == "candidate"
                and r["candidate_flags"].get("residual_both_halves") and r["candidate_flags"].get("e0_agrees")})
    c, c_partial = {}, {}
    for t, p in sorted(pairs.items()):
        for pop, blk in sorted(p["populations"].items()):
            for s in blk["survivors"]:
                key, tag = f"{t} [{pop}]", f"{s['a']} AND {s['b']}"
                if s.get("confirmed") == "yes":
                    c.setdefault(key, []).append(tag)
                elif s.get("confirmed") == "early+late only":
                    c_partial.setdefault(key, []).append(tag)
    d = []
    for t, m in sorted(models.items()):
        for pop, blk in sorted(m["populations"].items()):
            for model in ("tree", "hgb"):
                if blk[model]["candidate"]:
                    d.append(f"{model} [{t}, {pop}] (distant era not tested)")
    e = {}
    for r in (single_regime or {}).get("results", []):
        fl = r["candidate_flags"]
        if fl.get("beats_baseline_both_halves") and fl.get("e0_agrees") is not False:
            key = f"{r['target']} [{r['population']}]"
            e.setdefault(key, []).append(r["column"] + ("" if fl.get("e0_agrees") else " (distant era not tested)"))
    for t, m in sorted(models.items()):
        if t.startswith("label:"):
            continue
        for pop, blk in sorted(m["populations"].items()):
            for model in ("tree", "hgb"):
                if blk[model]["candidate"]:
                    e.setdefault(f"{t} [{pop}]", []).append(f"{model} model (distant era not tested)")
    return {"a": grouped(a, groups), "b": grouped(b, groups), "c": c, "c_partial": c_partial, "d": d,
            "e": {k: grouped(v, groups) for k, v in sorted(e.items())},
            "n": {"a": len(a), "b": len(b)}}


def build(d: Path = HERE) -> str:
    man = _load("manifest.json", d) or {}
    caus = _load("causality.json", d) or {}
    sl = _load("single_labels.json", d)
    sr = _load("single_regime.json", d)
    pairs = {p.stem[len("pairs_"):].replace("_", ":", 1): json.loads(p.read_text())
             for p in sorted(d.glob("pairs_*.json"))}
    models = {p.stem[len("models_"):].replace("_", ":", 1): json.loads(p.read_text())
              for p in sorted(d.glob("models_*.json"))}
    pre = _load("preregistration.json", d) or {}
    L = ["# 037 Indicator screen: report", "", SCREEN, ""]
    L += ["## Conduct", "",
          f"- Pre-registration lock: `{(sl or {}).get('prereg_lock') or 'not yet (no late window scored)'}`.",
          f"- Windows scored (labels): {', '.join((sl or {}).get('windows_scored', [])) or 'none'}.",
          "- Populations: all bars, and gbb candidate bars; the 10 pairs pooled per window; rows with "
          "`meta:data_ok` only (XRP's gap-filled 2021-01 → 2023-07 bars are not data).",
          "- Persistence: early (2023–24) and late (2025–26) are adjacent, so E0 (2021-04 → 2022-12) must "
          "agree for a single column or pair to count. Models and A2 cannot be scored on E0: "
          "\"distant era not tested\"; a later backtest must cover 2022.", ""]
    if pre.get("instrument", {}).get("amendment"):
        L += [f"- Instrument amendment: {pre['instrument']['amendment']}", ""]
    L += ["## Harvest", "",
          f"- In-scope classes: {len(man.get('classes', []))}; columns: {len(man.get('columns', []))}.",
          "- Excluded classes: " + _names(sorted({f"{e['class']} ({e['reason']})" for e in man.get("excluded", [])})),
          f"- Lookahead (truncation test) FAIL, excluded: {_names(caus.get('fails', []))}."]
    never = sorted({c for v in man.get("excluded_columns", {}).values() for c, why in v.items() if why == "constant"})
    L += [f"- Constant on some pair (never fires / never varies): {_names(never)}.", ""]
    if sl is None:
        L += ["## Labels", "", "Not run.", ""]
    elif sl.get("instrument") == "FAIL":
        L += ["## Labels", "", "**INSTRUMENT FAIL** -- no findings reported.", "",
              "```", json.dumps(sl.get("detail"), indent=1), "```", ""]
    else:
        ins = sl["instrument"]
        L += ["## Instrument check (SC-002)", "", f"- Gate ({'+'.join(ins['halves'])}): **{ins['verdict']}** ({ins['rule']})."]
        for c, v in sorted(ins["columns"].items()):
            L.append(f"  - `{c}`: pass={v['pass']}, rank {v['rank']}, AUC by window "
                     + ", ".join(f"{w} {_f(a)}" for w, a in sorted(v["auc_by_window"].items())))
        if sl.get("instrument_confirm"):
            L.append(f"- Confirmation over the 15 test windows: **{sl['instrument_confirm']['verdict']}**.")
        L.append("")
        for t in ("label:buy", "label:sell"):
            for pop in ("all", "candidate"):
                L += [f"### {t}, {pop} bars: top {TOP} by median strength", ""]
                L += _top_table(sl["results"], t, pop) + [""]
        L += ["### Residual beyond bb_width + guard_metric (label:buy, candidate bars)", "",
              "| column | strat AUC early | late | E0 | AUC | width-sort AUC | persistence |", "|---|---|---|---|---|---|---|"]
        rr = sorted((r for r in sl["results"] if r["target"] == "label:buy" and r["population"] == "candidate"
                     and r["halves"]["early"]["median_strat_auc"] is not None),
                    key=lambda r: (-abs(r["halves"]["early"]["median_strat_auc"] - 0.5), r["column"]))[:TOP]
        for r in rr:
            h = r["halves"]
            L.append(f"| `{r['column']}` | {_f(h['early']['median_strat_auc'])} | {_f(h['late']['median_strat_auc'])} | "
                     f"{_f(h['e0']['median_strat_auc'])} | {_f(r['median_auc'])} | {_f(r['width_sort_auc'])} | "
                     f"{_persist(r['candidate_flags'])} |")
        L.append("")
    L += ["## Signal pairs (both must fire)", ""]
    if not pairs:
        L += ["Not run.", ""]
    for t, p in sorted(pairs.items()):
        for pop, blk in sorted(p["populations"].items()):
            L += [f"### {t}, {pop} bars: {blk['n_pairs']} pairs, {blk['n_passing_rule']} pass the rule", ""]
            if not blk["survivors"]:
                L += ["No survivors.", ""]
                continue
            L += ["| pair | early lift | components early | late lift | E0 lift | confirmed |", "|---|---|---|---|---|---|"]
            for s in blk["survivors"]:
                L.append(f"| `{s['a']}` AND `{s['b']}` | {_f(s['early']['median_lift'], 2)} | "
                         f"{_f(s['lift_a_early'], 2)} / {_f(s['lift_b_early'], 2)} | "
                         f"{_f((s.get('late') or {}).get('median_lift'), 2)} | {_f(s.get('e0_lift'), 2)} | "
                         f"{s.get('confirmed', 'late not scored')} |")
            L.append("")
    L += ["## Multivariate models (per fold; full vs bb_width + guard_metric)", ""]
    if not models:
        L += ["Not run.", ""]
    for t, m in sorted(models.items()):
        for pop, blk in sorted(m["populations"].items()):
            L += [f"### {t}, {pop} bars (computable folds: {len(blk['computable_folds'])})", "",
                  "| model | early all | early restricted | late all | late restricted | width sort (early/late) | ARO (early/late) | candidate |",
                  "|---|---|---|---|---|---|---|---|"]
            for model in ("tree", "hgb"):
                h = blk[model]["halves"]
                L.append(f"| {model} | {_f(h['early']['median_auc_all'])} | {_f(h['early']['median_auc_restricted'])} | "
                         f"{_f(h['late']['median_auc_all'])} | {_f(h['late']['median_auc_restricted'])} | "
                         f"{_f(h['early']['median_width_sort_auc'])} / {_f(h['late']['median_width_sort_auc'])} | "
                         f"{_f(h['early']['median_aro_auc'])} / {_f(h['late']['median_aro_auc'])} | {blk[model]['candidate']} |")
            sp = blk["tree"]["recurring_top_splits"]
            L += ["", "Recurring tree splits: " + (", ".join(f"`{c}` ×{n}" for c, n in sp) or "none"), ""]
    L += ["## Regime screen", ""]
    if sr is None:
        L += ["Not run.", ""]
    else:
        L += ["Regime targets at 3 d / 7 d are heavily autocorrelated: one 90-day window holds only ~13–30 "
              "independent periods, so window medians and the same-direction share are the evidence, not "
              "row counts. `btc:` columns against market-wide targets are scored one row per timestamp.", ""]
        for t in sr["targets"]:
            rs = [r for r in sr["results"] if r["target"] == t and r["population"] == "all"]
            if not rs:
                continue
            bl = rs[0].get("baseline_auc")
            L += [f"### {t} (all bars): persistence baseline AUC {_f(bl)}", ""]
            L += _top_table(sr["results"], t, "all", n=8) + [""]
    ans = sc004(sl, sr, pairs, models)
    L += ["## SC-004 answers", "",
          f"**(a) Columns that beat the width sort on candidate bars in both halves (E0 agrees):** "
          f"{ans['n']['a']} columns in {len(ans['a'])} redundancy groups: " + _names(ans["a"]), "",
          f"**(b) Columns with residual strength beyond bb_width + guard_metric in both halves (E0 agrees):** "
          f"{ans['n']['b']} columns in {len(ans['b'])} redundancy groups: " + _names(ans["b"]), "",
          "**(c) Signal pairs that survive late confirmation (E0 agrees), per target:**"]
    L += [f"  - {k}: {_names(v)}" for k, v in ans["c"].items()] or ["  none"]
    L += ["", "  Early + late only (E0 fails or not computable): "
          + ("; ".join(f"{k}: {len(v)}" for k, v in ans["c_partial"].items()) or "none"), "",
          "**(d) Multivariate models that beat the restricted model in both halves:** " + _names(ans["d"]), "",
          "**(e) Regime columns/models that beat the persistence baseline in both halves:**"]
    if ans["e"]:
        for k, v in ans["e"].items():
            L.append(f"  - {k}: {_names(v)}")
    else:
        L.append("  none")
    L += ["", SCREEN, ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(HERE / "report.md"))
    a = ap.parse_args(argv)
    Path(a.out).write_text(build())
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

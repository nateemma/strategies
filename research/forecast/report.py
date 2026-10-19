"""038 report (T033): research/forecast/report.md from baselines.json, scores.json, verdicts.json,
instrument.json and truncation.json.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.report
"""
from __future__ import annotations

import json

from research.forecast import common as C

NET = f"mean_net_{C.VERDICT_BPS}"


def _load(name: str) -> dict:
    p = C.HERE / name
    return json.loads(p.read_text()) if p.exists() else {}


def bp(x, d=1) -> str:
    return "–" if x is None else f"{x * 1e4:+.{d}f}"


def num(x, d=4) -> str:
    return "–" if x is None else f"{x:+.{d}f}"


def main(argv=None) -> int:
    base, scores, ver = _load("baselines.json"), _load("scores.json"), _load("verdicts.json")
    inst, trunc = _load("instrument.json"), _load("truncation.json")
    L = ["# 038 Close-Price Forecast Study — Report", "",
         "Spec: `specs/038-close-forecast-study/`. Rule: `research/forecast/preregistration.json` "
         "(committed before any model result). Returns are log returns; bp = basis points. Trades are "
         "long only, entered at the next open and held N candles, one per pair at a time, net of 0.10% fee "
         f"+ {C.VERDICT_BPS:g} bp slippage per side ({C.cost_rt(C.VERDICT_BPS) * 1e4:.0f} bp round trip). "
         "Every PASS is PROVISIONAL: no distant-era data exists, so the early (f00–f07) and late (f08–f14) "
         "halves are the persistence test.", ""]

    L += ["## Summary", ""]
    if ver:
        L += [f"- Most viable horizon: **{ver.get('most_viable_n') or 'none'}**"
              + (f" ({ver['most_viable_method']})" if ver.get("most_viable_method") else ""),
              f"- Break-even horizon (any method net > 0 in both halves at {C.VERDICT_BPS:g} bp): "
              f"**{ver.get('break_even_n') or 'none'}**",
              f"- Null-control passes: {ver.get('null_passes') or 'none'}", ""]
        tally = {}
        for m, cs in ver["cells"].items():
            for c in cs.values():
                tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
        L += ["- Verdict counts: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())), ""]

    L += ["## Instrument", ""]
    for k, v in inst.items():
        if isinstance(v, dict):
            L.append(f"- {k}: {'PASS' if v.get('pass') else 'FAIL' if v.get('pass') is False else 'n/a'} "
                     f"— {json.dumps({x: y for x, y in v.items() if x != 'pass'})[:220]}")
    if trunc:
        bad = [c for c, ok in {**trunc.get("inputs", {}), **trunc.get("L_inputs", {})}.items() if not ok]
        L.append(f"- Truncation (FR-012): {'all inputs identical up to every cut' if not bad else 'FAILED: ' + ', '.join(bad)}")
    L.append("")

    if base:
        L += ["## Horizon and cost line (pooled, eligible candles, early + late)", "",
              "| N | hours | median abs move (bp) | move / cost @0 | @2.5 | @5 | @10 | |",
              "|---|---|---|---|---|---|---|---|"]
        for n in C.HORIZONS:
            r = base["horizons"]["pooled"][str(n)]
            L.append(f"| {n} | {n / 4:g} | {r['typical_move'] * 1e4:.1f} | "
                     + " | ".join(f"{r[f'move_to_cost_{s}']:.2f}" for s in C.SLIPPAGE_BPS)
                     + f" | {'untradeable' if r['untradeable'] else ''} |")
        L.append("")

    if ver:
        methods = list(ver["cells"])
        L += ["## Verdicts", "", "| method | " + " | ".join(f"N={n}" for n in C.HORIZONS) + " |",
              "|---|" + "---|" * len(C.HORIZONS)]
        post = set(ver.get("post_hoc_methods", []))
        if post:
            L += ["Methods marked † were added post-hoc (`preregistration_harmonic.json`, same rule).", ""]
        for m in methods:
            L.append(f"| {m}{' †' if m in post else ''} | " + " | ".join(ver["cells"][m].get(str(n), {}).get("verdict", "–")
                                               .replace(" (PROVISIONAL)", "") for n in C.HORIZONS) + " |")
        L.append("")

    if scores:
        for title, key, fmt in (("Mean net return per trade, bp (early / late)", NET, bp),
                                ("Edge over random entry, bp (early / late)", "edge_vs_rand", bp)):
            L += [f"## {title}", "", "| method | " + " | ".join(f"N={n}" for n in C.HORIZONS) + " |",
                  "|---|" + "---|" * len(C.HORIZONS)]
            for m, s in sorted(scores.items()):
                L.append(f"| {m} | " + " | ".join(
                    f"{fmt(s['n'][str(n)]['early'].get(key))} / {fmt(s['n'][str(n)]['late'].get(key))}"
                    for n in C.HORIZONS) + " |")
            L.append("")
        L += ["## Trades (early / late) and late lower bound (bp)", "",
              "| method | " + " | ".join(f"N={n}" for n in C.HORIZONS) + " |", "|---|" + "---|" * len(C.HORIZONS)]
        for m, s in sorted(scores.items()):
            L.append(f"| {m} | " + " | ".join(
                f"{s['n'][str(n)]['early']['n_trades']}/{s['n'][str(n)]['late']['n_trades']} "
                f"({bp(s['n'][str(n)]['late'].get('net_lb95'))})" for n in C.HORIZONS) + " |")
        L.append("")
        L += ["## Descriptive (late half, traded candles): R² vs zero / median IC / hit rate", "",
              "Reported, never decisive (FR-010).", "",
              "| method | " + " | ".join(f"N={n}" for n in C.HORIZONS) + " |", "|---|" + "---|" * len(C.HORIZONS)]
        for m, s in sorted(scores.items()):
            L.append(f"| {m} | " + " | ".join(
                f"{num(s['n'][str(n)]['late'].get('r2_oos_traded'), 4)} / "
                f"{num(s['n'][str(n)]['late'].get('ic_median_traded'), 3)} / "
                f"{num(s['n'][str(n)]['late'].get('hit_rate_traded'), 3)}" for n in C.HORIZONS) + " |")
        L.append("")
        L += ["## Short side (descriptive; not tradeable on spot): late mean net, bp", "",
              "| method | " + " | ".join(f"N={n}" for n in C.HORIZONS) + " |", "|---|" + "---|" * len(C.HORIZONS)]
        for m, s in sorted(scores.items()):
            L.append(f"| {m} | " + " | ".join(bp(s["n"][str(n)]["late"].get(f"short_mean_net_{C.VERDICT_BPS}"))
                                               for n in C.HORIZONS) + " |")
        L.append("")
        L += ["## Forward window 2026-09-07 → 2026-10-18 (descriptive): trades / mean net bp", "",
              "| method | " + " | ".join(f"N={n}" for n in C.HORIZONS) + " |", "|---|" + "---|" * len(C.HORIZONS)]
        for m, s in sorted(scores.items()):
            L.append(f"| {m} | " + " | ".join(
                f"{s['n'][str(n)]['fwd']['n_trades']} / {bp(s['n'][str(n)]['fwd'].get(NET))}" for n in C.HORIZONS) + " |")
        L.append("")
        L += ["## Per pair, late half: mean net bp at N=16 (trades)", "",
              "| method | " + " | ".join(p.split("/")[0] for p in C.pairs()) + " |", "|---|" + "---|" * len(C.pairs())]
        for m, s in sorted(scores.items()):
            pp = s["n"]["16"]["late"].get("pairs", {})
            L.append(f"| {m} | " + " | ".join(
                f"{bp(pp.get(p, {}).get(NET), 0)} ({pp.get(p, {}).get('n_trades', 0)})" for p in C.pairs()) + " |")
        L.append("")

    if scores:
        rows = []
        for m, v in scores.items():
            if m == "lgb_null":
                continue
            for n, c in v["n"].items():
                e, l = c["early"], c["late"]
                if e["n_trades"] and l["n_trades"]:
                    rows.append((min(e["mean_gross"], l["mean_gross"]), m, n, e, l))
        rows.sort(key=lambda r: r[0], reverse=True)
        L += ["## Closest to viable (descriptive, not a verdict)", "",
              "Ranked by the worse half's gross return per trade. Round-trip cost: "
              + ", ".join(f"{C.cost_rt(s_) * 1e4:.0f} bp at {s_:g} bp slippage" for s_ in C.SLIPPAGE_BPS) + ".", "",
              "| method | N | gross early / late (bp) | edge vs random early / late (bp) | net @0 bp early / late | trades e / l |",
              "|---|---|---|---|---|---|"]
        for g, m, n, e, l in rows[:10]:
            L.append(f"| {m} | {n} | {bp(e['mean_gross'])} / {bp(l['mean_gross'])} | {bp(e['edge_vs_rand'])} / "
                     f"{bp(l['edge_vs_rand'])} | {bp(e.get('mean_net_0.0'))} / {bp(l.get('mean_net_0.0'))} | "
                     f"{e['n_trades']} / {l['n_trades']} |")
        L.append("")
        if "lgb_LW" in scores and "lgb_L" in scores:
            L += ["## Decomposition inputs: LW − L (paired candles), mean net and edge, bp (early / late)", "",
                  "| N | Δ net | Δ edge |", "|---|---|---|"]
            for n in C.HORIZONS:
                w, b_ = scores["lgb_LW"]["n"][str(n)], scores["lgb_L"]["n"][str(n)]
                d = lambda k, h: (w[h][k] - b_[h][k]) if w[h][k] is not None and b_[h][k] is not None else None  # noqa: E731
                L.append(f"| {n} | {bp(d(NET, 'early'))} / {bp(d(NET, 'late'))} | "
                         f"{bp(d('edge_vs_rand', 'early'))} / {bp(d('edge_vs_rand', 'late'))} |")
            L.append("")

    vc = _load("vote_correlations.json")
    if vc:
        L += ["## N-of-M votes: member correlation (late half, Spearman, mean off-diagonal)", "",
              "Votes add information only when members disagree; a mean near 1 means the vote behaves like "
              "one member with a stricter threshold (`preregistration_vote.json`).", "",
              "| vote | " + " | ".join(f"N={n}" for n in C.HORIZONS) + " |", "|---|" + "---|" * len(C.HORIZONS)]
        for v, c in sorted(vc.items()):
            L.append(f"| {v} | " + " | ".join(f"{c[str(n)]['mean_offdiag']:.2f}" for n in C.HORIZONS) + " |")
        L.append("")

    L += ["## Limitations", "",
          "- No distant era (binance.com data is empty): every PASS is PROVISIONAL (constitution gate 2).",
          "- Trade size against candle volume is not modelled; slippage stands in for size (plan, Complexity "
          "Tracking). Capacity belongs to the follow-up filter/sizer feature and the 036 cost model.",
          "- Statistical models use a 64-candle window only (no window search, Principle VIII), scored every "
          "16th candle.", ""]
    (C.HERE / "report.md").write_text("\n".join(L))
    print(f"wrote {C.HERE / 'report.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

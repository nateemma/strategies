"""Feasibility gate for target-aware exits (feature 027 US1, contract C1, research R2/R3).

No backtests. On the EXISTING trades of the two reference arms -- momentum (`st_bgo_mx`) and gated
reversion (`st_book_rev2`) -- and their pairs' 15m price paths, this measures how today's exit
treats trades relative to the label target (a >= 2% gross touch within 48 bars):

  (a) stop exits whose price touched the target AFTER the stop, within the horizon / max hold;
  (b) label-positive entries the stop breached before the touch ("killed winners"), and the
      drawdown winners take on the way (MAE before touch);
  (c) time to target; trades still open at the horizon without a touch;
  (d) trades that touched the target during their life but exited below it.

Then it replays every trade under each candidate coupling C1-C5 (`paths.replay`), chooses each
coupling's value on f00-f07 ONLY, and applies the pre-declared support rules (research R3) in BOTH
halves. Replay deltas are against the NULL replay (same simulator, no coupling), so simulator error
does not masquerade as a coupling effect -- and the null replay must first reproduce >= 90% of
recorded exits (reason AND bar) per family, or no verdict is given (research R2, fidelity gate).

    python -m research.exit_coupling.feasibility --momentum st_bgo_mx --reversion st_book_rev2 \\
        --out specs/027-target-aware-exits/feasibility.md

Exit: 0 at least one SUPPORTED, 3 none (STOP), 2 missing input or a failed fidelity gate.
Replays ignore re-sequencing (023 Addendum 7): they choose what to test; walk-forward arms decide.
PROVISIONAL under quality gate 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import paths as P  # noqa: E402


EARLY = frozenset(f"f{i:02d}" for i in range(8))
LABEL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"
TARGET_GAIN, HORIZON_BARS = 0.02, 48  # declared targets (research R1)
FIDELITY_MIN = 0.90
C1_BOUND = -0.10
STOP_REASONS = {"stop_loss", "trailing_stop_loss", "stoploss_on_exchange"}

#: Pre-declared grids (research R3). C1 / C4 values are percentiles turned into depths on early
#: data.
GRID = {
    "C1": (80, 90),
    "C3": ("close", "breakeven"),
    "C4": ("base", "p95"),
    "C5": (0.0, 0.5),
}
#: Descriptive thresholds of the support rules (research R3).
C1_KILLED_MIN = 0.15
C2_TERCILE_RATIO = 0.5
C4_KILLED_IN_WINDOW_SHARE = 0.50
C4_KILLED_IN_WINDOW_OF_LP = 0.10


# ------------------------------------------------------------------------------- pure decisions


def half_of(fold: str) -> str:
    return "early" if fold in EARLY else "late"


def choose(df: pd.DataFrame, coupling: str, values) -> object:
    """The grid value with the highest summed replay delta over EARLY rows only (ties: first)."""
    early = df[df["half"] == "early"]
    sums = [float(early[f"d_{coupling}_{v}"].sum()) for v in values]
    return values[int(np.argmax(sums))]


def gains_by_half(df: pd.DataFrame, coupling: str, value) -> dict:
    col = f"d_{coupling}_{value}"
    return {h: float(df.loc[df["half"] == h, col].sum()) for h in ("early", "late")}


def supported(conditions: dict) -> bool:
    """`conditions` = {half: [bool, ...]}; SUPPORTED only if every condition holds in BOTH
    halves."""
    return all(all(conditions[h]) for h in ("early", "late"))


def exit_code(inputs_ok: bool, fidelity_ok: bool, verdicts: dict) -> int:
    if not inputs_ok or not fidelity_ok:
        return 2
    return 0 if any(v.get("verdict") == "SUPPORTED" for v in verdicts.values()) else 3


# ---------------------------------------------------------------------------------- summaries


def summarise(df: pd.DataFrame) -> dict:
    """(a)-(d) for one family x half. Ambiguous rows are excluded from (a)/(b) and counted."""
    amb = df["ambiguous"]
    clean = df[~amb]
    lp = clean[clean["touch_h"].notna()]
    stops = clean[clean["rec_reason"] == "stop"]
    killed = lp[lp["breach"].notna() & (lp["breach"] < lp["touch_h"])]
    killed_window = killed[killed["breach_after_grace"]]
    depth = -lp["mae"]
    past = df[(df["rec_idx"] - df["e"] >= HORIZON_BARS) & df["touch_h"].isna()]
    touched_life = df[df["touch_m"].notna() & (df["touch_m"] <= df["rec_idx"])]
    below = touched_life[touched_life["rec_profit"] < touched_life["target_net"]]
    q = lambda s, p: None if s.empty else float(np.percentile(s, p))  # noqa: E731
    return {
        "trades": len(df),
        "ambiguous": int(amb.sum()),
        "label_positive": len(lp),
        "a_stop_exits": len(stops),
        "a_touch_after_stop_h": _share((stops["touch_h_after"]).sum(), len(stops)),
        "a_touch_after_stop_maxhold": _share((stops["touch_m_after"]).sum(), len(stops)),
        "b_killed_share": _share(len(killed), len(lp)),
        "b_killed_after_grace_share_of_killed": _share(len(killed_window), len(killed)),
        "b_killed_after_grace_share_of_lp": _share(len(killed_window), len(lp)),
        "b_mae_depth_p50_p80_p90_p95": [q(depth, p) for p in (50, 80, 90, 95)],
        "b_stop_at_touch_median": q(lp["stop_at_touch"].dropna(), 50),
        "c_bars_to_touch_p25_p50_p75": [q(lp["touch_h"] - lp["e"], p) for p in (25, 50, 75)],
        "c_open_past_horizon_no_touch": len(past),
        "c_open_past_horizon_pp": float(past["rec_pp"].sum()),
        "d_touched_exited_below": len(below),
        "d_giveback_mean": None if below.empty else float((below["target_net"]
                                                            - below["rec_profit"]).mean()),
        "pp_recorded": float(df["rec_pp"].sum()),
    }


def _share(n, d) -> float | None:
    return None if not d else float(n) / float(d)


def tercile_hit_ratio(df: pd.DataFrame, cuts: tuple[float, float]) -> float | None:
    """Target hit-rate (touch within horizon) in the lowest entry-ATR tercile / the highest."""
    lo = df[df["atr_signal"] <= cuts[0]]
    hi = df[df["atr_signal"] > cuts[1]]
    if lo.empty or hi.empty:
        return None
    hr_hi = hi["touch_h"].notna().mean()
    return None if hr_hi == 0 else float(lo["touch_h"].notna().mean() / hr_hi)


# ------------------------------------------------------------------------------------- inputs


def exit_setup(class_name: str):
    """(ExitParams with grace + effective unclog_12h, roi) for a reference class, no instance."""
    from evaluation.fingerprint import load_strategy_class
    from utils.FamilyExits import class_level_exit_params

    cls = load_strategy_class(class_name)
    params = class_level_exit_params(cls)
    sell = getattr(cls, "sell_params", {}) or {}
    over = {"stoploss_grace_hours": float(sell.get("stoploss_grace_hours", 0.0)),
            "stoploss_grace_level": float(sell.get("stoploss_grace_level", 0.0))}
    hold = getattr(cls, "HOLD_WINDOW_HOURS", None)
    if hold is not None:
        over["unclog_12h_hours"] = max(params.unclog_12h_hours, float(hold))
    params = dataclasses.replace(params, **over)
    roi_map = getattr(cls, "minimal_roi", {}) or {}
    roi = float(roi_map["0"]) if "0" in roi_map else None
    return cls, params, roi


def family_params(cls, params, family: str):
    over = ((getattr(cls, "FAMILY_EXITS", None) or {}).get(family) or {})
    return dataclasses.replace(params, **over) if over else params


_BARS: dict = {}


def bars_for(pair: str):
    if pair not in _BARS:
        from research.clfstudy.data import load_pair, populate
        from research.outcomes.table import _rolling_rvol_series

        df = populate(load_pair(pair))
        b = P.Bars(
            open=df["open"].to_numpy(float), high=df["high"].to_numpy(float),
            low=df["low"].to_numpy(float), close=df["close"].to_numpy(float),
            atr=df["atr_pct_roll"].fillna(0.0).to_numpy(float),
            rvol=_rolling_rvol_series(df["volume"]),
            gm=df["guard_metric"].fillna(0.0).to_numpy(float),
        )
        _BARS[pair] = (b, pd.DatetimeIndex(df["date"]))
    return _BARS[pair]


_TICKS: dict = {}


def _tick(pair: str, fold: str, b, start: int, end: int):
    """The pair's price tick over one fold window, inferred from its own price grid."""
    key = (pair, fold)
    if key not in _TICKS:
        w = slice(max(start, 0), end + 1)
        _TICKS[key] = P.tick_size(np.concatenate([b.open[w], b.high[w], b.low[w], b.close[w]]))
    return _TICKS[key]


def archive_trades(tag: str) -> list[dict]:
    import evaluation.pool  # noqa: F401  (pins bootstrap.BT_RESULTS)
    from evaluation.bootstrap import _scale, _trades
    from evaluation.pool import arm_record

    plan = json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}.json").read_text())
    out = []
    for f in plan["folds"]:
        rec = arm_record(f["arm_tag"], f["fold_id"].upper())
        if not rec or rec.get("returncode") != 0:
            raise FileNotFoundError(f"{tag} {f['fold_id']}: no successful record")
        rows = _trades(rec) or []
        scale = _scale(rec, rows) if rows else 0.0
        for t in rows:
            out.append({**t, "fold": f["fold_id"], "test_start": f["test_start"],
                        "test_end": f["test_end"], "scale": scale})
    return out


# ------------------------------------------------------------------------------------ per trade


def trade_rows(tag: str, class_name: str, family: str) -> tuple[pd.DataFrame, dict]:
    """Pass 1: metrics, null replay, fidelity inputs, per trade."""
    from utils.ExitPolicy import net_from_price_ratio

    cls, params, roi = exit_setup(class_name)
    params = family_params(cls, params, family)
    rows = []
    for t in archive_trades(tag):
        b, dates = bars_for(t["pair"])
        e = int(dates.get_loc(pd.Timestamp(t["open_date"])))
        # A force-exit at the end of the DATA is stamped one bar past its last candle (§1.2).
        rec_idx = int(dates.searchsorted(pd.Timestamp(t["close_date"])))
        start = int(dates.searchsorted(pd.Timestamp(t["test_start"], tz="UTC")))
        end = int(dates.searchsorted(pd.Timestamp(t["test_end"], tz="UTC"))) - 1
        tick = _tick(t["pair"], t["fold"], b, start, end)
        open_rate = float(t["open_rate"])
        maxhold_end = min(e + int(params.max_hold_days * 24 * 60 / params.timeframe_minutes), end)
        m = P.path_metrics(b, e, open_rate, params, gain=TARGET_GAIN, horizon=HORIZON_BARS,
                           end=end, tick=tick)
        tm = P.first_touch(b, e, maxhold_end, open_rate * (1 + TARGET_GAIN))
        th = m["touch"]
        # ROI suppressed (`ignore_roi_if_entry_signal`): the bars of the recorded life where ROI
        # would have fired, yet the trade stayed open. Unknown beyond the recorded exit: not
        # blocked.
        roi_supp: frozenset = frozenset()
        if roi is not None:
            hi = b.high[e:rec_idx]
            ages = np.arange(len(hi)) * params.timeframe_minutes / 60.0
            prof = hi / open_rate * (1 - params.fee_per_side) / (1 + params.fee_per_side) - 1
            hit = (prof >= roi) & (ages >= params.min_discretionary_hold_hours)
            roi_supp = frozenset(int(e + i) for i in np.nonzero(hit)[0])
        null = P.replay(b, e, open_rate, params, roi=roi, roi_suppressed=roi_supp, end=end,
                        tick=tick)
        stops = P.stop_path(b, e, open_rate, params, P.Rules(), min(rec_idx, end), tick)
        rec_reason = "stop" if t["exit_reason"] in STOP_REASONS else t["exit_reason"]
        rows.append({
            "fold": t["fold"], "half": half_of(t["fold"]), "pair": t["pair"], "e": e, "end": end,
            "open_rate": open_rate, "stake": float(t["stake_amount"]), "scale": t["scale"],
            "rec_reason": rec_reason, "rec_idx": rec_idx,
            "rec_profit": float(t["profit_ratio"]),
            "rec_pp": float(t["profit_abs"]) * t["scale"],
            "rec_stop_abs": float(t.get("stop_loss_abs") or np.nan),
            "model_stop_at_exit": float(stops[min(rec_idx, end)]), "tick": tick,
            "touch_h": th, "touch_m": tm, "mae": m["mae"], "ambiguous": m["ambiguous"],
            "breach": m["breach"], "breach_after_grace": m["breach_after_grace"],
            "stop_at_touch": m["stop_at_touch"],
            "touch_h_after": th is not None and th > rec_idx,
            "touch_m_after": tm is not None and tm > rec_idx,
            "target_net": net_from_price_ratio(1 + TARGET_GAIN, params.fee_per_side),
            "atr_signal": float(b.atr[e - 1]),
            "roi_suppressed": roi_supp,
            "null_reason": null.reason, "null_idx": null.idx, "null_profit": null.profit,
        })
    df = pd.DataFrame(rows)
    for col in ("touch_h", "touch_m", "breach"):
        df[col] = df[col].astype("float")
    return df, {"class": class_name, "params": dataclasses.asdict(params), "roi": roi}


def fidelity(df: pd.DataFrame) -> dict:
    match = (df["null_reason"] == df["rec_reason"]) & (df["null_idx"] == df["rec_idx"])
    stops = df[df["rec_reason"] == "stop"]
    gap = (stops["model_stop_at_exit"] - stops["rec_stop_abs"]).abs() / stops["open_rate"]
    table = (df.assign(match=match).groupby("rec_reason")["match"]
             .agg(["count", "mean"]).round(4).to_dict(orient="index"))
    return {
        "null_replay_agreement": float(match.mean()) if len(df) else None,
        "by_recorded_reason": table,
        "stop_model_agreement": float((gap <= 0.001).mean()) if len(stops) else None,
        "stop_model_median_gap_pp": float(gap.median() * 100) if len(stops) else None,
    }


def replays(df: pd.DataFrame, class_name: str, family: str, values: dict) -> pd.DataFrame:
    """Pass 2: one replay per coupling value; delta pp vs the null replay."""
    cls, params, roi = exit_setup(class_name)
    params = family_params(cls, params, family)
    rules = {
        f"C1_{p}": P.Rules(stop_min_depth=values["C1"][p]) for p in GRID["C1"]
    } | {
        f"C3_{a}": P.Rules(horizon_action=a) for a in GRID["C3"]
    } | {
        f"C4_{d}": P.Rules(grace_hours=HORIZON_BARS * params.timeframe_minutes / 60.0,
                           grace_level=values["C4"][d]) for d in GRID["C4"]
    } | {
        f"C5_{lam}": P.Rules(lock_after_target=lam) for lam in GRID["C5"]
    }
    if values.get("C2") is not None:
        rules["C2_k"] = P.Rules(take_profit_atr_mult=values["C2"])
    out = {k: [] for k in rules}
    for r in df.itertuples():
        b, _ = bars_for(r.pair)
        for k, rule in rules.items():
            cf = P.replay(b, r.e, r.open_rate, params, roi=roi,
                          roi_suppressed=r.roi_suppressed, end=r.end, rules=rule, tick=r.tick)
            out[k].append((cf.profit - r.null_profit) * r.stake * r.scale)
    return df.assign(**{f"d_{k}": v for k, v in out.items()})


def early_profile(df: pd.DataFrame, params_take_profit: float, grace_level: float) -> dict:
    """C1 depths, C4 depths and C2's k from EARLY rows only (research R3)."""
    early = df[df["half"] == "early"]
    lp = early[early["touch_h"].notna() & ~early["ambiguous"]]
    depth = -lp["mae"]
    c1 = {p: max(-float(np.percentile(depth, p)), C1_BOUND) for p in GRID["C1"]}
    p95 = float(np.percentile(depth, 95))
    c4 = {"base": None, "p95": -min(p95, abs(grace_level))}
    k = None
    if params_take_profit < 1.0:  # a family with a real take-profit
        atr = early["atr_signal"][early["atr_signal"] > 0]
        k = float(np.median(params_take_profit / atr))
    cuts = tuple(float(x) for x in np.percentile(early["atr_signal"], [100 / 3, 200 / 3]))
    return {"C1": c1, "C4": c4, "C2": k, "atr_tercile_cuts": cuts}


def verdicts(df: pd.DataFrame, prof: dict, summaries: dict) -> dict:
    out = {}
    v1 = choose(df, "C1", list(GRID["C1"]))
    g1 = gains_by_half(df, "C1", v1)
    out["C1"] = {"value": {"percentile": v1, "depth": prof["C1"][v1]}, "gain_pp": g1,
                 "conditions": {h: [summaries[h]["b_killed_share"] is not None
                                    and summaries[h]["b_killed_share"] >= C1_KILLED_MIN,
                                    g1[h] > 0] for h in g1}}
    if prof["C2"] is None:
        out["C2"] = {"verdict": "N/A", "reason": "no take-profit in this family"}
    else:
        g2 = gains_by_half(df, "C2", "k")
        ratios = {h: tercile_hit_ratio(df[df["half"] == h], prof["atr_tercile_cuts"])
                  for h in ("early", "late")}
        out["C2"] = {"value": {"k": prof["C2"]}, "gain_pp": g2, "tercile_hit_ratio": ratios,
                     "conditions": {h: [ratios[h] is not None and ratios[h] < C2_TERCILE_RATIO,
                                        g2[h] > 0] for h in g2}}
    v3 = choose(df, "C3", list(GRID["C3"]))
    g3 = gains_by_half(df, "C3", v3)
    out["C3"] = {"value": {"action": v3}, "gain_pp": g3,
                 "conditions": {h: [g3[h] > 0] for h in g3}}
    v4 = choose(df, "C4", list(GRID["C4"]))
    g4 = gains_by_half(df, "C4", v4)
    out["C4"] = {"value": {"depth_choice": v4, "grace_level": prof["C4"][v4]}, "gain_pp": g4,
                 "conditions": {h: [
                     (summaries[h]["b_killed_after_grace_share_of_killed"] or 0)
                     >= C4_KILLED_IN_WINDOW_SHARE,
                     (summaries[h]["b_killed_after_grace_share_of_lp"] or 0)
                     >= C4_KILLED_IN_WINDOW_OF_LP,
                     g4[h] > 0] for h in g4}}
    v5 = choose(df, "C5", list(GRID["C5"]))
    g5 = gains_by_half(df, "C5", v5)
    out["C5"] = {"value": {"lambda": v5}, "gain_pp": g5,
                 "conditions": {h: [g5[h] > 0] for h in g5}}
    for k, v in out.items():
        if "conditions" in v:
            v["verdict"] = "SUPPORTED" if supported(v["conditions"]) else "NOT SUPPORTED"
            cols = sorted(c for c in df.columns if c.startswith(f"d_{k}_"))
            v["all_gains_pp"] = {c[len(f"d_{k}_"):]: gains_by_half(df, k, c[len(f"d_{k}_"):])
                                 for c in cols}
    return out


# ------------------------------------------------------------------------------------- output


def _f(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, (list, tuple)):
        return "/".join(_f(v, nd) for v in x)
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render(result: dict) -> str:
    lines = ["# Feasibility: target-aware exits (027 US1)", "", f"*{LABEL}*", ""]
    for fam, r in result["families"].items():
        fid = r["fidelity"]
        lines += [f"## {fam} (`{r['tag']}`)", "",
                  (f"fidelity: null replay reproduces {_f(fid['null_replay_agreement'])} of "
                   f"recorded exits (reason AND bar); stop model agrees on "
                   f"{_f(fid['stop_model_agreement'])} of stop exits (median gap "
                   f"{_f(fid['stop_model_median_gap_pp'])} pp)"), "",
                  "| metric | early | late |", "|---|---|---|"]
        for key in r["summary"]["early"]:
            lines.append(f"| {key} | {_f(r['summary']['early'][key])} | "
                         f"{_f(r['summary']['late'][key])} |")
        lines += ["", "| coupling | value (early-chosen) | gain pp early / late | verdict |",
                  "|---|---|---|---|"]
        for c, v in r.get("verdicts", {}).items():
            gain = v.get("gain_pp")
            g = f"{gain['early']:+.2f} / {gain['late']:+.2f}" if gain else "—"
            lines.append(f"| {c} | {json.dumps(v.get('value'))} | {g} | "
                         f"**{v.get('verdict')}** |")
        lines += ["", ("all grid values, replay gain pp (early / late; reported, not used to "
                       "choose):"), ""]
        for c, v in r.get("verdicts", {}).items():
            for val, g in (v.get("all_gains_pp") or {}).items():
                lines.append(f"- {c} `{val}`: {g['early']:+.2f} / {g['late']:+.2f}")
            if v.get("conditions"):
                lines.append(f"- {c} conditions (early / late): "
                             f"{v['conditions']['early']} / {v['conditions']['late']}")
        lines += ["", (f"null-replay agreement by recorded reason: "
                       f"`{json.dumps(fid['by_recorded_reason'])}`"), ""]
    lines += [f"overall: **{result['overall']}** (exit {result['exit']})", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--momentum", default="st_bgo_mx")
    ap.add_argument("--reversion", default="st_book_rev2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    arms = {"momentum": a.momentum, "reversion": a.reversion}
    result: dict = {"label": LABEL, "families": {}}
    try:
        built = {}
        for fam, tag in arms.items():
            plan = json.loads((STRAT / "evaluation" / "folds" / f"plan_{tag}.json").read_text())
            built[fam] = (tag, plan["base_strategy"], *trade_rows(tag, plan["base_strategy"],
                                                                  fam))
    except FileNotFoundError as exc:
        print(f"*** {exc}")
        return 2
    fidelity_ok = True
    all_verdicts = {}
    for fam, (tag, cls_name, df, setup) in built.items():
        fid = fidelity(df)
        ok = (fid["null_replay_agreement"] or 0) >= FIDELITY_MIN and (
            fid["stop_model_agreement"] is None or fid["stop_model_agreement"] >= FIDELITY_MIN)
        fidelity_ok &= ok
        summ = {h: summarise(df[df["half"] == h]) for h in ("early", "late")}
        entry = {"tag": tag, "class": cls_name, "setup": setup, "fidelity": fid,
                 "fidelity_ok": ok, "summary": summ}
        if ok:
            prof = early_profile(df, setup["params"]["take_profit"],
                                 setup["params"]["stoploss_grace_level"])
            df = replays(df, cls_name, fam, prof)
            v = verdicts(df, prof, summ)
            entry |= {"profile": prof, "verdicts": v}
            all_verdicts |= {f"{fam}:{k}": x for k, x in v.items()}
        result["families"][fam] = entry
    code = exit_code(True, fidelity_ok, all_verdicts)
    result["exit"] = code
    result["overall"] = {0: "SUPPORTED (at least one)", 3: "STOP (none supported)",
                         2: "NO VERDICT (fidelity gate failed)"}[code]
    out = Path(a.out)
    out.write_text(render(result))
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=str))
    print(out.read_text())
    return code


if __name__ == "__main__":
    sys.exit(main())

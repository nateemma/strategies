"""SimpleTrigger grace screen (2026-10-13): does NNNC's adopted 24 h grace window also suit the
model-free framework? Replays the grace-length grid on the two SimpleTrigger reference books'
2023-26 trades (027 replay, `research/exit_coupling`). 2022 is NOT replayed here: it is the
confirmatory window of the pre-registered arms.

    PYTHONPATH=. ../../.venv/bin/python -m research.st_grace.screen
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

from research.exit_coupling import feasibility as F  # noqa: E402
from research.exit_coupling import paths as P  # noqa: E402


HERE = Path(__file__).resolve().parent
#: key -> (reference tag, class, family)
REFS = {
    "MOM": ("st_bgo_mx", "SimpleTrigger_BreakoutGbb_MomExit", "momentum"),
    "REV": ("st_prom_c2", "SimpleTrigger_RevOnlyVolreg2_P2", "reversion"),
}
GRACE = (6.5, 12.0, 18.0, 24.0, 36.0, 48.0)


def screen(tag: str, cls_name: str, family: str) -> dict:
    df, _ = F.trade_rows(tag, cls_name, family)
    cls, params, roi = F.exit_setup(cls_name)
    params = F.family_params(cls, params, family)
    # 028 C2 lives on the params; the replay applies it through Rules (trade_rows' null does not).
    k = params.take_profit_atr_mult
    if k is not None:
        params = dataclasses.replace(params, take_profit_atr_mult=None)
    rules = {g: P.Rules(grace_hours=g, take_profit_atr_mult=k) for g in GRACE}
    base = params.stoploss_grace_hours
    out = {g: {"early": 0.0, "late": 0.0} for g in GRACE}
    match = 0
    for t in df.itertuples():
        b, _ = F.bars_for(t.pair)
        res = {g: P.replay(b, t.e, t.open_rate, params, roi=roi, roi_suppressed=t.roi_suppressed,
                           end=t.end, rules=r, tick=t.tick) for g, r in rules.items()}
        null = res[base]
        match += int(null.reason == t.rec_reason and null.idx == t.rec_idx)
        for g, cf in res.items():
            out[g][t.half] += (cf.profit - null.profit) * t.stake * t.scale
    return {"trades": len(df), "base_grace": base, "fidelity": round(match / len(df), 4),
            "delta_pp": {str(g): {h: round(v, 3) for h, v in d.items()} for g, d in out.items()}}


def main() -> int:
    res = {"label": "PROVISIONAL -- replay deltas vs each reference's own grace (6.5 h); 2023-26 "
                    "only; the replay ignores re-sequencing (arms keep ~0-0.7x of grace gains)"}
    for key, ref in REFS.items():
        res[key] = screen(*ref)
        print(key, json.dumps(res[key]), flush=True)
    (HERE / "screen.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

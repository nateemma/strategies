"""036 US4 (T033): NNNC and the momentum basket in ONE wallet -- an order-level replay.

    PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.shared_wallet

DEPLOYMENT INFORMATION ONLY (spec US4, FR-017); it changes no NNNC default.

`evaluation.books.shared_wallet` treats each trade as one fill held for its whole life. That is
right for NNNC, but wrong for the basket, which adds and trims hourly while aiming to be fully
invested. So this replays every recorded FILL of both books (cap_aro_ref, mom_basket) through one
wallet of CAPACITY USDT, fold by fold, in time order. Exits come before entries at a timestamp.
Same-time entries go in book priority order: NNNC first by default; `--priority basket` reverses
it.

* An entry fill takes min(requested cost, free cash); an order cut below MIN_STAKE by a
  lack of cash is refused (an order recorded below it fills as recorded).
* An exit returns its recorded proceeds scaled by the trade's filled share, so realised
  profit flows back into the shared wallet (the basket compounds; its later orders are sized
  from a grown wallet).
* A trade's P&L is scaled by its filled share of requested entry cost. This is APPROXIMATE: it
  assumes a shrink spreads evenly over the trade's lots, and a refused entry would have left cash
  or a pair free for a later signal the archive does not hold.

Reported per book and in total (USDT, and % of one CAPACITY wallet), per fold and per half:
- separate = each book in its own wallet (the recorded P&L; 2 × CAPACITY of capital);
- shared = both books in one CAPACITY wallet;
- refused and shrunk orders per book, and the P&L they cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

HERE = Path(__file__).resolve().parent
CAPACITY = 10000.0
MIN_STAKE = 10.0
BOOKS = ("cap_aro_ref", "mom_basket")
EARLY = tuple(f"f{i:02d}" for i in range(8))


def events(trades: list[dict], book: str) -> list[tuple]:
    """(ts_ms, is_entry, book, trade_idx, order_idx) for every filled order."""
    out = []
    for ti, t in enumerate(trades):
        for oi, o in enumerate(t.get("orders") or []):
            ts = o.get("order_filled_timestamp")
            if ts is None or not o.get("amount"):
                continue
            out.append((int(ts), bool(o["ft_is_entry"]), book, ti, oi))
    return out


def replay(books: dict[str, list[dict]], capacity: float = CAPACITY,
           priority: tuple[str, ...] = BOOKS, min_stake: float = MIN_STAKE) -> dict:
    """books: {book: [trade dicts with orders and profit_abs]} for ONE fold."""
    rank = {b: i for i, b in enumerate(priority)}
    ev = [e for b, ts in books.items() for e in events(ts, b)]
    ev.sort(key=lambda e: (e[0], e[1], rank[e[2]], e[3], e[4]))   # exits (False) first
    cash = capacity
    state = {}   # (book, ti) -> dict(req, fill, rec_held, basis)
    stats = {b: {"orders": 0, "refused": 0, "shrunk": 0} for b in books}
    for ts, is_entry, b, ti, oi in ev:
        o = books[b][ti]["orders"][oi]
        st = state.setdefault((b, ti), {"req": 0.0, "fill": 0.0, "rec_held": 0.0, "basis": 0.0})
        if is_entry:
            req = float(o.get("cost") or o["amount"] * o["safe_price"])
            stats[b]["orders"] += 1
            fill = min(req, cash)
            # refuse only an order CUT below the minimum by a lack of cash; a small order the
            # book really placed (the basket's top-ups are often < 10 USDT) fills as recorded
            if fill < req * (1 - 1e-9) and fill < min_stake:
                fill = 0.0
                stats[b]["refused"] += 1
            elif fill < req * (1 - 1e-9):
                stats[b]["shrunk"] += 1
            cash -= fill
            st["req"] += req
            st["fill"] += fill
            st["rec_held"] += float(o["amount"])
            st["basis"] += fill
        else:
            if st["rec_held"] <= 0:
                continue
            frac = min(1.0, float(o["amount"]) / st["rec_held"])
            # proceeds, not just the basis: a compounding book (the basket's "unlimited" stake)
            # sizes later orders from realised profit, so the shared wallet must grow with it
            scale = st["fill"] / st["req"] if st["req"] > 0 else 0.0
            proceeds = float(o.get("cost") or o["amount"] * o["safe_price"]) * scale
            cash += proceeds
            st["basis"] -= st["basis"] * frac
            st["rec_held"] -= float(o["amount"])
    pnl = {b: {"separate": 0.0, "shared": 0.0} for b in books}
    for (b, ti), st in state.items():
        p = float(books[b][ti].get("profit_abs") or 0.0)
        ratio = st["fill"] / st["req"] if st["req"] > 0 else 1.0
        pnl[b]["separate"] += p
        pnl[b]["shared"] += p * ratio
    return {"pnl": pnl, "stats": stats}


def summarise(per_fold: dict) -> dict:
    def agg(ids):
        out = {}
        for b in BOOKS:
            sep = sum(per_fold[f]["pnl"][b]["separate"] for f in ids if f in per_fold)
            sh = sum(per_fold[f]["pnl"][b]["shared"] for f in ids if f in per_fold)
            st = {k: sum(per_fold[f]["stats"][b][k] for f in ids if f in per_fold)
                  for k in ("orders", "refused", "shrunk")}
            out[b] = {"separate_usdt": round(sep, 2), "shared_usdt": round(sh, 2),
                      "lost_usdt": round(sep - sh, 2), **st}
        sep = sum(v["separate_usdt"] for v in out.values())
        sh = sum(v["shared_usdt"] for v in out.values())
        out["total"] = {"separate_usdt (two wallets)": round(sep, 2),
                        "shared_usdt (one wallet)": round(sh, 2),
                        "shared_pct_of_one_wallet": round(100 * sh / CAPACITY, 2),
                        "separate_pct_of_two_wallets": round(100 * sep / (2 * CAPACITY), 2)}
        return out
    late = tuple(f for f in per_fold if f not in EARLY)
    return {"early": agg(EARLY), "late": agg(late), "all": agg(tuple(per_fold))}


def _fold_trades(tag: str) -> dict[str, list[dict]]:
    from research.live_pnl.restate import ARMS, RESULTS, FOLDS
    import zipfile

    plan = json.loads((FOLDS / f"plan_{tag}.json").read_text())
    out = {}
    for f in plan["folds"]:
        rec = json.loads((ARMS / f"{f['arm_tag']}_{f['fold_id'].upper()}.json").read_text())
        if rec.get("returncode") != 0:
            raise SystemExit(f"{tag} {f['fold_id']}: arm record failed")
        if not rec.get("n_trades"):   # a completed zero-trade fold (e.g. the basket risk-off)
            out[f["fold_id"]] = []
            continue
        with zipfile.ZipFile(RESULTS / rec["result_file"]) as z:
            name = [n for n in z.namelist() if n.endswith(".json") and "config" not in n
                    and "meta" not in n][0]
            out[f["fold_id"]] = list(json.loads(z.read(name))["strategy"].values())[0]["trades"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", choices=("nnnc", "basket"), default="nnnc")
    a = ap.parse_args(argv)
    prio = BOOKS if a.priority == "nnnc" else tuple(reversed(BOOKS))
    data = {b: _fold_trades(b) for b in BOOKS}
    per_fold = {f: replay({b: data[b].get(f, []) for b in BOOKS}, priority=prio)
                for f in data[BOOKS[0]]}
    out = {"label": "036 US4 shared wallet -- DEPLOYMENT INFORMATION ONLY (PROVISIONAL; order-level "
                    "replay, approximate: shrink spread over lots, refused entries not re-sequenced)",
           "capacity_usdt": CAPACITY, "priority": list(prio), "summary": summarise(per_fold),
           "per_fold": {f: {"pnl": {b: {k: round(v, 2) for k, v in d.items()} for b, d in r["pnl"].items()},
                            "stats": r["stats"]} for f, r in per_fold.items()}}
    name = "shared_wallet.json" if a.priority == "nnnc" else "shared_wallet_basket_first.json"
    (HERE / name).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"]["all"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

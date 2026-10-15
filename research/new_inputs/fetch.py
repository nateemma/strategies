"""035 T014: download the free sources into `user_data/data/external/` (research R7).

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.fetch --source global_spot \\
        [--pairs ...]

  global_spot  data.binance.vision spot 15m klines: monthly 2021-01..2026-08, daily 2026-09-01..07
  um_metrics   data.binance.vision UM futures daily metrics 2021-12-01..2026-09-07 (no monthly)
  um_premium   data.binance.vision UM premium-index 15m klines: monthly + daily for 2026-09
  us_spot      api.binance.us /api/v3/klines, 15m, paged 1000 bars from each pair's listing

Downloaded files are untrusted data: each archive is stored as fetched, verified against its
published `.CHECKSUM` (sha256), and later read with zipfile; nothing is extracted or executed.
A 404 is an absent period (the pair was not listed), recorded, not an error. A checksum mismatch
is refetched once, then recorded as a coverage gap. Resumable: verified files are skipped. Refuses
to run while a trainer or backtest chain is running.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.new_inputs import inputs as I  # noqa: E402


EXTERNAL = STRAT.parent / "data" / "external"
MANIFEST = EXTERNAL / "manifest.json"
DV = "https://data.binance.vision/data"
US = "https://api.binance.us/api/v3/klines"
FIRST_MONTH, LAST_MONTH = date(2021, 1, 1), date(2026, 8, 1)
TAIL_DAYS = (date(2026, 9, 1), date(2026, 9, 7))
METRICS_DAYS = (date(2021, 12, 1), date(2026, 9, 7))
US_START_MS, US_END_MS = 1577836800000, 1788825600000   # 2020-01-01 .. 2026-09-08
WORKERS = 8


def _get(url: str, timeout: int = 60) -> tuple[int, bytes]:
    # only the module's own https constants (DV, US) are ever passed here
    req = urllib.request.Request(url, headers={"User-Agent": "freqtrade-research/035"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def _months(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def _days(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def archive_urls(source: str, sym: str) -> list[str]:
    if source == "global_spot":
        m = [f"{DV}/spot/monthly/klines/{sym}/15m/{sym}-15m-{d:%Y-%m}.zip"
             for d in _months(FIRST_MONTH, LAST_MONTH)]
        return m + [f"{DV}/spot/daily/klines/{sym}/15m/{sym}-15m-{d:%Y-%m-%d}.zip"
                    for d in _days(*TAIL_DAYS)]
    if source == "um_premium":
        m = [f"{DV}/futures/um/monthly/premiumIndexKlines/{sym}/15m/{sym}-15m-{d:%Y-%m}.zip"
             for d in _months(FIRST_MONTH, LAST_MONTH)]
        return m + [f"{DV}/futures/um/daily/premiumIndexKlines/{sym}/15m/{sym}-15m-{d:%Y-%m-%d}.zip"
                    for d in _days(*TAIL_DAYS)]
    if source == "um_metrics":
        return [f"{DV}/futures/um/daily/metrics/{sym}/{sym}-metrics-{d:%Y-%m-%d}.zip"
                for d in _days(*METRICS_DAYS)]
    raise ValueError(source)


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def save_manifest(m: dict, path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=0, sort_keys=True))
    tmp.replace(path)


def verified_paths(path: Path = MANIFEST) -> set:
    return {k for k, v in load_manifest(path).items() if v.get("status") == "verified"}


def manifest_sha(path: Path = MANIFEST) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""


def fetch_one(url: str, dest: Path, get=_get) -> dict:
    """Download one archive with its CHECKSUM; one refetch on mismatch. Returns a manifest entry."""
    st, ck = get(url + ".CHECKSUM")
    if st == 404:
        return {"url": url, "status": "absent"}
    if st != 200:
        return {"url": url, "status": f"error_checksum_http_{st}"}
    want = ck.decode("ascii", "replace").split()[0].strip().lower()
    for _attempt in (1, 2):
        st, body = get(url)
        if st != 200:
            return {"url": url, "status": f"error_http_{st}"}
        got = hashlib.sha256(body).hexdigest()
        if got == want:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            return {"url": url, "status": "verified", "sha256": got, "bytes": len(body),
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return {"url": url, "status": "checksum_mismatch", "want": want, "got": got}


def fetch_archives(source: str, pairs, get=_get, root: Path = EXTERNAL,
                   manifest_path: Path = MANIFEST, workers: int = WORKERS) -> dict:
    man = load_manifest(manifest_path)
    jobs = []
    for p in pairs:
        sym = I.symbol(p)
        for url in archive_urls(source, sym):
            dest = root / source / sym / url.rsplit("/", 1)[1]
            e = man.get(str(dest))
            if e and e.get("status") in ("verified", "absent") and (
                    e["status"] == "absent" or dest.is_file()):
                continue
            jobs.append((url, dest))
    counts: dict = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, u, d, get): d for u, d in jobs}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            e = fu.result()
            man[str(futs[fu])] = e
            counts[e["status"]] = counts.get(e["status"], 0) + 1
            if i % 500 == 0:
                save_manifest(man, manifest_path)
                print(source, i, "/", len(jobs), counts, flush=True)
    save_manifest(man, manifest_path)
    return counts


def fetch_us(pairs, get=_get, root: Path = EXTERNAL, manifest_path: Path = MANIFEST,
             pause: float = 0.25) -> dict:
    """Binance.US REST klines (no published checksum: entries record pages and the last bar)."""
    man = load_manifest(manifest_path)
    out = {}
    for p in pairs:
        sym = I.symbol(p)
        dest = root / "us_spot" / sym / "klines.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        start, pages = US_START_MS, 0
        if dest.is_file():
            lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
            if lines:
                start = json.loads(lines[-1])[-1][0] + 15 * 60 * 1000
                pages = len(lines)
        while start < US_END_MS:
            st, body = get(f"{US}?symbol={sym}&interval=15m&limit=1000&startTime={start}")
            if st != 200:
                raise RuntimeError(f"{sym}: Binance.US HTTP {st} at {start}")
            rows = json.loads(body)
            if not isinstance(rows, list):
                raise RuntimeError(f"{sym}: unexpected Binance.US payload")
            rows = [r for r in rows if r[0] < US_END_MS]
            if not rows:
                break
            with dest.open("a") as fh:
                fh.write(json.dumps(rows) + "\n")
            pages += 1
            start = rows[-1][0] + 15 * 60 * 1000
            time.sleep(pause)
        man[str(dest)] = {"url": US, "status": "verified", "checksum": "none (REST)",
                          "pages": pages,
                          "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        save_manifest(man, manifest_path)
        out[p] = pages
        print(p, "pages", pages, flush=True)
    return out


def _busy() -> list[str]:
    from research.candidate_training import train as T

    return T.other_chains(T._ps(), os.getpid())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=("global_spot", "um_metrics", "um_premium", "us_spot"))
    ap.add_argument("--pairs", help="comma-separated; default the 10 config_wf pairs")
    a = ap.parse_args(argv)
    busy = _busy()
    if busy:
        print("REFUSED: a chain is running:\n  " + "\n  ".join(busy))
        return 2
    pairs = tuple(a.pairs.split(",")) if a.pairs else I.PAIRS
    if a.source == "us_spot":
        print(fetch_us(pairs))
    else:
        print(a.source, fetch_archives(a.source, pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

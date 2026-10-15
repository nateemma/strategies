"""035 T014: archive fetch rules with a fake opener (offline)."""

from __future__ import annotations

import hashlib
import json

from research.new_inputs import fetch as FE


def _fake(store: dict, calls: list):
    def get(url):
        calls.append(url)
        v = store.get(url)
        if callable(v):
            v = v()
        return (404, b"") if v is None else (200, v)
    return get


def test_verified_skipped_absent_recorded_mismatch_refetched_once(tmp_path, monkeypatch):
    monkeypatch.setattr(FE, "archive_urls", lambda s, sym: [f"https://x/{sym}-a.zip",
                                                            f"https://x/{sym}-b.zip",
                                                            f"https://x/{sym}-c.zip"])
    good = b"zipbytes"
    ck = hashlib.sha256(good).hexdigest().encode() + b"  a.zip"
    store = {"https://x/ZECUSDT-a.zip": good, "https://x/ZECUSDT-a.zip.CHECKSUM": ck,
             "https://x/ZECUSDT-c.zip": b"corrupt", "https://x/ZECUSDT-c.zip.CHECKSUM": ck}
    calls = []
    man = tmp_path / "manifest.json"
    counts = FE.fetch_archives("global_spot", ["ZEC/USDT"], _fake(store, calls), tmp_path, man, 1)
    assert counts == {"verified": 1, "absent": 1, "checksum_mismatch": 1}
    assert calls.count("https://x/ZECUSDT-c.zip") == 2          # refetched once
    m = json.loads(man.read_text())
    assert (tmp_path / "global_spot" / "ZECUSDT" / "ZECUSDT-a.zip").read_bytes() == good
    assert not (tmp_path / "global_spot" / "ZECUSDT" / "ZECUSDT-c.zip").exists()
    assert {v["status"] for v in m.values()} == {"verified", "absent", "checksum_mismatch"}
    calls.clear()
    FE.fetch_archives("global_spot", ["ZEC/USDT"], _fake(store, calls), tmp_path, man, 1)
    # verified and absent files are skipped
    assert not any(c.startswith(("https://x/ZECUSDT-a", "https://x/ZECUSDT-b")) for c in calls)
    assert FE.verified_paths(man) == {str(tmp_path / "global_spot" / "ZECUSDT" / "ZECUSDT-a.zip")}


def test_us_pages_resume_from_the_last_bar(tmp_path):
    bar = 15 * 60 * 1000
    t0 = FE.US_END_MS - 3 * bar

    def row(t):
        return [t, "1", "1", "1", "1", "1", t + bar - 1, "1", 3, "1", "1", "0"]

    def get(url):
        start = int(url.rsplit("startTime=", 1)[1])
        rows = [row(t) for t in range(max(start, t0), FE.US_END_MS, bar)][:2]
        return 200, json.dumps(rows).encode()

    man = tmp_path / "m.json"
    FE.fetch_us(["ZEC/USDT"], get, tmp_path, man, pause=0)
    lines = (tmp_path / "us_spot" / "ZECUSDT" / "klines.jsonl").read_text().splitlines()
    got = [r[0] for ln in lines for r in json.loads(ln)]
    assert got == [t0, t0 + bar, t0 + 2 * bar]
    FE.fetch_us(["ZEC/USDT"], get, tmp_path, man, pause=0)       # nothing new to fetch
    assert len((tmp_path / "us_spot" / "ZECUSDT" / "klines.jsonl").read_text().splitlines()) == 2

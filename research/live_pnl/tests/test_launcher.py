"""036 T008: the launcher's pre-start refusals (research R2)."""

from __future__ import annotations

import os
import socket

from research.live_pnl import launch_check as LC


REC = [{"run_id": "live1", "bot_class": "X_REC", "model_key": "k", "status": "picked",
        "retired": None}]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_clean_start_passes(tmp_path):
    assert LC.refusals(_free_port(), "tradesv3.live_nnnc.sqlite", tmp_path / "pid", REC,
                       keycheck=lambda r: None) == []


def test_port_in_use_is_refused(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        out = LC.refusals(port, "tradesv3.live_nnnc.sqlite", tmp_path / "pid", REC,
                          keycheck=lambda r: None)
    assert any("port" in x for x in out)


def test_momentum_db_is_refused(tmp_path):
    out = LC.refusals(_free_port(), "tradesv3.dryrun.sqlite", tmp_path / "pid", REC,
                      keycheck=lambda r: None)
    assert any("momentum" in x for x in out)


def test_running_pid_is_refused(tmp_path):
    pf = tmp_path / "pid"
    pf.write_text(str(os.getpid()))
    out = LC.refusals(_free_port(), "tradesv3.live_nnnc.sqlite", pf, REC, keycheck=lambda r: None)
    assert any("already running" in x for x in out)
    pf.write_text("999999")  # stale pid: not running
    assert LC.refusals(_free_port(), "tradesv3.live_nnnc.sqlite", pf, REC,
                       keycheck=lambda r: None) == []


def test_key_mismatch_and_no_record_are_refused(tmp_path):
    out = LC.refusals(_free_port(), "tradesv3.live_nnnc.sqlite", tmp_path / "pid", REC,
                      keycheck=lambda r: "X_REC resolves to key a, record says k")
    assert any("resolves to key" in x for x in out)
    out = LC.refusals(_free_port(), "tradesv3.live_nnnc.sqlite", tmp_path / "pid", [],
                      keycheck=lambda r: None)
    assert any("no current live record" in x for x in out)

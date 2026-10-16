"""036 T009: pre-start refusals for scripts/live_nnnc.sh (research R2).

    PYTHONPATH=user_data/strategies .venv/bin/python -m research.live_pnl.launch_check   # repo root

On success, prints `BOT_CLASS=<class>` and `RUN_ID=<id>` for the shell. Otherwise it exits 2 with
the reasons. It refuses when:
- the API port is in use;
- the database path is the momentum bot's;
- the pid file names a running process;
- there is no current live record;
- the bot class's model key differs from the record, or its weights are missing.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
ROOT = STRAT.parents[1]
RECORDS = Path(__file__).resolve().parent / "live_models.json"
MOMENTUM_DB = "tradesv3.dryrun.sqlite"
DEFAULTS = {"port": 8081, "db": "tradesv3.live_nnnc.sqlite",
            "pidfile": str(ROOT / "user_data" / "logs" / "live_nnnc.pid")}


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def pid_running(pidfile: Path) -> int | None:
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def current_record(records: list) -> dict | None:
    live = [r for r in records if r.get("retired") is None and r.get("status") in ("picked", "live")]
    return live[-1] if live else None


def key_problem(rec: dict) -> str | None:
    from evaluation.fingerprint import load_config, load_strategy_class
    from Framework import ModelKey as MK

    cfg = load_config(STRAT / "config" / "config_live_nnnc.json")
    cls = load_strategy_class(rec["bot_class"])
    key = MK.key_hash(MK.resolve(cls, cfg))
    if key != rec["model_key"]:
        return f"{rec['bot_class']} resolves to key {key}, record says {rec['model_key']}"
    if MK.locate_for(cls, cfg, rec["bot_class"]).kind == "absent":
        return f"no weights for {rec['bot_class']} (key {key})"
    return None


def refusals(port: int, db: str, pidfile: Path, records: list, keycheck=key_problem) -> list[str]:
    out = []
    if Path(db).name == MOMENTUM_DB:
        out.append(f"database {db} is the momentum dry-run's")
    pid = pid_running(pidfile)
    if pid:
        out.append(f"already running (pid {pid} in {pidfile})")
    if port_in_use(port):
        out.append(f"port {port} is in use")
    rec = current_record(records)
    if rec is None:
        out.append("no current live record in live_models.json (run research.live_pnl.live_fold)")
    else:
        why = keycheck(rec)
        if why:
            out.append(why)
    return out


def main() -> int:
    port = int(os.environ.get("LIVE_NNNC_PORT", DEFAULTS["port"]))
    db = os.environ.get("LIVE_NNNC_DB", DEFAULTS["db"])
    pidfile = Path(os.environ.get("LIVE_NNNC_PIDFILE", DEFAULTS["pidfile"]))
    records = json.loads(RECORDS.read_text()) if RECORDS.is_file() else []
    bad = refusals(port, db, pidfile, records)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    rec = current_record(records)
    print(f"BOT_CLASS={rec['bot_class']}")
    print(f"RUN_ID={rec['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

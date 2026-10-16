"""036 T007: the live config is a dry-run, isolated from the momentum bot's."""

from pathlib import Path

from evaluation.fingerprint import load_config


CFG = Path(__file__).resolve().parents[3] / "config"


def test_live_config_is_isolated_dry_run():
    live = load_config(CFG / "config_live_nnnc.json")
    mom = load_config(CFG / "config_mom_dryrun_15m.json")
    wf = load_config(CFG / "config_wf.json")
    assert live["dry_run"] is True
    assert live["api_server"]["listen_port"] != mom["api_server"]["listen_port"]
    assert live["bot_name"] != mom.get("bot_name")
    assert live["exchange"]["pair_whitelist"] == wf["exchange"]["pair_whitelist"]
    assert (live["stake_amount"], live["max_open_trades"]) == (wf["stake_amount"], wf["max_open_trades"])
    assert live["exchange"]["key"] == "" and live["exchange"]["secret"] == ""
    for k in ("entry_pricing", "exit_pricing", "order_types", "unfilledtimeout"):
        assert live[k] == mom[k], k

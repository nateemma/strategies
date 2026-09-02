"""The pinned baseline exit configuration (T002, FR-028, FR-028b).

WHY THESE ARE LITERALS AND NOT READ FROM THE STRATEGY JSON.

`NNNC/NNNC_MLX.json` is untracked working state. During this study a hyperopt
overwrote its entire exit block (grace 5.6h -> 4.3h, take-profit 0.024 -> 0.005,
and more). A measurement taken after that silently used the new values and
produced numbers identical to a different arm - the confound was only caught
because a pinned run disagreed with it.

So the baseline lives here, in source, and every script imports it from here.
Reading it from the JSON is prohibited (FR-028b).

On import this module verifies the recorded archives resolve and that their
trade counts match what characterisation assumed. A mismatch raises rather than
letting the study characterise a configuration nobody chose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("/Users/philprice95/projects/freqtrade")
ARMS = ROOT / "user_data/strategies/regime/arms"
BT_RESULTS = ROOT / "user_data/backtest_results"

# The trade counts the characterisation in research.md was computed from.
EXPECTED_TRADES = {"W1": 51, "W2": 96, "W3": 20}
EXPECTED_TOTAL = 167


@dataclass(frozen=True)
class BaselineConfig:
    """The exit configuration that produced the 167 characterised trades."""

    grace_hours: float = 5.6
    grace_level: float = -0.27          # INERT - see research R1
    take_profit: float = 0.024
    max_days: int = 10
    roi: float = 0.03
    stoploss: float = -0.05
    exit_signal_enabled: bool = False

    # Settings measured as having NO effect in this configuration. Listed so no
    # reader mistakes one for a tunable and no candidate proposes touching one.
    inert: tuple = field(default=(
        "stoploss_grace_level",          # after_fill branch unreachable (R1)
        "trailing_stop_positive",        # inert under use_custom_stoploss (R3)
        "trailing_stop_positive_offset",
        "trailing_only_offset_is_reached",
        "exit_guard_threshold",          # behind the enable_exit_signal return
        "exit_close_norm_threshold",
    ))

    def run_arm_args(self) -> list[str]:
        """The exact `run_arm.py` flags that reproduce this baseline.

        Pass as a list; in zsh an unquoted string does NOT word-split, which
        silently sends the whole thing as one argument.
        """
        return [
            "--set-sell", f"stoploss_grace_hours={self.grace_hours}",
            "--set-sell", f"stoploss_grace_level={self.grace_level}",
            "--set-sell", f"cexit_take_profit={self.take_profit}",
            "--set-sell", f"cexit_max_days={self.max_days}",
            "--set-sell", "enable_exit_signal=false",
            "--roi", str(self.roi),
            "--stoploss", str(self.stoploss),
        ]


BASELINE = BaselineConfig()


def archive_path(window: str) -> Path:
    """The recorded backtest archive for one window."""
    rec = json.loads((ARMS / f"applied_{window}.json").read_text())
    return BT_RESULTS / rec["result_file"]


def load_trades(window: str | None = None):
    """Recorded baseline trades. No model is loaded and nothing trains."""
    import pandas as pd
    from freqtrade.data.btanalysis import load_backtest_data

    windows = [window] if window else list(EXPECTED_TRADES)
    frames = []
    for w in windows:
        df = load_backtest_data(archive_path(w))
        df = df.assign(window=w)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def verify() -> dict:
    """Assert the baseline is what characterisation assumed. Raises if not."""
    counts = {}
    for w, expected in EXPECTED_TRADES.items():
        p = archive_path(w)
        if not p.exists():
            raise SystemExit(f"baseline archive missing for {w}: {p}")
        n = len(load_trades(w))
        counts[w] = n
        if n != expected:
            raise SystemExit(
                f"{w}: baseline archive has {n} trades, expected {expected}. "
                "The recorded results are not the ones characterisation used - "
                "halt rather than characterise a different configuration."
            )
    total = sum(counts.values())
    if total != EXPECTED_TOTAL:
        raise SystemExit(f"baseline total {total} trades, expected {EXPECTED_TOTAL}")
    return counts


if __name__ == "__main__":
    counts = verify()
    print("BASELINE (pinned literals, NOT read from NNNC/NNNC_MLX.json):")
    for k, v in vars(BASELINE).items():
        if k == "inert":
            continue
        print(f"  {k:<22}{v}")
    print(f"\n  INERT in this configuration ({len(BASELINE.inert)}):")
    for name in BASELINE.inert:
        print(f"    - {name}")
    print(f"\n  archives verified: {counts}, total {sum(counts.values())}")
    print("\n  run_arm args:")
    print("    " + " ".join(BASELINE.run_arm_args()))

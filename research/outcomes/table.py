"""The per-bar outcome table (feature 021, FR-006 through FR-009b).

Builds one `OutcomeRecord` per bar per pair by replaying `utils.ExitPolicy.simulate_forward`
forward from a hypothetical long entry at that bar's close. Contains NO exit arithmetic of
its own: `simulate_forward` is the single source of truth for the stop / grace /
take-profit / hold-cap logic (contracts/outcome-table.md, FR-001/FR-002). This module's job
is sourcing the per-bar inputs `simulate_forward` reads, deciding which production stake
size a bar could have filled (FR-009a), and persisting the result (FR-007).

`ExitConfig` WRAPS `utils.ExitPolicy.ExitParams` rather than restating its field list
(deliberate, see `ExitConfig`'s docstring below) -- the whole point of feature 021 is one
exit model; a second struct re-declaring the same eleven numbers would be exactly the kind
of duplication Phase 2 removed.

HAZARD (surfaced in Phase 2, repeated here because it bites at construction time): three
different grace configurations exist in this repository. `ExitConfig.from_operating_point`
is the ONLY constructor that should be used to build a table-ready config -- it reads grace
from the pinned `NNNC/NNNC_MLX_Open.json`, never from `ExitParams`' (deliberately absent)
default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from research.outcomes import CACHE_DIR
from research.outcomes.leakage import (
    OUTCOME_COLUMN_CLOSE_REASON,
    OUTCOME_COLUMN_FILLABLE,
    OUTCOME_COLUMN_FILLABLE_FRACTION,
    OUTCOME_COLUMN_GAPFILL_SHARE,
    OUTCOME_COLUMN_HOLD_BARS,
    OUTCOME_COLUMN_POSITIVE,
    OUTCOME_COLUMN_PROFIT,
)
from utils.ExitPolicy import SIMULATION_LOGIC_VERSION, ExitParams, simulate_forward

#: `atr_stoploss_multiplier` / `atr_stoploss_cap` / `atr_stoploss_floor` /
#: `use_volume_confirmation_stoploss` are plain class attributes (not freqtrade
#: `Parameter`s) on `BaseStrategy`, and `use_atr_adaptive_stoploss` likewise on
#: `BaseNNStrategy` -- reading the class reads the live value with no risk of a
#: strategy-JSON override shadowing it (the same reasoning
#: `Framework/tests/test_exit_policy_fixed_params.py`'s T011 uses). Importing these
#: classes costs ~3.5s (freqtrade/MLX/TensorFlow transitively) but does not
#: instantiate a strategy or touch a dataprovider -- see this module's docstring in
#: the handback report for why a lighter path was not available.
from Framework.BaseStrategy import BaseStrategy  # noqa: E402
from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: E402

#: The ONLY live source of the Open operating point's `minimal_roi` (feature 021's FR-001
#: amendment). `BaseStrategy.minimal_roi` is {"0": 0.03} -- the 0.05 the amendment's research
#: cites is NNNCStrategy's override, not a freqtrade Parameter, so there is no `.value` to
#: read off the JSON the way take_profit/max_hold_days are; the class attribute is the only
#: place it lives.
from NNNC.NNNCStrategy import NNNCStrategy  # noqa: E402

from research.clfstudy.data import load_pair, populate, zero_volume_share  # noqa: E402

#: The pinned production operating point (feature 021's FR-019 anchor). `ExitConfig`
#: MUST source grace / take-profit / hold-cap from here, never from a default --
#: see the module docstring.
OPEN_OPERATING_POINT_JSON = (
    Path(__file__).resolve().parents[2] / "NNNC" / "NNNC_MLX_Open.json"
)

#: The production per-trade stake (`config_binanceus.json`'s `stake_amount`).
#: Deliberately NOT a field of `ExitConfig`/`ExitParams`: it does not change
#: `simulate_forward`'s output (profit is a fraction, independent of stake size), it
#: only changes whether a bar counts as fillable (FR-009a) -- so it is a parameter of
#: the table *builder*, not of the exit policy, and is not part of `config_hash`.
PRODUCTION_STAKE = 900.0

#: 22-55% of 15m bars on six of eleven production pairs are gap-fill (FR-020,
#: SC-009); see `zero_volume_share`'s docstring for the exact freqtrade behaviour
#: this measures.
_GAPFILL_CRITERION = zero_volume_share


# =============================================================================================
# ExitConfig
# =============================================================================================


@dataclass(frozen=True)
class ExitConfig:
    """The parameters the exit policy reads (data-model.md), immutable, hashable.

    COMPOSITION OVER RESTATEMENT: this wraps a `utils.ExitPolicy.ExitParams` instance
    rather than re-declaring its eleven fields as a second dataclass. data-model.md's
    table lists eleven names; `ExitParams` (the actual argument `simulate_forward`
    reads) has a twelfth, `timeframe_minutes`, added in Phase 2 after the data model
    was written. The contract's hash rule -- "every parameter `simulate_forward`
    reads, including `fee_per_side`" (contracts/outcome-table.md) -- is the binding
    rule, and it supersedes the data model's field count: omitting
    `timeframe_minutes` from the hash would under-key the cache (two configs that
    differ only in timeframe would collide), which is exactly the staleness bug
    FR-007 exists to prevent. Wrapping, rather than subclassing a frozen dataclass or
    retyping the fields, means that rule only has to be satisfied once, in
    `ExitParams.__post_init__`, and `config_hash` can hash `asdict(self.params)`
    wholesale with no field list to keep in sync by hand.
    """

    params: ExitParams

    def __post_init__(self) -> None:
        if self.params.stoploss_grace_hours is None or self.params.stoploss_grace_level is None:
            raise ValueError(
                "ExitConfig requires grace to be sourced from the operating point being "
                "modelled -- construct via ExitConfig.from_operating_point(), never "
                "ExitConfig(ExitParams()) with the (deliberately absent) grace default. "
                "See this module's docstring and NNNC_MLX_Open.json."
            )

    # -- the eleven data-model fields, delegated to `params` (no restatement) --------------
    @property
    def atr_stoploss_multiplier(self) -> float:
        return self.params.atr_stoploss_multiplier

    @property
    def atr_stoploss_cap(self) -> float:
        return self.params.atr_stoploss_cap

    @property
    def atr_stoploss_floor(self) -> float:
        return self.params.atr_stoploss_floor

    @property
    def use_atr_adaptive_stoploss(self) -> bool:
        return self.params.use_atr_adaptive_stoploss

    @property
    def use_volume_confirmation_stoploss(self) -> bool:
        return self.params.use_volume_confirmation_stoploss

    @property
    def static_stoploss(self) -> float:
        return self.params.static_stoploss

    @property
    def stoploss_grace_hours(self) -> float:
        return self.params.stoploss_grace_hours

    @property
    def stoploss_grace_level(self) -> float:
        return self.params.stoploss_grace_level

    @property
    def take_profit(self) -> float:
        return self.params.take_profit

    @property
    def max_hold_days(self) -> int:
        return self.params.max_hold_days

    @property
    def fee_per_side(self) -> float:
        return self.params.fee_per_side

    # -- feature 021 FR-001 amendment fields, delegated to `params` (no restatement) -------
    @property
    def roi(self) -> Optional[float]:
        return self.params.roi

    @property
    def cexit_enable_profit_checks(self) -> bool:
        return self.params.cexit_enable_profit_checks

    @property
    def config_hash(self) -> str:
        """First 16 hex chars of SHA-256 over the canonical JSON of every field
        `simulate_forward` reads -- `fee_per_side` included (contracts/outcome-table.md):
        a positive outcome is net of fees (FR-006a), so a fee change changes every
        outcome, and omitting it would be exactly the silent-staleness bug FR-007
        exists to prevent.

        Also folds in `utils.ExitPolicy.SIMULATION_LOGIC_VERSION`. FR-007 guards
        CONFIGURATION drift (an `ExitParams` field changing); it does nothing for CODE
        drift -- `simulate_forward`'s or `custom_exit_decision`'s BEHAVIOUR changing with
        every `ExitParams` field held fixed, which is exactly what the 2026-10-08 fee
        double-count fix was (and exactly how run 3 silently got the SAME `config_hash`
        as run 2's pre-fix-price code). Hashing the version constant alongside the
        params means a behavioural change with no parameter change still changes the
        hash -- PROVIDED the constant is bumped; see its docstring in `utils.ExitPolicy`.
        """
        canonical = json.dumps(
            {**asdict(self.params), "simulation_logic_version": SIMULATION_LOGIC_VERSION},
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_operating_point(
        cls,
        json_path: Path = OPEN_OPERATING_POINT_JSON,
        *,
        fee_per_side: float = 0.001,
    ) -> "ExitConfig":
        """The only constructor meant to build a table-ready config.

        Reads `atr_stoploss_multiplier` / `_cap` / `_floor` / `use_volume_confirmation_stoploss`
        and `static_stoploss` off the live `BaseStrategy` class attributes,
        `use_atr_adaptive_stoploss` off `BaseNNStrategy` (True for the NN families
        the Open operating point belongs to -- `BaseStrategy`'s own default is
        False), and grace / take-profit / hold-cap off `json_path`'s pinned
        `params.sell` block -- never off `ExitParams`' defaults, which is exactly the
        hazard Phase 2 flagged (three different grace configurations exist; a
        default would silently simulate an exit the strategy does not run).

        `fee_per_side` has no single live source (freqtrade reads it from the
        exchange's fee tier at runtime, not from strategy code -- see
        `test_exit_policy_fixed_params.py`'s equivalent note) so it is the one field
        callers may still override explicitly.

        Feature 021's FR-001 amendment adds `cexit_enable_profit_checks` (read off the
        same JSON `sell` block as take_profit/max_hold_days -- it IS a freqtrade
        Parameter) and `roi` (read off `NNNCStrategy.minimal_roi`, a plain class
        attribute, NOT a Parameter and NOT in the JSON -- `BaseStrategy.minimal_roi`
        is {"0": 0.03}; the Open operating point's family overrides it to 0.05 in
        Python, not in the hyperopt JSON, so this is the only live source).
        """
        payload = json.loads(Path(json_path).read_text())
        sell = payload["params"]["sell"]
        roi_table = NNNCStrategy.minimal_roi
        if set(roi_table.keys()) != {"0"}:
            raise ValueError(
                f"NNNCStrategy.minimal_roi is no longer a flat table ({roi_table!r}) -- "
                "ExitConfig.from_operating_point assumes a single '0' entry (feature 021 "
                "FR-001 amendment); update this constructor before trusting the result."
            )
        params = ExitParams(
            atr_stoploss_multiplier=BaseStrategy.atr_stoploss_multiplier,
            atr_stoploss_cap=BaseStrategy.atr_stoploss_cap,
            atr_stoploss_floor=BaseStrategy.atr_stoploss_floor,
            static_stoploss=BaseStrategy.stoploss,
            use_atr_adaptive_stoploss=BaseNNStrategy.use_atr_adaptive_stoploss,
            use_volume_confirmation_stoploss=BaseStrategy.use_volume_confirmation_stoploss,
            stoploss_grace_hours=float(sell["stoploss_grace_hours"]),
            stoploss_grace_level=float(sell["stoploss_grace_level"]),
            take_profit=float(sell["cexit_take_profit"]),
            max_hold_days=int(sell["cexit_max_days"]),
            fee_per_side=fee_per_side,
            cexit_enable_profit_checks=bool(sell["cexit_enable_profit_checks"]),
            roi=float(roi_table["0"]),
        )
        return cls(params)


# =============================================================================================
# OutcomeTable
# =============================================================================================


@dataclass(frozen=True)
class OutcomeTable:
    """One `(dataset, ExitConfig)`'s worth of `OutcomeRecord`s (data-model.md)."""

    records: pd.DataFrame
    config_hash: str
    exit_config: ExitConfig
    dataset_id: str
    built_at: pd.Timestamp


def _rolling_rvol_series(volume: pd.Series, window: int = 20, min_periods: int = 5) -> np.ndarray:
    """Vectorised form of `utils.ExitPolicy.rolling_rvol`, which is defined only at
    "the last bar" (the live, per-call shape). Computing one rvol per bar for an
    entire pair needs the same window/min_periods applied at every position, so this
    reproduces that definition elementwise rather than calling the scalar function
    ~200k times -- same arithmetic, vectorised, not a second implementation of it
    (the function has no clamp/branch of its own to duplicate)."""
    vol = volume.astype(float)
    vol_mean = vol.rolling(window, min_periods=min_periods).mean()
    rvol = vol / vol_mean.replace(0.0, np.nan)
    return rvol.fillna(0.0).to_numpy()


def _gapfill_share(df: pd.DataFrame, start_index: int, hold_bars: int) -> float:
    """Share of the hold window (the bars strictly after entry, through and
    including the close bar) that is gap-fill, reusing
    `research.clfstudy.data.zero_volume_share` rather than re-deriving its
    volume==0 criterion (FR-020)."""
    if hold_bars <= 0:
        return 0.0
    window = df.iloc[start_index + 1: start_index + 1 + hold_bars]
    share = _GAPFILL_CRITERION(window)
    return 0.0 if pd.isna(share) else float(share)


def build_outcome_table(
    df: pd.DataFrame,
    pair: str,
    exit_config: ExitConfig,
    *,
    stake: float = PRODUCTION_STAKE,
) -> OutcomeTable:
    """One `OutcomeRecord` per bar of `df` (T014).

    `df` must carry `date`, `open`, `high`, `low`, `close`, `volume` and
    `atr_pct_roll` (the production ATR input -- `research.clfstudy.data.populate`
    adds it via `DataframePopulator`'s MINIMAL set, matching
    `BaseStrategy._atr_bar_inputs`'s live source exactly). `guard_metric` is read if
    present but not required -- `DataframePopulator` already produces it
    unconditionally alongside `atr_pct_roll` (feature 021 FR-001 amendment;
    `metric_overbought` cannot be modelled without it, degrading gracefully to "not
    modelled" per `simulate_forward`'s `guard_metric=None` convention if it is ever
    absent from an older/different populator run).

    Every exit decision is delegated to `simulate_forward` (via `open_`/`guard_metric`,
    feature 021's FR-001 amendment, for `metric_overbought`/`take_profit`/`unclog_12h`/
    `unclog_1d`, and via `exit_config.params.roi` for the native ROI exit); this function
    supplies its inputs (atr_pct, rvol, open/high/low/close, guard_metric, start_index)
    and turns the `(profit, hold_bars, close_reason)` result into the record's columns --
    it contains no exit arithmetic or exit decision logic of its own.
    """
    required = {"date", "open", "high", "low", "close", "volume", "atr_pct_roll"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_outcome_table: df missing required columns {sorted(missing)}")

    df = df.reset_index(drop=True)
    n = len(df)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr_pct = df["atr_pct_roll"].to_numpy(dtype=float)
    guard_metric = df["guard_metric"].to_numpy(dtype=float) if "guard_metric" in df.columns else None
    rvol = (
        _rolling_rvol_series(df["volume"])
        if exit_config.use_volume_confirmation_stoploss
        else None
    )

    quote_volume = df["volume"].to_numpy(dtype=float) * close
    fillable_stake = quote_volume / BaseStrategy.QUOTE_VOLUME_HEADROOM_MULT
    fillable_fraction = np.clip(fillable_stake / stake, 0.0, 1.0) if stake > 0 else np.ones(n)

    profits = np.empty(n, dtype=float)
    hold_bars = np.empty(n, dtype=np.int64)
    close_reasons: list[str] = [""] * n
    gapfill_shares = np.empty(n, dtype=float)

    for i in range(n):
        profit, bars, reason = simulate_forward(
            high, low, close, atr_pct, rvol, start_index=i, params=exit_config.params,
            open_=open_, guard_metric=guard_metric,
        )
        profits[i] = profit
        hold_bars[i] = bars
        close_reasons[i] = reason
        gapfill_shares[i] = _gapfill_share(df, i, bars)

    records = pd.DataFrame({
        "date": df["date"],
        "pair": pair,
        OUTCOME_COLUMN_PROFIT: profits,
        OUTCOME_COLUMN_HOLD_BARS: hold_bars,
        OUTCOME_COLUMN_CLOSE_REASON: close_reasons,
        OUTCOME_COLUMN_POSITIVE: profits > 0.0,
        OUTCOME_COLUMN_FILLABLE: fillable_fraction >= 1.0,
        OUTCOME_COLUMN_FILLABLE_FRACTION: fillable_fraction,
        OUTCOME_COLUMN_GAPFILL_SHARE: gapfill_shares,
    })

    return OutcomeTable(
        records=records,
        config_hash=exit_config.config_hash,
        exit_config=exit_config,
        dataset_id=f"{pair}|{exit_config.params.timeframe_minutes}m",
        built_at=pd.Timestamp.now(tz=timezone.utc),
    )


# =============================================================================================
# Cache (T017) -- feather per (pair, config_hash) + JSON sidecar. FR-007/SC-004: requesting a
# table whose stored hash differs from the requested config's hash is an ERROR, never a
# silent reuse.
# =============================================================================================


class CacheMismatchError(ValueError):
    """Raised when a cached table's sidecar does not agree with the config hash it
    is stored under -- corruption or a manually-edited sidecar, never silently
    trusted (FR-007)."""


def _base(pair: str) -> str:
    return pair.split("/")[0]


def _feather_path(pair: str, config_hash: str) -> Path:
    return CACHE_DIR / f"{_base(pair)}__{config_hash}.feather"


def _sidecar_path(config_hash: str) -> Path:
    return CACHE_DIR / f"{config_hash}.json"


def _sidecar_payload(table: OutcomeTable) -> dict:
    return {
        "config_hash": table.config_hash,
        "dataset_id": table.dataset_id,
        "built_at": table.built_at.isoformat(),
        **asdict(table.exit_config.params),
    }


def _hash_from_sidecar(payload: dict) -> str:
    """Recompute the hash an `ExitConfig` built from `payload`'s params would
    produce, so a tampered/corrupted sidecar is DETECTED rather than trusted.

    Any failure to reconstruct a valid `ExitParams` (a missing/renamed field, a bad
    type) is a corrupt sidecar, not a crash -- the caller turns it into
    `CacheMismatchError` rather than letting a `KeyError`/`TypeError` escape."""
    param_fields = {f: payload[f] for f in asdict(ExitParams()).keys()}
    return ExitConfig(ExitParams(**param_fields)).config_hash


def save_table(table: OutcomeTable, pair: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feather_path = _feather_path(pair, table.config_hash)
    table.records.reset_index(drop=True).to_feather(feather_path)
    sidecar = _sidecar_path(table.config_hash)
    sidecar.write_text(json.dumps(_sidecar_payload(table), indent=2, sort_keys=True))
    return feather_path


def load_table(pair: str, exit_config: ExitConfig) -> pd.DataFrame:
    """Load the cached table for `(pair, exit_config)`.

    Missing cache -> `FileNotFoundError` (a plain cache miss: rebuild). Present but
    whose sidecar hash disagrees with what its own stored params recompute to, or
    with the filename's embedded hash -> `CacheMismatchError` -- this is the "error,
    never silent reuse" case the contract requires (FR-007/SC-004), distinct from a
    cache miss.
    """
    requested_hash = exit_config.config_hash
    feather_path = _feather_path(pair, requested_hash)
    sidecar = _sidecar_path(requested_hash)
    if not feather_path.exists() or not sidecar.exists():
        raise FileNotFoundError(
            f"no cached outcome table for pair={pair!r} config_hash={requested_hash!r}"
        )
    payload = json.loads(sidecar.read_text())
    stored_hash = payload.get("config_hash")
    try:
        recomputed_hash = _hash_from_sidecar(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheMismatchError(
            f"cache for pair={pair!r}: sidecar {sidecar} is corrupt/unreadable ({exc}). "
            "Refusing to reuse it (FR-007) -- rebuild."
        ) from exc
    if stored_hash != requested_hash or recomputed_hash != requested_hash:
        raise CacheMismatchError(
            f"cache for pair={pair!r}: sidecar {sidecar} records config_hash="
            f"{stored_hash!r}, recomputes to {recomputed_hash!r}, but the requested "
            f"ExitConfig hashes to {requested_hash!r}. Refusing to reuse a stale/"
            "tampered cache (FR-007) -- rebuild."
        )
    return pd.read_feather(feather_path)


def status() -> list[dict]:
    """Every cached `(pair, config_hash)` and its sidecar summary (`--status`)."""
    rows = []
    if not CACHE_DIR.exists():
        return rows
    for feather_path in sorted(CACHE_DIR.glob("*.feather")):
        stem = feather_path.stem
        if "__" not in stem:
            continue
        base, config_hash = stem.split("__", 1)
        sidecar = _sidecar_path(config_hash)
        payload = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        rows.append({
            "pair_base": base,
            "config_hash": config_hash,
            "dataset_id": payload.get("dataset_id"),
            "built_at": payload.get("built_at"),
            "rows": len(pd.read_feather(feather_path)),
        })
    return rows


# =============================================================================================
# CLI (T018): --build / --status / --verify
# =============================================================================================


def _load_and_build(pair: str, exit_config: ExitConfig, timerange: Optional[str]) -> OutcomeTable:
    df = populate(load_pair(pair))
    if timerange:
        start_str, end_str = timerange.split(":")
        start, end = pd.Timestamp(start_str, tz="UTC"), pd.Timestamp(end_str, tz="UTC")
        df = df[(df["date"] >= start) & (df["date"] < end)].reset_index(drop=True)
    return build_outcome_table(df, pair, exit_config)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m research.outcomes.table")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="build and cache the table")
    group.add_argument("--status", action="store_true", help="list what is cached")
    group.add_argument("--verify", action="store_true", help="rebuild one pair, assert identical (SC-003)")
    parser.add_argument("--pairs", nargs="+", default=None, help="pairs to build/verify")
    parser.add_argument("--timerange", default=None, help="START:END, ISO dates")
    args = parser.parse_args(argv)

    exit_config = ExitConfig.from_operating_point()

    if args.status:
        for row in status():
            print(row)
        return

    pairs = args.pairs or []
    if args.build:
        if not pairs:
            parser.error("--build requires --pairs")
        for pair in pairs:
            table = _load_and_build(pair, exit_config, args.timerange)
            path = save_table(table, pair)
            print(f"{pair}: {len(table.records)} rows -> {path}")
        return

    if args.verify:
        if not pairs:
            parser.error("--verify requires --pairs")
        for pair in pairs:
            rebuilt = _load_and_build(pair, exit_config, args.timerange)
            cached = load_table(pair, exit_config)
            same = rebuilt.records.reset_index(drop=True).equals(cached.reset_index(drop=True))
            print(f"{pair}: identical={same}")
            if not same:
                sys.exit(1)
        return


if __name__ == "__main__":
    main()

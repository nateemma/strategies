# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0302, C0303, C0325, C0411, C0413
# pragma pylint: disable=W0105, W1203, W1309, W1514, W0613, W0621,
# type: ignore
# pylint: disable=import-error

"""
NNNCStrategy - Base class for Neural Network N-ary Classification strategies
"""

import sys
from pathlib import Path
from pandas import DataFrame
import numpy as np
import traceback
from freqtrade.strategy import DecimalParameter

# Add parent directory to path to import NNNC
group_dir = str(Path(__file__).parent)
sys.path.append(group_dir)

from Framework.BaseNNStrategy import BaseNNStrategy, HAS_MLX, StrategyConfig
from utils.Environment import Environment
from Framework.BaseStrategy import BaseStrategy, ScalerType, MarketRegime, TradingAction, FlowDirection, MomentumDirection, RiskLevel, GANType
from typing import Any
import NNNClassifier
from Framework.BaseStrategy import ModelType, NormalizationType
from utils.EntryFramework import AUTO, EntryFrameworkMixin


class NNNCStrategy(EntryFrameworkMixin, BaseNNStrategy):
    # Feature 029: NNNC shares SimpleTrigger's entry framework (utils.EntryFramework) -- stop
    # invariants, per-family exits, the regime gate, the trigger gate -- with the MODEL as its
    # entry source. "model" is today's entry exactly: the model's prediction against its
    # threshold, through BaseStrategy's unchanged guard chain. Model entries are tagged with the
    # training label's family (utils.LabelFamilies; gbb_path -> reversion). Every framework
    # feature is off until a subclass declares it.
    ENTRY_MODE = "model"
    MODEL_ENTRY_FAMILY = AUTO

    # Strategy configuration
    strategy_config = StrategyConfig(
        normalization=NormalizationType.ROLLING_ROBUST,
        model_type=ModelType.KERAS,
        needs_training=True,
        seq_len=16,
    )

    apply_label_conflict_resolution = True  # resolve buy/sell label conflicts before training

    # Model architecture is selected declaratively: subclasses set
    # ``classifier_type`` to a ``NNNClassifier.ClassifierType`` (Keras) or a
    # ``ClassifierTypeMLX`` (MLX, via MLXClassifierMixin) value instead of
    # overriding ``get_classifier_type()``. The framework reads it through
    # ``get_classifier_type()`` below (see BaseNNStrategy: it caches
    # ``self.classifier_type = self.get_classifier_type()`` before building).
    classifier_type = NNNClassifier.ClassifierType.LSTM

    # NNNC-family ATR-adaptive stoploss tuning. NNNC has no model-driven
    # soft exit (unlike NNMT's strategy_custom_exit on the profit/regime
    # heads), so the stop is the only safety net for losing trades.
    # BaseNNStrategy defaults to multiplier=2.5, floor=-0.04, cap=-0.02
    # (range [-2%, -4%]); on NNNC that fires too aggressively and kills
    # trades that would have recovered (backtest 2026-05-26: 167 stops at
    # -2.53% avg vs static -5% baseline at 75 stops, net -8pp).
    # Loosen to range [-4%, -6%] — adaptive but with enough room for
    # the unclog / model_exit winners to materialise.
    atr_stoploss_multiplier = 4.0
    atr_stoploss_floor = -0.06   # loosest stop allowed (most negative)
    atr_stoploss_cap = -0.04     # tightest stop allowed (closest to zero)

    # First-hour stop-loss grace (BaseStrategy): wide stop for the first 3h,
    # then tighten to the ATR-adaptive stop. Restores the pre-confirm_trade_exit
    # "don't stop out on entry noise" behaviour, safely. See NNMT sweep (3h/-0.15).
    # Grace window: 6.5h. Swept 2.0-12.0h on the post-fix exit stack (trailing
    # off, ROI 0.05). 6.5h gives the best worst-window Calmar (18.09 vs 16.30 at
    # 5.6h), +1.10pp summed, and leaves W3 EXACTLY unchanged.
    #
    # There is a CLIFF between 8h and 9h: W3 collapses from +2.01% to +0.07% and
    # its Calmar from 17.42 to 0.15. 7.0h scores higher on raw summed return
    # (18.61 vs 18.29) but on worse worst-window Calmar and a W3 decline; 6.5h
    # keeps margin from the cliff. The upper bound is raised from 6.0 to 12.0 so
    # the value is inside the range rather than pinned to it.
    #
    # 2026-10-13: default 24h (was 6.5h). The NNNC framework study
    # (research/nnnc_framework/, pre-registered f879338) tested 24h walk-forward on
    # f00-f14 on four models: ADOPTED -- PASS on arm D (+5.56/+1.69 pp early/late),
    # GBR and NNNC_MLX, GFM late -1.25; 15-fold totals arm D 42.71 -> 49.97, GBR
    # 43.65 -> 53.46, GFM 25.61 -> 39.76, NNNC_MLX 25.32 -> 35.19; paired and risk
    # UNCHANGED on all four. Post-grace stop exits fall by more than half (arm D
    # 469 -> 199): those trades recover and leave through unclog. The 8-9h cliff
    # above came from the 3-window sweep on an older stack and did not reappear.
    # 24h was the edge of the screened grid; the range now reaches 48h.
    stoploss_grace_hours = DecimalParameter(
        0.0, 48.0, default=24.0, decimals=1, space="sell", load=True, optimize=True
    )
    stoploss_grace_level = DecimalParameter(
        -0.30, -0.05, default=-0.15, decimals=2, space="sell", load=True, optimize=True
    )

    # 2026-10-13: held-pair add ON by default, in-profit only (031 follow-up,
    # research/held_pair/, pre-registered cb158ba, verdicts e398714). A fresh entry signal on a
    # pair holding a trade that is IN PROFIT adds one stake (freqtrade position adjustment); the
    # trade's clocks then count from the add. f00-f14, all four models PASS: ARO +3.19/+3.31 pp
    # early/late (paired IMPROVED p 0.006), GBR +2.70/+3.32 (IMPROVED), GFM +2.59/+2.65, MLX
    # +0.69/+0.82; Sharpe up, max drawdown not up, exposure +3-10%. EXPLORATORY: both halves had
    # been seen for the held-pair question; SimpleTrigger reversion confirmed it on 2022 (+5.85).
    # The 2026-09 profit-gated pyramiding test on NNNC_MLX (research/posadj, 5d6ef83: not
    # resolved) used 6.5 h grace, no clock restart and 2 adds; MLX is again ~flat here.
    # Momentum-labelled models are exempt (`_resolve_held_pair_add` below): the add failed on the
    # momentum book (2022 -38 pp). A class declares HELD_PAIR_ADD = None to turn it off.
    HELD_PAIR_ADD = {"in_profit_only": True}

    @classmethod
    def _resolve_held_pair_add(cls) -> None:
        super()._resolve_held_pair_add()
        if cls.HELD_PAIR_ADD_RESOLVED is None or cls._model_family() != "momentum":
            return
        below = cls.__mro__[: cls.__mro__.index(NNNCStrategy)]
        if not any("HELD_PAIR_ADD" in k.__dict__ for k in below):
            cls._held_pair_add_off()

    # HORIZON is now inherited from BaseNNStrategy → TrainingConfig.HORIZON.
    # Production = 3. The 2026-05-28 univariate H sweep landed at H=3 at
    # threshold=0.005 (+21.68% vs +15.97%); the 2026-05-30 learnability
    # sweep then showed (H=3, threshold=0.003) is the right joint setting.
    # Threshold lives in TrainingConfig too — change them together, not
    # in isolation. Don't reintroduce the override here.

    # Freqtrade's NATIVE trailing stop is disabled for this family.
    #
    # ft_stoploss_adjust runs the trailing block IN ADDITION to custom_stoploss,
    # not instead of it, and `trailing_only_offset_is_reached = False` means the
    # guard never skips - so it ran on every candle of every trade and raised the
    # stop above the configured -5%.
    #
    # This completes a fix that was already half-made. BaseStrategy.custom_stoploss
    # carries a comment recording that returning self.stoploss there "created an
    # implicit trailing stop even with trailing_stop=False" and caught a large
    # fraction of winners; that half was fixed with `return 1.0`. The native
    # trailing stop was left on, doing the same thing from the other side.
    #
    # Measured 2026-09-02 across three pinned windows, same entry config, same
    # model, no retrain:
    #   W1  +3.95% PF2.33 Calmar26.4 win64.7%  ->  +7.11% PF3.70 Calmar48.7 win75.5%
    #   W2  +5.75% PF1.66 Calmar13.1 win69.8%  ->  +6.84% PF1.81 Calmar13.7 win73.7%
    #   W3  +1.72% PF3.86 Calmar17.8 win75.0%  ->  +2.01% PF4.57 Calmar20.8 win85.0%
    #   summed 11.43% -> 15.95%, +4.53pp; better in ALL THREE windows on return,
    #   PF, Calmar and win rate.
    #
    # Exit mix: stops 83 -> 40, take-profit 42 -> 73 at a higher mean. Trades that
    # were being cut on noise now survive to reach their targets. The surviving
    # stops are LARGER individual losses (-4.23% vs -0.87% mean) because they are
    # genuine losers rather than noise casualties - the gain comes entirely from
    # the win side, and W2 drawdown worsens slightly (2.30% -> 2.62%).
    #
    # trailing_stop_positive / _offset are left as-is: they are unreachable while
    # ROI fires at +3.00% and the offset is 0.03, so a trade never survives above
    # the offset for the positive branch to engage.
    trailing_stop = False

    # ROI 0.05, up from the base 0.03. Swept 0.03-0.06 plus two ladders on the
    # post-fix stack: every variant beat flat 0.03, consistently in all three
    # windows, with trade counts IDENTICAL (49/95/20) so this moves only where
    # trades exit. 0.05 is where W1 peaks and W3 saturates; 0.06 gains 0.13pp in
    # W2 alone. Ladders were rejected - flat 0.06 and ladder 8/5/3 differ by
    # 0.08pp, far below this test's resolution, for two extra parameters.
    #
    # Raising ROI does NOT let winners run: it hands control to cexit_take_profit,
    # which caps them at a +3.55% mean. Raising THAT was swept and rejected -
    # every value from 0.030 to 0.060 lost, and disabling profit checks entirely
    # was worst of all, so 0.024 is doing real work.
    minimal_roi = {"0": 0.05}

    # Inherit the shared entry/exit controls; override only what differs.
    #
    # Values RE-VALIDATED 2026-09-02 on the REPAIRED exit stack (native trailing
    # stop off - see regime/exit/README.md), by a full 4x4 rvol x guard_metric
    # grid plus a 7-point rvol curve at thr=0.69 across three windows
    # (W1 2025-09..2026-08, W2 2024-06..2025-05, W3 2023). See regime/README.md.
    #
    #   entry_rvol_threshold 2.0  WAS 3.1, and 3.1 was an ARTIFACT of the broken
    #       exit. It had been picked on worst-window Calmar, and won that only
    #       because loose rvol collapsed in W1 - rvol 1.5 scored +0.04% there
    #       under the old stack and scores +9.57% now. The collapse WAS the
    #       trailing stop: the marginal trades a looser gate admits were exactly
    #       the ones being cut on noise.
    #
    #       The curve now has a real interior optimum, peaking at 1.5 and
    #       falling off BOTH sides (summed return over three windows):
    #           0.5   1.0   1.5   2.0   2.5   3.1   4.0
    #         22.26 22.24 26.98 25.32 21.29 18.29  9.22
    #
    #       The per-window optimum is still not identifiable (W1 wants 2.5, W2
    #       and W3 want 1.5), so MINIMAX REGRET decides: 2.0 has the lowest max
    #       regret of all seven points (2.09pp, vs 2.34 at 1.5 and 4.54 at 3.1).
    #       2.0 also dominates 1.5 on EVERY metric in W1, the deployment regime
    #       that carries the headline verdict - 10.40% vs 9.57%, drawdown 1.50%
    #       vs 2.25%, Calmar 36.45 vs 22.30, Sortino 1.82 vs 0.97 - and it was
    #       the best summed-return cell under the OLD stack too, so it is the one
    #       value that is good under both. Worth +7.03pp (18.29% -> 25.32%).
    #
    #       NOT promoted to the base class: rvol 2.0 costs NNMT 7.62pp
    #       (19.48% -> 11.85%), so the optimum is family-specific. Both families
    #       want LOOSE rvol; NNMT is already there at 0.5, NNNC was the outlier.
    #
    #   entry_guard_threshold 0.3  Uniformly dominant, and re-confirmed on the
    #       repaired stack: 0.3 >= 0.0 >= -0.4 on both summed return and
    #       worst-window Calmar in ALL 16 cells of the new grid, with 0.6 still
    #       byte-identical to 0.3 at every rvol (verified by trade hash). A
    #       saturating plateau, not a peak - which is precisely why it did not
    #       move when the exit stack changed underneath it.
    #
    #   adx 38 / atr_pct 0.006 / bb_width 0.06 are LOAD-BEARING, not slack.
    #       Loosening adx to 30 costs -3.92%; atr_pct to 0.003 costs -7.10%.
    #       Removing bb_width is invisible on W1 (-0.05pp) but turns W3 from
    #       +3.01% into -0.23% - it is the guard that carries the distant era.
    # ----------------------------------------------------------------
    # NNNC family defaults, updated 2026-09-06 to the model-driven operating
    # point (regime/CANDIDATE_STATUS.md, LIQUIDITY_STUDY.md).
    #
    # WHAT CHANGED and WHY: the seven-guard chain selected ~99% of entries, so
    # the model contributed nothing and a PERFECT model was worth only 0.68pp
    # more than the real one -- no model, GAN or architecture work could ever
    # pay at the old point. Opening the six QUALITY guards (keeping `volume > 0`)
    # and raising prediction_threshold to 0.80 moves selection to the model:
    # ~25pp of headroom, 4.5x the trades, and positive in all three windows on
    # 3 of 3 seeds.
    #
    # HONESTLY: this is NOT a higher-return setting. Across seeds its summed
    # return is +27.20 vs the old point's +24.13 with sd 6.38 -- indistinguishable
    # -- and it LOSES the W1 acceptance window on 3 of 3 seeds. It is chosen for
    # headroom and MEASURABILITY, not performance.
    #
    # The six guards are neutralised by threshold rather than by
    # `entry_enable_guards = False`, because that flag also disables the
    # `volume > 0` data-sanity check, which sits in the same block. Out-of-range
    # values are honoured, not clamped (verified 2026-09-05).
    #
    # BLAST RADIUS: ~60 NNNC variants carry no <Strategy>.json and inherit these
    # defaults -- including every NNNC_DDPM_*, NNNC_CGP_* and NNNC_WGAN* GAN
    # variant. Their historical arm records were produced at the OLD defaults and
    # will not reproduce. Variants WITH their own .json (NNNC_MLX included) are
    # unaffected until that file is updated.
    # ----------------------------------------------------------------
    buy_params = { **BaseNNStrategy.buy_params,
        # six quality guards OPEN; `volume > 0` stays live via enable_guards
        "entry_adx_threshold": -1.0,
        "entry_atr_pct": -1.0,
        "entry_bb_width_threshold": -1.0,
        "entry_close_norm_threshold": 1.1,
        "entry_enable_guards": True,
        "entry_guard_threshold": 1.1,
        "entry_rvol_threshold": -1.0,
        "prediction_threshold": 0.8,
    }

    plot_config = {
        "main_plot": {
            "close": {"color": "lightsteelblue"},
        },
        "subplots": {
            "Diff": {
                "%train_buy": {"color": "lightgreen"},
                "%predict_prob_buy": {"color": "green"},
                "%train_sell": {"color": "orange"},
                "%predict_prob_sell": {"color": "red"},
            },
        },
    }

    # Label: gain threshold 0.02 rather than the framework default 0.007.
    # A 0.7% target against a -4%..-6% ATR stop plus round-trip fees cannot pay
    # however perfectly it is predicted -- at the ceiling, 0.02 turns W1 from
    # -18.89% to +26.54% (regime/LABEL_TRIGGER_GATE.md).
    #
    # SUPERSEDED 2026-09-06: this was first set here to avoid relabelling NNMT,
    # then promoted to Framework/TrainingConfig at the user's direction so all
    # families share it. Kept as an explicit restatement of the NNNC value; it
    # now matches the framework default rather than overriding it.
    #
    # Existing NNNC models were trained against 0.007. This affects the NEXT
    # retrain, not inference by a model already on disk.
    MIN_BUY_GAIN_THRESHOLD = 0.02
    MIN_SELL_LOSS_THRESHOLD = 0.02
    
    # Feature 030: the model key (Framework/ModelKey.py). This class declares the names it
    # owns; the lists merge along the MRO. ATTRIBUTES/METHODS shape trained weights;
    # IGNORED act only after the model. The audit test keeps every name classified.
    MODEL_KEY_ATTRIBUTES = (
        "AUG_DILATION",
    )
    # NNNC_AblateG's `_thr`/`_guards` feed buy_params (prediction threshold, guards) and `_os`
    # is the os module it reads them with: all post-model.
    MODEL_KEY_IGNORED = (
        "ORACLE_PRECISION", "ORACLE_SEED", "ORACLE_SUBSAMPLE", "TWIN_TRIGGER",
        "_guards", "_os", "_thr",
    )

    def get_classifier_type(self):
        """Return the type of classifier used for training/predicting"""
        return self.classifier_type

    def get_classifier(
        self, classifier_type, pair, seq_len, num_features
    ) -> Any:
        """Return the classifier used for training/predicting"""
        clf, _ = NNNClassifier.create_classifier(
            classifier_type, pair, num_features, seq_len, 3
        )
        return clf

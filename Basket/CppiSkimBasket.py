"""
CppiSkimBasket — the income configuration: CPPI exposure + profit-skim.

CPPI defends a capital floor on the way down; the profit-skim overlay takes a
fixed fraction of every new equity high off the table into a reserved bucket
(the income you'd withdraw in live). Skim is fixed here (a fixed income POLICY,
not a tuned knob), so a hyperopt tunes only the CPPI parameters — you decide the
income rate, the optimiser decides how to run the risky sleeve underneath it.

Watch the ``INCOME`` log lines (banked cumulative, monthly) for the income
stream; judge this on income + drawdown of the at-risk sleeve, not total return
(banked cash sits idle in backtest and can't compound).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from CppiBasket import CppiBasket


class CppiSkimBasket(CppiBasket):
    # Fixed income policy (plain values → not part of the hyperopt space).
    profit_skim_enable = True
    skim_fraction = 0.25  # bank 25% of each new equity high

    # Risk policy (plain values → pinned, not tuned):
    #  * multiplier 1.5 keeps modest upside participation at ~half the drawdown
    #    of the levered default (3.0), and stops hyperopt levering into whatever
    #    coin happens to moon in-sample.
    #  * cap any single coin at 20% of the book so a runaway winner can't quietly
    #    become the portfolio between rebalances.
    cppi_multiplier = 1.5
    max_position_weight = 0.20

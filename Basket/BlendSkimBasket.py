"""
BlendSkimBasket — the income configuration: Blend + profit-skim.

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
from BlendBasket import BlendBasket


class BlendSkimBasket(BlendBasket):
    # Fixed income policy (plain values → not part of the hyperopt space).
    profit_skim_enable = True
    skim_fraction = 0.25  # bank 25% of each new equity high

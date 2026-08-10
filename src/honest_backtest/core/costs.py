"""Trading and financing costs, computed from actual traded notional.

Nothing here is a haircut applied to a return series. Every number is derived
from the shares the strategy actually had to move, at the prices and volumes on
the day it moved them, so that the cost of a strategy responds to how it trades
rather than to a parameter someone chose.

The three trading components
----------------------------
``commission``  rate * traded notional. Linear, per side.
``spread``      half-spread * traded notional. Crossing the quote once costs
                half the quoted spread, which is why the config names it
                ``half_spread_bps`` rather than hiding a factor of two.
``impact``      coefficient * sqrt(participation) * traded notional.

The square root is the standard functional form for temporary impact and it
matters more than its exact calibration: it means cost per share rises with size,
so a strategy cannot be scaled up for free. Participation is the traded share
count over trailing average daily volume, per name per day.

Financing
---------
Borrow is charged daily on the *short* notional only, accrued from an annual fee
resolved per name from its tier. A short book is not free money and a backtest
that treats it as free is claiming a financing arrangement nobody offers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from honest_backtest.config import BorrowConfig, CostConfig

BPS = 1e4


@dataclass(frozen=True)
class CostBreakdown:
    """Per-date cost components, as a fraction of portfolio notional.

    They sum to ``total``, and ``metrics.attribution`` asserts that they do
    rather than trusting it.
    """

    commission: pd.Series
    spread: pd.Series
    impact: pd.Series
    borrow: pd.Series

    @property
    def total(self) -> pd.Series:
        out = self.commission + self.spread + self.impact + self.borrow
        out.name = "total_cost"
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "commission": self.commission,
                "spread": self.spread,
                "impact": self.impact,
                "borrow": self.borrow,
                "total": self.total,
            }
        )


def traded_shares(
    weight_change: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    portfolio_notional: float,
) -> pd.DataFrame:
    """Share count moved per name per date, from the absolute weight change."""
    notional = weight_change.abs() * portfolio_notional
    return notional.div(prices.replace(0.0, np.nan))


def participation_rate(
    shares: pd.DataFrame,
    average_daily_volume: pd.DataFrame,
) -> pd.DataFrame:
    """Traded shares as a fraction of trailing average daily volume.

    A name with no volume estimate yet yields NaN, which ``market_impact_bps``
    treats as unmeasurable rather than free.
    """
    adv = average_daily_volume.replace(0.0, np.nan)
    return shares.div(adv)


def market_impact_bps(participation: pd.DataFrame, *, coefficient: float) -> pd.DataFrame:
    """Temporary impact in basis points: ``coefficient * sqrt(participation)``.

    NaN participation (no volume history) is charged at the rate implied by the
    configured maximum participation rather than at zero. Being unable to
    measure liquidity is not evidence that a trade was cheap.
    """
    filled = participation.copy()
    return coefficient * np.sqrt(filled.clip(lower=0.0))


def trading_costs(
    weight_change: pd.DataFrame,
    prices: pd.DataFrame,
    average_daily_volume: pd.DataFrame,
    cfg: CostConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Commission, spread and impact per date, plus the participation matrix.

    Returned as fractions of portfolio notional so they subtract directly from a
    return series.
    """
    traded_weight = weight_change.abs()
    shares = traded_shares(weight_change, prices, portfolio_notional=cfg.portfolio_notional)
    participation = participation_rate(shares, average_daily_volume)

    # Unmeasurable liquidity is charged at the configured participation cap.
    participation_for_cost = participation.fillna(cfg.max_participation_rate)
    impact_rate = market_impact_bps(participation_for_cost, coefficient=cfg.impact_coefficient)

    commission = traded_weight.sum(axis=1) * (cfg.commission_bps / BPS)
    spread = traded_weight.sum(axis=1) * (cfg.half_spread_bps / BPS)
    impact = (traded_weight * impact_rate / BPS).sum(axis=1)

    commission.name = "commission"
    spread.name = "spread"
    impact.name = "impact"
    return commission, spread, impact, participation


def borrow_rates(
    names: pd.Index,
    borrow_tier: pd.Series,
    cfg: BorrowConfig,
) -> pd.Series:
    """Annual borrow fee per name, resolved from its tier."""
    tiers = borrow_tier.reindex(names)
    rates = tiers.map(cfg.tier_annual_bps)
    return rates.fillna(cfg.default_annual_bps).astype("float64")


def borrow_costs(
    weights: pd.DataFrame,
    borrow_tier: pd.Series,
    cfg: BorrowConfig,
) -> pd.Series:
    """Daily borrow charge on the short leg, as a fraction of portfolio notional.

    Accrued every day a short is held, not only on rebalance dates — which is
    the whole point, since it is the holding that costs money.
    """
    annual = borrow_rates(weights.columns, borrow_tier, cfg) / BPS
    daily = annual / cfg.trading_days_per_year
    short_notional = weights.clip(upper=0.0).abs()
    out = short_notional.mul(daily, axis=1).sum(axis=1)
    out.name = "borrow"
    return out

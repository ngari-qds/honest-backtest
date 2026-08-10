"""Where the gross return went, and how much money the strategy could hold.

Attribution here is an identity, not an estimate:

    net = gross - commission - spread - impact - borrow

``attribute`` asserts that identity holds to floating-point tolerance rather
than assuming it. If a cost component is ever added without being wired into the
total, the assertion fires instead of the discrepancy quietly appearing as alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honest_backtest.config import CostConfig, MetricsConfig
from honest_backtest.core.costs import CostBreakdown

#: Tolerance for the accounting identity, per period.
CONSERVATION_TOLERANCE = 1e-12


class AttributionError(AssertionError):
    """Raised when net return does not equal gross minus the itemised costs."""


def attribute(
    gross: pd.Series,
    net: pd.Series,
    costs: CostBreakdown,
    *,
    periods_per_year: int,
) -> pd.DataFrame:
    """Annualised decomposition of gross return into net plus each cost component."""
    residual = (gross - costs.total) - net
    worst = float(residual.abs().max()) if len(residual) else 0.0
    if worst > CONSERVATION_TOLERANCE:
        raise AttributionError(
            f"Cost accounting does not conserve: largest discrepancy {worst:.3e} "
            "between (gross - itemised costs) and the reported net return."
        )

    def annualise(series: pd.Series) -> float:
        return float(series.mean() * periods_per_year)

    rows = {
        "gross_return": annualise(gross),
        "commission": -annualise(costs.commission),
        "spread": -annualise(costs.spread),
        "market_impact": -annualise(costs.impact),
        "borrow": -annualise(costs.borrow),
        "net_return": annualise(net),
    }
    frame = pd.DataFrame({"annualised_contribution": pd.Series(rows)})
    frame.index.name = "component"

    total_costs = -(
        frame.loc["commission", "annualised_contribution"]
        + frame.loc["spread", "annualised_contribution"]
        + frame.loc["market_impact", "annualised_contribution"]
        + frame.loc["borrow", "annualised_contribution"]
    )
    gross_annual = frame.loc["gross_return", "annualised_contribution"]
    frame["share_of_gross"] = (
        frame["annualised_contribution"] / gross_annual if gross_annual else np.nan
    )
    frame.attrs["total_annualised_cost"] = total_costs
    return frame


def capacity_estimate(
    gross: pd.Series,
    impact_cost: pd.Series,
    participation: pd.DataFrame,
    cost_cfg: CostConfig,
    metrics_cfg: MetricsConfig,
) -> pd.DataFrame:
    """Notional at which estimated impact eats a given share of gross return.

    ``impact_cost`` is the impact series the engine actually charged, taken
    straight from the ``CostBreakdown``, so this figure cannot drift away from
    the attribution table by being recomputed differently here.

    Impact scales as ``sqrt(participation)`` and participation scales linearly
    with notional, so impact cost scales as ``sqrt(notional)``. Solving

        impact(N) = base_impact * sqrt(N / base_notional) = erosion * gross

    for ``N`` gives the capacity figure.

    Read it as an order of magnitude and no more. It assumes the square-root law
    holds far outside the range it was fitted on, that liquidity does not react
    to the strategy's own presence, that trading cannot be spread over more days,
    and that nobody else is trading the same signal. Every one of those
    assumptions pushes the true number DOWN.
    """
    base_notional = cost_cfg.portfolio_notional
    periods = metrics_cfg.trading_days_per_year

    mean_gross = float(gross.mean()) * periods
    base_impact = float(impact_cost.mean()) * periods

    if base_impact <= 0 or mean_gross <= 0:
        capacity = float("nan")
    else:
        target = metrics_cfg.capacity_erosion_fraction * mean_gross
        capacity = base_notional * (target / base_impact) ** 2

    traded = participation.to_numpy()
    non_zero = traded[np.isfinite(traded) & (traded > 0)]

    breaches = (
        float((non_zero > cost_cfg.max_participation_rate).mean()) if non_zero.size else np.nan
    )
    worst = float(non_zero.max()) if non_zero.size else np.nan

    # The capacity figure and the participation tail are two views of the same
    # question, and when they disagree the tail is the one to believe: it is a
    # direct observation, whereas the capacity number extrapolates a fitted law
    # far outside its range. This flag exists so the table criticises itself
    # rather than relying on the reader to notice.
    scaling = capacity / base_notional if np.isfinite(capacity) and base_notional else np.nan
    credible = bool(
        np.isfinite(scaling)
        and scaling < 10.0
        and np.isfinite(worst)
        and worst <= cost_cfg.max_participation_rate
    )

    return pd.DataFrame(
        {
            "assumed_notional": [base_notional],
            "annualised_gross_return": [mean_gross],
            "annualised_impact_cost": [base_impact],
            "impact_share_of_gross": [base_impact / mean_gross if mean_gross else np.nan],
            "erosion_fraction": [metrics_cfg.capacity_erosion_fraction],
            "estimated_capacity_notional": [capacity],
            "implied_scaling_multiple": [scaling],
            "mean_participation_rate": [float(non_zero.mean()) if non_zero.size else np.nan],
            "p99_participation_rate": [
                float(np.quantile(non_zero, 0.99)) if non_zero.size else np.nan
            ],
            "max_participation_rate": [worst],
            "share_of_trades_over_participation_cap": [breaches],
            "capacity_estimate_is_credible": [credible],
        }
    )

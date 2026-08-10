"""Cost conservation: gross minus itemised costs must equal net, exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from honest_backtest.config import Config, RealismConfig
from honest_backtest.core.costs import CostBreakdown
from honest_backtest.core.engine import run
from honest_backtest.core.signal import Signal
from honest_backtest.data.market import MarketData
from honest_backtest.metrics import attribution
from honest_backtest.metrics.attribution import AttributionError
from honest_backtest.study import momentum_builder
from tests.conftest import business_days


def test_engine_output_conserves_exactly(synthetic_market: MarketData, cfg: Config) -> None:
    signal = Signal.from_builder(momentum_builder(), synthetic_market.closes)
    result = run(synthetic_market, signal, cfg)

    residual = (result.gross_return - result.costs.total) - result.net_return
    assert float(residual.abs().max()) == 0.0


def test_conservation_holds_with_every_protection_off(
    synthetic_market: MarketData, cfg: Config
) -> None:
    signal = Signal.from_builder(momentum_builder(), synthetic_market.closes)
    naive = run(synthetic_market, signal, cfg.with_realism(RealismConfig.naive()))

    residual = (naive.gross_return - naive.costs.total) - naive.net_return
    assert float(residual.abs().max()) == 0.0
    # With costs off, gross and net must be the same series.
    pd.testing.assert_series_equal(naive.gross_return, naive.net_return, check_names=False)


def test_components_sum_to_the_total() -> None:
    index = business_days(5)
    rng = np.random.default_rng(3)
    breakdown = CostBreakdown(
        commission=pd.Series(rng.uniform(0, 1e-4, 5), index=index),
        spread=pd.Series(rng.uniform(0, 1e-4, 5), index=index),
        impact=pd.Series(rng.uniform(0, 1e-4, 5), index=index),
        borrow=pd.Series(rng.uniform(0, 1e-4, 5), index=index),
    )
    manual = breakdown.commission + breakdown.spread + breakdown.impact + breakdown.borrow
    pd.testing.assert_series_equal(breakdown.total, manual, check_names=False)


def test_attribution_rejects_a_broken_identity() -> None:
    """If a cost is ever added without being wired into the total, this must fire."""
    index = business_days(4)
    gross = pd.Series(0.001, index=index)
    breakdown = CostBreakdown(
        commission=pd.Series(0.0001, index=index),
        spread=pd.Series(0.0, index=index),
        impact=pd.Series(0.0, index=index),
        borrow=pd.Series(0.0, index=index),
    )
    wrong_net = gross  # forgot to subtract anything

    with pytest.raises(AttributionError, match="does not conserve"):
        attribution.attribute(gross, wrong_net, breakdown, periods_per_year=252)


def test_attribution_rows_reconstruct_the_net_return(
    synthetic_market: MarketData, cfg: Config
) -> None:
    """The published decomposition must add up, not merely look plausible."""
    signal = Signal.from_builder(momentum_builder(), synthetic_market.closes)
    result = run(synthetic_market, signal, cfg)
    table = attribution.attribute(
        result.gross_return, result.net_return, result.costs, periods_per_year=252
    )

    rebuilt = (
        table.loc["gross_return", "annualised_contribution"]
        + table.loc["commission", "annualised_contribution"]
        + table.loc["spread", "annualised_contribution"]
        + table.loc["market_impact", "annualised_contribution"]
        + table.loc["borrow", "annualised_contribution"]
    )
    assert rebuilt == pytest.approx(table.loc["net_return", "annualised_contribution"], abs=1e-12)


def test_costs_are_never_negative(synthetic_market: MarketData, cfg: Config) -> None:
    """No cost component may pay the strategy."""
    signal = Signal.from_builder(momentum_builder(), synthetic_market.closes)
    result = run(synthetic_market, signal, cfg)
    for name, component in result.costs.to_frame().items():
        assert float(component.min()) >= 0.0, f"{name} went negative"


def test_honest_never_beats_naive_on_cost(synthetic_market: MarketData, cfg: Config) -> None:
    """Charging for things cannot make the strategy cheaper."""
    signal = Signal.from_builder(momentum_builder(), synthetic_market.closes)
    honest = run(synthetic_market, signal, cfg)
    naive = run(synthetic_market, signal, cfg.with_realism(RealismConfig.naive()))
    assert float(honest.costs.total.sum()) > float(naive.costs.total.sum())
    assert float(naive.costs.total.sum()) == 0.0

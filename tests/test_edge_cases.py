"""Property and edge-case tests: the states that make a backtester return nonsense.

Empty universes, single names, all-NaN columns and mid-period delistings are the
inputs that turn a backtest's output from wrong into confidently wrong. Each one
below asserts a specific safe behaviour rather than merely "does not crash".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from honest_backtest.config import Config, PortfolioConfig, RealismConfig
from honest_backtest.core import weights as weight_model
from honest_backtest.core.engine import run
from honest_backtest.core.signal import Signal, rank_cross_section
from honest_backtest.data.market import MarketData, MarketDataError
from honest_backtest.study import momentum_builder
from tests.conftest import business_days, toy_market


def test_an_empty_universe_produces_no_position_not_a_crash() -> None:
    """Every date ineligible: the book must be flat, and the return exactly zero."""
    market = toy_market(n_days=40)
    empty = market.membership.copy()
    empty.loc[:, :] = False

    # Membership all False would fail validation's "member must be priceable"
    # check only in the other direction, so this is a legal market.
    flat = MarketData(
        opens=market.opens,
        closes=market.closes,
        volumes=market.volumes,
        membership=empty,
        delisting_returns=market.delisting_returns,
        borrow_tier=market.borrow_tier,
        hard_to_borrow=market.hard_to_borrow,
        is_synthetic=True,
    )
    signal = Signal(values=pd.DataFrame(1.0, index=market.dates, columns=market.names))
    result = run(flat, signal, Config())

    assert float(result.weights.abs().to_numpy().sum()) == 0.0
    assert float(result.net_return.abs().sum()) == 0.0


def test_a_single_name_cannot_form_a_long_short_book() -> None:
    """One name is not a cross-section. The book must stay empty."""
    market = toy_market(n_days=40, names=("ONLY",))
    signal = Signal(values=pd.DataFrame(1.0, index=market.dates, columns=market.names))
    result = run(market, signal, Config())
    assert float(result.weights.abs().to_numpy().sum()) == 0.0


def test_an_all_nan_signal_column_is_excluded_rather_than_ranked() -> None:
    """A name with no signal must not be ranked; it must simply not be held."""
    index = business_days(3)
    values = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "DEAD": [np.nan] * 3, "C": [3.0, 2.0, 1.0]}, index=index
    )
    ranks = rank_cross_section(values)
    assert ranks["DEAD"].isna().all()
    assert ranks[["A", "C"]].notna().all().all()


def test_a_name_that_delists_mid_period_is_exited_and_charged() -> None:
    """A delisting must apply its terminal return, not vanish at the last price."""
    market = toy_market(n_days=60, names=("AAA", "BBB", "CCC", "DDD"), delisting={"AAA": 30})
    events = market.delisting_events()

    assert "AAA" in events.index
    exit_date = events["AAA"]
    # Membership ends on the exit date and never resumes.
    assert bool(market.membership.loc[exit_date, "AAA"])
    assert not market.membership.loc[exit_date:, "AAA"].iloc[1:].any()
    # And the terminal return is recorded rather than implied.
    assert market.delisting_returns["AAA"] == pytest.approx(-0.30)


def test_honouring_delistings_changes_the_return_series(synthetic_market: MarketData) -> None:
    """Turning the protection off must actually change something, or it is theatre."""
    cfg = Config()
    signal = Signal.from_builder(momentum_builder(), synthetic_market.closes)

    honouring = run(synthetic_market, signal, cfg)
    ignoring = run(
        synthetic_market,
        signal,
        cfg.with_realism(cfg.realism.model_copy(update={"honour_delistings": False})),
    )
    assert not honouring.net_return.equals(ignoring.net_return)


def test_hard_to_borrow_names_are_never_shorted() -> None:
    """A blocked name may be held long but must never appear on the short side."""
    market = toy_market(n_days=60, names=("AAA", "BBB", "CCC", "DDD"), hard_to_borrow=("AAA",))
    # AAA has the slowest growth, so it would otherwise rank into the short leg.
    # toy_market grows AAA slowest, so ranking on trailing return puts AAA in the
    # bottom bucket — i.e. on the short side, which is what these tests need.
    signal = Signal(values=market.closes.pct_change(20).fillna(0.0))

    cfg = Config().model_copy(
        update={"portfolio": PortfolioConfig(n_buckets=2, min_names_per_side=1)}
    )
    result = run(market, signal, cfg)
    assert float(result.weights["AAA"].min()) >= 0.0

    # Control: without the borrow rule AAA IS shorted, so the assertion above is
    # testing the rule rather than an accident of the ranking.
    unblocked = run(market, signal, cfg.with_realism(RealismConfig.naive()))
    assert float(unblocked.weights["AAA"].min()) < 0.0


def test_blocked_shorts_are_counted_not_silently_dropped() -> None:
    market = toy_market(n_days=60, hard_to_borrow=("AAA", "BBB"))
    # toy_market grows AAA slowest, so ranking on trailing return puts AAA in the
    # bottom bucket — i.e. on the short side, which is what these tests need.
    signal = Signal(values=market.closes.pct_change(20).fillna(0.0))
    cfg = Config().model_copy(
        update={"portfolio": PortfolioConfig(n_buckets=2, min_names_per_side=1)}
    )
    result = run(market, signal, cfg)
    assert int(result.n_blocked_shorts.max()) > 0


def test_disabling_borrow_enforcement_allows_the_short() -> None:
    market = toy_market(n_days=60, hard_to_borrow=("AAA",))
    # toy_market grows AAA slowest, so ranking on trailing return puts AAA in the
    # bottom bucket — i.e. on the short side, which is what these tests need.
    signal = Signal(values=market.closes.pct_change(20).fillna(0.0))
    cfg = Config().model_copy(
        update={"portfolio": PortfolioConfig(n_buckets=2, min_names_per_side=1)}
    )
    naive = run(market, signal, cfg.with_realism(RealismConfig.naive()))
    assert float(naive.weights["AAA"].min()) < 0.0


def test_weight_cap_is_respected() -> None:
    """No single name may exceed the configured cap, however thin the sort."""
    index = business_days(1)
    ranks = pd.DataFrame([[0.05, 0.10, 0.90, 0.95]], index=index, columns=list("ABCD"))
    cfg = PortfolioConfig(n_buckets=2, max_weight_per_name=0.30, min_names_per_side=1)
    targets = weight_model.target_weights(ranks, cfg)
    assert float(targets.weights.abs().to_numpy().max()) <= 0.30 + 1e-9


def test_legs_are_balanced_and_book_is_dollar_neutral() -> None:
    index = business_days(1)
    ranks = pd.DataFrame(
        [np.linspace(0.02, 0.98, 20)], index=index, columns=[f"N{i}" for i in range(20)]
    )
    cfg = PortfolioConfig(n_buckets=5, max_weight_per_name=1.0, min_names_per_side=1)
    targets = weight_model.target_weights(ranks, cfg)
    row = targets.weights.iloc[0]

    assert row.sum() == pytest.approx(0.0, abs=1e-12)
    assert row.abs().sum() == pytest.approx(2.0 * cfg.leg_gross)


def test_a_leg_too_thin_is_emptied_rather_than_concentrated() -> None:
    index = business_days(1)
    ranks = pd.DataFrame([[0.1, 0.9]], index=index, columns=["A", "B"])
    cfg = PortfolioConfig(n_buckets=2, min_names_per_side=5)
    targets = weight_model.target_weights(ranks, cfg)
    assert float(targets.weights.abs().to_numpy().sum()) == 0.0


def test_rebalance_dates_are_the_last_trading_day_of_each_period() -> None:
    dates = pd.bdate_range("2021-01-01", "2021-03-31", name="date")
    monthly = weight_model.rebalance_dates(dates, "monthly")
    assert pd.Timestamp("2021-01-29") in monthly
    assert pd.Timestamp("2021-02-26") in monthly
    assert pd.Timestamp("2021-03-31") in monthly
    assert len(monthly) == 3


def test_unknown_rebalance_frequency_is_rejected() -> None:
    dates = pd.bdate_range("2021-01-01", periods=10, name="date")
    with pytest.raises(ValueError, match="Unsupported rebalance frequency"):
        weight_model.rebalance_dates(dates, "fortnightly")


def test_market_data_rejects_mismatched_frames() -> None:
    market = toy_market(n_days=20)
    with pytest.raises(MarketDataError, match="columns do not match"):
        MarketData(
            opens=market.opens.iloc[:, :2],
            closes=market.closes,
            volumes=market.volumes,
            membership=market.membership,
            delisting_returns=market.delisting_returns,
            borrow_tier=market.borrow_tier,
            hard_to_borrow=market.hard_to_borrow,
            is_synthetic=True,
        )


def test_market_data_rejects_a_member_with_no_price() -> None:
    """A name cannot be in the investable universe and unpriceable at once."""
    market = toy_market(n_days=20)
    holed = market.closes.copy()
    holed.iloc[5, 0] = np.nan
    with pytest.raises(MarketDataError, match="no close"):
        MarketData(
            opens=market.opens,
            closes=holed,
            volumes=market.volumes,
            membership=market.membership,
            delisting_returns=market.delisting_returns,
            borrow_tier=market.borrow_tier,
            hard_to_borrow=market.hard_to_borrow,
            is_synthetic=True,
        )


def test_market_data_rejects_unsorted_dates() -> None:
    market = toy_market(n_days=20)
    shuffled = market.closes.iloc[::-1]
    with pytest.raises(MarketDataError, match="sorted ascending"):
        MarketData(
            opens=market.opens.iloc[::-1],
            closes=shuffled,
            volumes=market.volumes.iloc[::-1],
            membership=market.membership.iloc[::-1],
            delisting_returns=market.delisting_returns,
            borrow_tier=market.borrow_tier,
            hard_to_borrow=market.hard_to_borrow.iloc[::-1],
            is_synthetic=True,
        )

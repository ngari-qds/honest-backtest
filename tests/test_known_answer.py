"""The known-answer test: a toy portfolio whose numbers are computed by hand here.

These are the tests that make the engine's arithmetic checkable by a reader who
does not trust the engine. Every expected value below is derived in the comment
above it from first principles, not copied from a previous run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from honest_backtest.config import BorrowConfig, CostConfig
from honest_backtest.core import costs as cost_model
from tests.conftest import business_days


def test_commission_on_a_hand_computed_trade() -> None:
    """Trade 40% of the book at 1bp per side.

    Weight changes: +0.20 on one name, -0.20 on another. Absolute weight
    traded = 0.40 of book. At 1bp, commission = 0.40 * 0.0001 = 0.00004,
    i.e. 0.4 basis points of the book.
    """
    index = business_days(1)
    change = pd.DataFrame([[0.20, -0.20]], index=index, columns=["A", "B"])
    prices = pd.DataFrame([[100.0, 100.0]], index=index, columns=["A", "B"])
    adv = pd.DataFrame([[1e9, 1e9]], index=index, columns=["A", "B"])

    cfg = CostConfig(commission_bps=1.0, half_spread_bps=0.0, impact_coefficient=0.0)
    commission, spread, impact, _ = cost_model.trading_costs(change, prices, adv, cfg)

    assert commission.iloc[0] == pytest.approx(0.00004)
    assert commission.iloc[0] * 1e4 == pytest.approx(0.4)
    assert spread.iloc[0] == 0.0
    assert impact.iloc[0] == 0.0


def test_spread_is_charged_once_per_side() -> None:
    """A 4bp half-spread on 0.40 of book traded costs 0.40 * 0.0004 = 16bp/100."""
    index = business_days(1)
    change = pd.DataFrame([[0.20, -0.20]], index=index, columns=["A", "B"])
    prices = pd.DataFrame([[50.0, 50.0]], index=index, columns=["A", "B"])
    adv = pd.DataFrame([[1e9, 1e9]], index=index, columns=["A", "B"])

    cfg = CostConfig(commission_bps=0.0, half_spread_bps=4.0, impact_coefficient=0.0)
    _, spread, _, _ = cost_model.trading_costs(change, prices, adv, cfg)
    assert spread.iloc[0] == pytest.approx(0.40 * 0.0004)
    assert spread.iloc[0] * 1e4 == pytest.approx(1.6)


def test_participation_rate_is_shares_traded_over_adv() -> None:
    """$100m book, 10% weight change, $50 price -> 200,000 shares. ADV 2m -> 10%."""
    index = business_days(1)
    change = pd.DataFrame([[0.10]], index=index, columns=["A"])
    prices = pd.DataFrame([[50.0]], index=index, columns=["A"])
    adv = pd.DataFrame([[2_000_000.0]], index=index, columns=["A"])

    shares = cost_model.traded_shares(change, prices, portfolio_notional=100_000_000.0)
    assert shares.iloc[0, 0] == pytest.approx(200_000.0)

    participation = cost_model.participation_rate(shares, adv)
    assert participation.iloc[0, 0] == pytest.approx(0.10)


def test_impact_follows_the_square_root_law() -> None:
    """coefficient 10 at 1% participation is 10*0.1 = 1bp; at 4% it is 2bp.

    Quadrupling participation must exactly double the rate, which is the whole
    content of the square-root law.
    """
    index = business_days(1)
    participation = pd.DataFrame([[0.01, 0.04]], index=index, columns=["A", "B"])
    rate = cost_model.market_impact_bps(participation, coefficient=10.0)

    assert rate.iloc[0, 0] == pytest.approx(1.0)
    assert rate.iloc[0, 1] == pytest.approx(2.0)
    assert rate.iloc[0, 1] == pytest.approx(2.0 * rate.iloc[0, 0])


def test_borrow_is_charged_only_on_the_short_leg_and_accrues_daily() -> None:
    """A 1.0 short at 30bp/yr over 252 days costs 30bp/252 per day.

    The long leg must contribute nothing: you are not paying to borrow something
    you own.
    """
    index = business_days(3)
    weights = pd.DataFrame(
        [[1.0, -1.0]] * 3,
        index=index,
        columns=["LONG", "SHORT"],
    )
    tiers = pd.Series(
        {"LONG": "general_collateral", "SHORT": "general_collateral"}, name="borrow_tier"
    )
    cfg = BorrowConfig(tier_annual_bps={"general_collateral": 30.0}, trading_days_per_year=252)

    charge = cost_model.borrow_costs(weights, tiers, cfg)
    expected_daily = (30.0 / 1e4) / 252.0
    assert charge.iloc[0] == pytest.approx(expected_daily)
    assert charge.to_numpy() == pytest.approx(expected_daily)

    # Same book with no short leg pays nothing at all.
    long_only = pd.DataFrame([[1.0, 0.0]] * 3, index=index, columns=["LONG", "SHORT"])
    assert (cost_model.borrow_costs(long_only, tiers, cfg) == 0.0).all()


def test_borrow_tier_lookup_uses_the_right_rate() -> None:
    """A 'hot' name at 800bp/yr costs the hot rate, not the default."""
    names = pd.Index(["GC", "HOT", "UNKNOWN_TIER"])
    tiers = pd.Series({"GC": "general_collateral", "HOT": "hot"})
    cfg = BorrowConfig(
        default_annual_bps=40.0,
        tier_annual_bps={"general_collateral": 30.0, "hot": 800.0},
    )
    rates = cost_model.borrow_rates(names, tiers, cfg)

    assert rates["GC"] == pytest.approx(30.0)
    assert rates["HOT"] == pytest.approx(800.0)
    # A name with no tier falls back to the default rather than to zero.
    assert rates["UNKNOWN_TIER"] == pytest.approx(40.0)


def test_unmeasurable_liquidity_is_not_charged_as_free() -> None:
    """A name with no ADV history must not get a zero impact charge."""
    index = business_days(1)
    change = pd.DataFrame([[0.10]], index=index, columns=["A"])
    prices = pd.DataFrame([[50.0]], index=index, columns=["A"])
    adv = pd.DataFrame([[np.nan]], index=index, columns=["A"])

    cfg = CostConfig(commission_bps=0.0, half_spread_bps=0.0, impact_coefficient=10.0)
    _, _, impact, _ = cost_model.trading_costs(change, prices, adv, cfg)

    # Charged at the configured participation cap: 10 * sqrt(0.10) = 3.16bp.
    expected_rate = 10.0 * np.sqrt(cfg.max_participation_rate)
    assert impact.iloc[0] == pytest.approx(0.10 * expected_rate / 1e4)
    assert impact.iloc[0] > 0.0


def test_zero_trading_costs_nothing() -> None:
    index = business_days(2)
    change = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    prices = pd.DataFrame(100.0, index=index, columns=["A", "B"])
    adv = pd.DataFrame(1e6, index=index, columns=["A", "B"])

    commission, spread, impact, _ = cost_model.trading_costs(change, prices, adv, CostConfig())
    assert (commission == 0.0).all()
    assert (spread == 0.0).all()
    assert (impact == 0.0).all()

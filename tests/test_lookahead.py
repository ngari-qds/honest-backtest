"""Look-ahead. The engine must make it hard to cheat and loud when you do."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from honest_backtest.config import Config
from honest_backtest.core.engine import run
from honest_backtest.core.signal import LookaheadError, Signal, assert_causal
from honest_backtest.data.market import MarketData
from honest_backtest.study import momentum_builder


def _causal_builder(closes: pd.DataFrame) -> pd.DataFrame:
    return closes.shift(21) / closes.shift(252) - 1.0


#: Horizon of the oracle signal, in trading days. Long enough to survive the
#: reporting lag and the next-day execution delay and still cover the holding
#: period — a one-day peek is destroyed by the engine's own delays, which is
#: itself a good sign, but makes for a useless control.
PEEK_HORIZON_DAYS = 30


def _peeking_builder(closes: pd.DataFrame) -> pd.DataFrame:
    """Deliberately reads future prices. This is the thing being detected."""
    return closes.shift(-PEEK_HORIZON_DAYS) / closes - 1.0


def _subtly_peeking_builder(closes: pd.DataFrame) -> pd.DataFrame:
    """Centres on a FULL-SAMPLE mean, so every past value depends on the future.

    This is the realistic mistake: nobody writes shift(-1) by accident, but
    standardising against a statistic computed over the whole sample is a
    mistake people make constantly, and it contaminates every date.
    """
    momentum = closes.shift(21) / closes.shift(252) - 1.0
    return momentum - momentum.mean().mean()


def test_a_causal_builder_passes(synthetic_market: MarketData) -> None:
    assert_causal(_causal_builder, synthetic_market.closes, seed=1)


def test_an_obvious_peek_is_caught(synthetic_market: MarketData) -> None:
    with pytest.raises(LookaheadError, match="reading the future"):
        assert_causal(_peeking_builder, synthetic_market.closes, seed=1)


def test_a_full_sample_statistic_is_caught(synthetic_market: MarketData) -> None:
    """The subtle, common mistake must be caught too, not just the blatant one."""
    with pytest.raises(LookaheadError):
        assert_causal(_subtly_peeking_builder, synthetic_market.closes, seed=1)


def test_the_error_names_the_cutoff_and_the_damage(synthetic_market: MarketData) -> None:
    with pytest.raises(LookaheadError) as info:
        assert_causal(_peeking_builder, synthetic_market.closes, seed=1)
    message = str(info.value)
    assert "changed" in message
    assert "earliest affected date" in message


def test_from_builder_refuses_to_construct_a_peeking_signal(
    synthetic_market: MarketData,
) -> None:
    """The checked path is the easy path, and it fails closed."""
    with pytest.raises(LookaheadError):
        Signal.from_builder(_peeking_builder, synthetic_market.closes, check_causality=True)


def test_from_builder_records_whether_it_checked(synthetic_market: MarketData) -> None:
    checked = Signal.from_builder(_causal_builder, synthetic_market.closes, check_causality=True)
    unchecked = Signal.from_builder(_causal_builder, synthetic_market.closes, check_causality=False)
    assert checked.was_causality_checked
    assert not unchecked.was_causality_checked
    # And the flag reaches the result, so a run manifest can report it.
    assert not Signal(values=checked.values).was_causality_checked


def test_reporting_lag_shifts_availability_forward() -> None:
    """A value referring to date t must first be usable at t + lag."""
    index = pd.bdate_range("2020-01-01", periods=5, name="date")
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    signal = Signal(values=values)

    same_day = signal.available_at(reporting_lag_days=0)
    lagged = signal.available_at(reporting_lag_days=2)

    assert same_day.iloc[2, 0] == 3.0
    assert np.isnan(lagged.iloc[0, 0])
    assert np.isnan(lagged.iloc[1, 0])
    assert lagged.iloc[2, 0] == 1.0  # the value from two business days earlier


def test_negative_reporting_lag_is_rejected() -> None:
    index = pd.bdate_range("2020-01-01", periods=3, name="date")
    signal = Signal(values=pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index))
    with pytest.raises(ValueError, match="must not be negative"):
        signal.available_at(reporting_lag_days=-1)


def test_a_signal_shaped_wrongly_is_rejected(synthetic_market: MarketData) -> None:
    def wrong_shape(closes: pd.DataFrame) -> pd.DataFrame:
        return closes.iloc[:, :2]

    with pytest.raises(ValueError, match="same dates and names"):
        Signal.from_builder(wrong_shape, synthetic_market.closes)


def test_execution_happens_strictly_after_the_signal_date(
    synthetic_market: MarketData, cfg: Config
) -> None:
    """No weight may be established on the date the signal that chose it is dated.

    The engine strikes targets on rebalance dates and executes on the following
    trading day. If any weight appeared on the rebalance date itself, the book
    would be trading on information it had only just received.
    """
    signal = Signal.from_builder(momentum_builder(), synthetic_market.closes)
    result = run(synthetic_market, signal, cfg)

    traded = result.traded_notional
    first_trade = traded[traded > 0].index.min()
    first_signal = signal.values.dropna(how="all").index.min()
    assert first_trade > first_signal


def test_a_strategy_that_peeks_earns_far_more_than_one_that_does_not(
    synthetic_market: MarketData, cfg: Config
) -> None:
    """The control: the harness must be able to tell the two apart.

    Without this, every assertion above could pass on an engine too blunt to
    notice look-ahead at all.
    """
    honest = Signal.from_builder(momentum_builder(), synthetic_market.closes)
    peeking = Signal(values=_peeking_builder(synthetic_market.closes))

    honest_result = run(synthetic_market, honest, cfg)
    peeking_result = run(synthetic_market, peeking, cfg)

    def sharpe(series: pd.Series) -> float:
        clean = series.dropna()
        return float(clean.mean() / clean.std() * np.sqrt(252))

    assert sharpe(peeking_result.gross_return) > sharpe(honest_result.gross_return) + 2.0

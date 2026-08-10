"""Engine behaviour, realism switches, metrics, and the performance budget."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from honest_backtest.config import Config, RealismConfig, SyntheticDataConfig
from honest_backtest.core.engine import run
from honest_backtest.core.signal import Signal
from honest_backtest.core.weights import drift_within_periods
from honest_backtest.data.market import MarketData
from honest_backtest.data.synthetic import (
    PLANTED_LOOKBACK_DAYS,
    PLANTED_SKIP_DAYS,
    generate_market,
)
from honest_backtest.metrics import performance
from honest_backtest.study import (
    MOMENTUM_LOOKBACK_DAYS,
    MOMENTUM_SKIP_DAYS,
    build_signal,
    momentum_builder,
    run_comparison,
)
from tests.conftest import business_days, toy_market


def test_planted_effect_matches_the_signal_the_demo_trades() -> None:
    """The generator and the demo signal must use the same formation window.

    If they drift apart the demo trades one thing while the market rewards
    another, and the naive-versus-honest comparison stops measuring anything.
    This caught a real mismatch during development.
    """
    assert PLANTED_LOOKBACK_DAYS == MOMENTUM_LOOKBACK_DAYS
    assert PLANTED_SKIP_DAYS == MOMENTUM_SKIP_DAYS


def test_the_planted_effect_is_actually_recoverable(strong_effect_market: MarketData) -> None:
    """The generator must plant an effect the demo's signal can actually find.

    Uses an exaggerated planted strength, so a failure means the generator or
    the signal is broken rather than that the sample was too small to resolve a
    three-percent-a-year effect.
    """
    signal = build_signal(strong_effect_market, Config())
    forward = strong_effect_market.closes.pct_change().shift(-1)
    frame = pd.concat(
        [signal.values.stack().rename("signal"), forward.stack().rename("forward")], axis=1
    ).dropna()
    assert frame["signal"].corr(frame["forward"]) > 0.0


def test_naive_beats_honest(demo_synthetic_market: MarketData, cfg: Config) -> None:
    """The whole premise: switching protections off must flatter the result.

    Run against the demo's own configuration, because that is the claim the
    README makes. On a 40-name fixture the difference sits inside the noise.
    """
    signal = build_signal(demo_synthetic_market, cfg)
    comparison = run_comparison(demo_synthetic_market, signal, cfg)
    assert comparison.summary.loc["sharpe", "naive"] > comparison.summary.loc["sharpe", "honest"]
    assert (
        comparison.summary.loc["ann_return", "naive"]
        > comparison.summary.loc["ann_return", "honest"]
    )


def test_the_ladder_starts_naive_and_ends_honest(
    demo_synthetic_market: MarketData, cfg: Config
) -> None:
    synthetic_market = demo_synthetic_market
    signal = build_signal(synthetic_market, cfg)
    comparison = run_comparison(synthetic_market, signal, cfg)
    ladder = comparison.ladder

    assert ladder.iloc[0]["enabled"] == "none"
    assert ladder.iloc[0]["sharpe"] == pytest.approx(
        comparison.summary.loc["sharpe", "naive"], abs=1e-9
    )
    assert ladder.iloc[-1]["sharpe"] == pytest.approx(
        comparison.summary.loc["sharpe", "honest"], abs=1e-9
    )


def test_realism_naive_disables_everything() -> None:
    naive = RealismConfig.naive()
    assert naive.is_naive
    assert len(naive.disabled_protections) == 6
    assert not RealismConfig().is_naive
    assert RealismConfig().disabled_protections == ()


def test_result_carries_its_own_caveat(synthetic_market: MarketData, cfg: Config) -> None:
    """A number from this engine cannot be quoted without its assumptions."""
    signal = build_signal(synthetic_market, cfg)
    honest = run(synthetic_market, signal, cfg)
    naive = run(synthetic_market, signal, cfg.with_realism(RealismConfig.naive()))

    assert "SYNTHETIC DATA" in honest.caveat()
    assert "protections disabled" in naive.caveat()
    assert naive.is_naive
    assert not honest.is_naive


def test_survivorship_membership_only_contains_final_members() -> None:
    market = toy_market(n_days=60, delisting={"AAA": 30})
    naive_membership = market.survivors_only_membership()
    # AAA left the universe, so a survivors-only view never includes it...
    assert not naive_membership["AAA"].any()
    # ...but does include it for names that stayed, across their whole history.
    assert naive_membership["BBB"].all()


def test_trade_timing_changes_the_result(synthetic_market: MarketData, cfg: Config) -> None:
    """Filling at the open and filling at the close must not be the same thing."""
    signal = build_signal(synthetic_market, cfg)
    at_open = run(synthetic_market, signal, cfg)
    at_close = run(
        synthetic_market,
        signal,
        cfg.model_copy(
            update={"execution": cfg.execution.model_copy(update={"trade_on": "next_close"})}
        ),
    )
    assert not at_open.net_return.equals(at_close.net_return)


def test_drift_preserves_leg_gross_exposure() -> None:
    index = business_days(4)
    names = ["A", "B"]
    targets = pd.DataFrame([[0.5, -0.5]] * 4, index=index, columns=names)
    returns = pd.DataFrame([[0.10, 0.0]] * 4, index=index, columns=names)
    period = pd.Series(0.0, index=index)

    drifted = drift_within_periods(targets, returns, period)
    row = drifted.iloc[-1]
    assert row[row > 0].sum() == pytest.approx(0.5)
    assert row[row < 0].sum() == pytest.approx(-0.5)


def test_annualised_return_is_geometric() -> None:
    index = business_days(252)
    series = pd.Series(0.001, index=index)
    assert performance.annualised_return(series, periods_per_year=252) == pytest.approx(
        1.001**252 - 1.0
    )


def test_max_drawdown_of_a_known_path() -> None:
    index = business_days(2)
    series = pd.Series([0.5, -0.5], index=index)
    assert performance.max_drawdown(series) == pytest.approx(-0.5)


def test_drawdown_table_reports_an_open_episode_as_unrecovered() -> None:
    index = business_days(3)
    series = pd.Series([0.2, -0.5, -0.1], index=index)
    table = performance.drawdown_table(series, threshold=0.05)
    assert len(table) == 1
    assert not bool(table.loc[0, "recovered"])
    assert pd.isna(table.loc[0, "recovery"])


def test_sharpe_of_a_constant_series_is_undefined() -> None:
    index = business_days(50)
    assert np.isnan(performance.sharpe_ratio(pd.Series(0.001, index=index), periods_per_year=252))


def test_full_study_runs_in_under_a_minute() -> None:
    """The stated performance budget: a few hundred names over 15 years, under 60s.

    This is a claim in the README, so it gets an assertion rather than a note.
    Both configurations are run, plus the six-rung ladder — eight backtests in
    total — so passing this comfortably covers the single-run claim.
    """
    cfg = Config()
    market = generate_market(SyntheticDataConfig(n_names=400, n_days=3_780), seed=cfg.seed)
    signal = Signal.from_builder(momentum_builder(), market.closes, check_causality=False)

    started = time.perf_counter()
    run(market, signal, cfg)
    elapsed = time.perf_counter() - started

    assert market.closes.shape == (3_780, 400)
    assert elapsed < 60.0, f"one backtest of 400 names over 15 years took {elapsed:.1f}s"

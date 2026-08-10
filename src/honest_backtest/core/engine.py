"""The backtest engine.

WHAT THE CALLER DOES NOT CONTROL
--------------------------------
The caller supplies a ``Signal`` and a ``MarketData``. It does not supply
returns, does not choose an alignment, and cannot ask for the signal to be
applied on the day it was computed. The engine derives returns from prices,
applies the reporting lag, and delays execution to the next open or close. Those
are the three places a backtest usually cheats, and none of them is an argument.

HOW A DAY IS ACCOUNTED FOR
--------------------------
On an ordinary day the book earns ``weights . close_to_close_return``.

On a rebalance day with ``trade_on="next_open"``, the day is split in two, and
the split is the reason this engine bothers with open prices at all:

    return = old_weights . (open/prev_close - 1)  +  new_weights . (close/open - 1)

You held yesterday's book overnight; you hold today's book from the opening
print. Charging the whole day to the new weights would credit the strategy with
an overnight move it had not yet positioned for — a small effect on any one day
and a systematic one over three thousand of them.

DELISTINGS
----------
A name that leaves the universe has its terminal return applied on its final
membership date, and the position is then gone. It is not dropped at its last
observed price. Deleting a position instead of resolving it is the most common
way a backtest silently removes its own losses, which is why
``RealismConfig.honour_delistings`` exists and why turning it off is visible in
the result label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from honest_backtest.config import Config
from honest_backtest.core import costs as cost_model
from honest_backtest.core import weights as weight_model
from honest_backtest.core.signal import Signal, rank_cross_section
from honest_backtest.data.market import MarketData

#: Trailing window for average daily volume, in trading days.
ADV_WINDOW_DAYS = 21


@dataclass(frozen=True)
class BacktestResult:
    """Everything one run produced, with its honesty settings attached.

    ``realism`` and ``is_synthetic`` travel with the result deliberately: a
    number from this engine cannot be quoted without the assumptions that
    produced it being available in the same object.
    """

    gross_return: pd.Series
    net_return: pd.Series
    costs: cost_model.CostBreakdown
    weights: pd.DataFrame
    traded_notional: pd.Series
    participation: pd.DataFrame
    n_long: pd.Series
    n_short: pd.Series
    n_blocked_shorts: pd.Series
    realism_disabled: tuple[str, ...]
    is_synthetic: bool
    signal_causality_checked: bool
    label: str

    @property
    def is_naive(self) -> bool:
        return len(self.realism_disabled) == 6

    @property
    def turnover(self) -> pd.Series:
        out = self.traded_notional / 2.0
        out.name = "turnover"
        return out

    def caveat(self) -> str:
        """One line naming everything that makes this result less than real."""
        parts: list[str] = []
        if self.is_synthetic:
            parts.append("SYNTHETIC DATA")
        if self.realism_disabled:
            parts.append("protections disabled: " + ", ".join(self.realism_disabled))
        if not self.signal_causality_checked:
            parts.append("signal not causality-checked")
        return "; ".join(parts) if parts else "fully realistic settings"


def run(
    market: MarketData,
    signal: Signal,
    cfg: Config,
    *,
    label: str = "run",
) -> BacktestResult:
    """Run one backtest under one set of realism settings."""
    realism = cfg.realism
    dates = market.dates

    # ---- what is tradeable, and when ---------------------------------
    membership = (
        market.membership
        if realism.respect_universe_membership
        else market.survivors_only_membership()
    )
    priced = market.closes.notna()
    eligible = membership & priced

    shortable = None
    if realism.enforce_hard_to_borrow:
        shortable = eligible & ~market.hard_to_borrow

    # ---- signal availability -----------------------------------------
    lag = cfg.signal.reporting_lag_days if realism.apply_reporting_lag else 0
    available = signal.available_at(reporting_lag_days=lag)

    # ---- targets on rebalance dates only -----------------------------
    rebalances = weight_model.rebalance_dates(dates, cfg.execution.rebalance)
    ranks = rank_cross_section(available.loc[rebalances], eligible=eligible.loc[rebalances])
    targets = weight_model.target_weights(
        ranks,
        cfg.portfolio,
        shortable=None if shortable is None else shortable.loc[rebalances],
        signal_values=available.loc[rebalances],
    )

    # ---- execution delay ---------------------------------------------
    # A target struck on a rebalance date is executed on the NEXT trading day.
    positions = dates.get_indexer(rebalances)
    trade_positions = positions + 1
    valid = trade_positions < len(dates)
    trade_dates = dates[trade_positions[valid]]

    effective = pd.DataFrame(np.nan, index=dates, columns=market.names)
    effective.loc[trade_dates] = targets.weights.loc[rebalances[valid]].to_numpy()

    period_id = pd.Series(np.nan, index=dates)
    period_id.loc[trade_dates] = np.arange(len(trade_dates))
    period_id = period_id.ffill()

    close_returns = market.close_to_close_returns()
    close_returns = _apply_delistings(close_returns, market, honour=realism.honour_delistings)

    held = weight_model.drift_within_periods(
        effective.ffill().fillna(0.0), close_returns, period_id
    )
    held = held.where(period_id.notna(), 0.0)

    # ---- returns ------------------------------------------------------
    pre_trade = held.shift(1).fillna(0.0)
    is_trade_day = pd.Series(dates.isin(trade_dates), index=dates)

    ordinary = (held * close_returns.fillna(0.0)).sum(axis=1)

    if cfg.execution.trade_on == "next_open":
        overnight = market.overnight_returns().fillna(0.0)
        intraday = market.intraday_returns().fillna(0.0)
        split_day = (pre_trade * overnight).sum(axis=1) + (held * intraday).sum(axis=1)
        gross = ordinary.where(~is_trade_day, split_day)
    else:
        # Filled at the close: the whole day belongs to the previous book.
        gross = (pre_trade * close_returns.fillna(0.0)).sum(axis=1).where(is_trade_day, ordinary)

    gross.name = "gross_return"

    # ---- costs --------------------------------------------------------
    weight_change = (held - pre_trade).where(is_trade_day, 0.0)
    traded_notional = weight_change.abs().sum(axis=1)
    traded_notional.name = "traded_notional"

    fill_prices = market.opens if cfg.execution.trade_on == "next_open" else market.closes
    adv = market.average_daily_volume(ADV_WINDOW_DAYS)

    if realism.apply_costs:
        commission, spread, impact, participation = cost_model.trading_costs(
            weight_change, fill_prices, adv, cfg.costs
        )
    else:
        zero = pd.Series(0.0, index=dates)
        commission, spread, impact = zero.copy(), zero.copy(), zero.copy()
        commission.name, spread.name, impact.name = "commission", "spread", "impact"
        participation = pd.DataFrame(0.0, index=dates, columns=market.names)

    if realism.apply_borrow_costs:
        borrow = cost_model.borrow_costs(held, market.borrow_tier, cfg.borrow)
    else:
        borrow = pd.Series(0.0, index=dates, name="borrow")

    breakdown = cost_model.CostBreakdown(
        commission=commission, spread=spread, impact=impact, borrow=borrow
    )
    net = gross - breakdown.total
    net.name = "net_return"

    return BacktestResult(
        gross_return=gross,
        net_return=net,
        costs=breakdown,
        weights=held,
        traded_notional=traded_notional,
        participation=participation,
        n_long=targets.n_long.reindex(dates).ffill().fillna(0).astype("int64"),
        n_short=targets.n_short.reindex(dates).ffill().fillna(0).astype("int64"),
        n_blocked_shorts=targets.n_blocked_shorts.reindex(dates).ffill().fillna(0).astype("int64"),
        realism_disabled=realism.disabled_protections,
        is_synthetic=market.is_synthetic,
        signal_causality_checked=signal.was_causality_checked,
        label=label,
    )


def _apply_delistings(
    returns: pd.DataFrame,
    market: MarketData,
    *,
    honour: bool,
) -> pd.DataFrame:
    """Write each delisting's terminal return onto its final membership date.

    With ``honour=False`` the position simply stops existing at its last observed
    price, which is what a careless backtest does and why the naive run looks
    better than it should.
    """
    if not honour:
        return returns

    out = returns.copy()
    events = market.delisting_events()
    for ticker, exit_date in events.items():
        if exit_date in out.index and ticker in out.columns:
            out.loc[exit_date, ticker] = market.delisting_returns[ticker]
    return out

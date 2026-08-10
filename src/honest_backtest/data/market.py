"""The market data container, and the validation it refuses to skip.

``MarketData`` is deliberately awkward to construct badly. Every frame must
share an index and a column set, membership must be explicit, and the
``is_synthetic`` flag is required rather than defaulted — it travels with the
data into every result, table and chart title, so a generated demo can never be
mistaken for a market result further downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class MarketDataError(ValueError):
    """Raised when the inputs cannot describe a coherent market."""


@dataclass(frozen=True)
class MarketData:
    """Daily prices, volumes, universe membership and borrow terms.

    Attributes
    ----------
    opens, closes
        Adjusted prices. ``closes`` drives returns; ``opens`` exists so a
        rebalance can be filled at the open of the day *after* the signal date.
    volumes
        Daily share volume, used for average daily volume and hence for the
        participation rate in the impact model.
    membership
        Boolean, True where a name is in the investable universe on that date.
        This is what makes entries and exits explicit rather than implied by a
        non-null price.
    delisting_returns
        Terminal return applied on a name's final membership date. A delisting
        is an event with a price attached, not an absence of data.
    borrow_tier
        Per-name borrow tier label, resolved to a fee by ``data.borrow``.
    hard_to_borrow
        Boolean, True where a name cannot be shorted on that date.
    is_synthetic
        True when the panel was generated rather than observed. Propagates into
        every downstream artefact.
    """

    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame
    membership: pd.DataFrame
    delisting_returns: pd.Series
    borrow_tier: pd.Series
    hard_to_borrow: pd.DataFrame
    is_synthetic: bool

    def __post_init__(self) -> None:
        self.validate()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.closes.index)

    @property
    def names(self) -> pd.Index:
        return self.closes.columns

    def validate(self) -> None:
        frames = {
            "opens": self.opens,
            "volumes": self.volumes,
            "membership": self.membership,
            "hard_to_borrow": self.hard_to_borrow,
        }
        for label, frame in frames.items():
            if not frame.index.equals(self.closes.index):
                raise MarketDataError(f"{label} index does not match closes")
            if not frame.columns.equals(self.closes.columns):
                raise MarketDataError(f"{label} columns do not match closes")

        if not isinstance(self.closes.index, pd.DatetimeIndex):
            raise MarketDataError("closes must be indexed by date")
        if not self.closes.index.is_monotonic_increasing:
            raise MarketDataError("dates must be sorted ascending")
        if self.closes.index.has_duplicates:
            raise MarketDataError("duplicate dates in the price index")

        if self.membership.to_numpy().dtype != np.bool_:
            raise MarketDataError("membership must be boolean")
        if self.hard_to_borrow.to_numpy().dtype != np.bool_:
            raise MarketDataError("hard_to_borrow must be boolean")

        missing_price = self.membership & self.closes.isna()
        if bool(missing_price.to_numpy().any()):
            n = int(missing_price.to_numpy().sum())
            raise MarketDataError(
                f"{n} (date, name) cells are flagged as universe members but have no close "
                "price. A name in the universe must be priceable; fix membership or the prices."
            )

        unknown = set(self.borrow_tier.index) - set(self.names)
        if unknown:
            raise MarketDataError(f"borrow_tier contains unknown names: {sorted(unknown)[:5]}")

    # -- derived views ----------------------------------------------------

    def close_to_close_returns(self) -> pd.DataFrame:
        """Simple daily returns from close to close, with no forward filling."""
        return self.closes.pct_change(fill_method=None)

    def overnight_returns(self) -> pd.DataFrame:
        """Close of ``d-1`` to open of ``d``. The part of a day traded at old weights."""
        return self.opens / self.closes.shift(1) - 1.0

    def intraday_returns(self) -> pd.DataFrame:
        """Open of ``d`` to close of ``d``. The part of a day traded at new weights."""
        return self.closes / self.opens - 1.0

    def average_daily_volume(self, window: int) -> pd.DataFrame:
        """Trailing average share volume, ending at and including each date.

        Used for the participation rate. The window ends at ``t``, so the
        liquidity estimate applied to a trade is one that was observable before
        the trade was placed.
        """
        return self.volumes.rolling(window, min_periods=max(window // 4, 1)).mean()

    def last_membership_date(self) -> pd.Series:
        """The final date each name is a universe member, or NaT if it never exits."""
        member_dates = self.membership.apply(
            lambda column: column[column].index.max() if column.any() else pd.NaT
        )
        never_exits = member_dates == self.dates.max()
        return member_dates.mask(never_exits, pd.NaT)

    def delisting_events(self) -> pd.Series:
        """Name -> (date, terminal return) for names that leave and do not return."""
        last_dates = self.last_membership_date()
        events = last_dates.dropna()
        return events[events.index.isin(self.delisting_returns.dropna().index)]

    def survivors_only_membership(self) -> pd.DataFrame:
        """Membership as a naive backtest would have it: today's names, all history.

        Every name that is a member on the FINAL date is treated as a member on
        every date it has a price. This is the survivorship bias, implemented
        deliberately so the honest run has something to be compared against.
        """
        final = self.membership.iloc[-1]
        survivors = final[final].index
        naive = pd.DataFrame(False, index=self.closes.index, columns=self.closes.columns)
        naive.loc[:, survivors] = self.closes.loc[:, survivors].notna()
        return naive

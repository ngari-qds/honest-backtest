"""The ``Signal`` type, and the causality check the engine runs before it will trade.

POINT-IN-TIME DISCIPLINE BY CONSTRUCTION
----------------------------------------
The usual way a backtest lies is that somebody aligns a signal to a return and
gets the shift backwards. Documenting the convention does not prevent it; the
person who makes that mistake is the person who did not read the docstring.

So this engine does not accept an alignment from the caller at all:

* A ``Signal`` carries values whose row ``t`` is known **at the close of date t**.
  That is the only convention there is. There is no parameter to change it.
* The caller never supplies returns. The engine derives them from ``MarketData``,
  so there is no opportunity to pair a signal with the wrong one.
* The engine applies the reporting lag and the execution delay itself.
* ``Signal.from_builder`` runs a causality check before returning: it recomputes
  the signal on data whose future has been overwritten with noise, and refuses
  to construct if any past value changed.

The result is that producing a look-ahead result requires deliberately
constructing a ``Signal`` whose values already contain the future — which is a
thing you have to mean.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: A builder maps a panel of close prices to a signal frame of the same shape.
SignalBuilder = Callable[[pd.DataFrame], pd.DataFrame]


class LookaheadError(AssertionError):
    """Raised when a signal builder's past output depends on future data."""


@dataclass(frozen=True)
class Signal:
    """Signal values whose row ``t`` is observable at the close of date ``t``.

    Construct with ``from_builder`` wherever possible: it is the path that gets
    checked. The bare constructor exists for signals that came from outside this
    process, and ``was_causality_checked`` records which path was taken so the
    run manifest can say so.
    """

    values: pd.DataFrame
    name: str = "signal"
    was_causality_checked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.values.index, pd.DatetimeIndex):
            raise ValueError("Signal values must be indexed by date.")
        if not self.values.index.is_monotonic_increasing:
            raise ValueError("Signal dates must be sorted ascending.")

    @classmethod
    def from_builder(
        cls,
        builder: SignalBuilder,
        closes: pd.DataFrame,
        *,
        name: str = "signal",
        check_causality: bool = True,
        seed: int = 0,
    ) -> Signal:
        """Build a signal and verify it cannot see the future.

        Set ``check_causality=False`` only for a builder that is provably causal
        and expensive to run twice; the resulting ``Signal`` records that it was
        not checked, and the run manifest reports it.
        """
        values = builder(closes)
        if values.shape != closes.shape:
            raise ValueError(
                f"Signal builder returned {values.shape}, expected {closes.shape}. "
                "The signal must be defined on the same dates and names as the market."
            )
        if check_causality:
            assert_causal(builder, closes, seed=seed)
        return cls(values=values, name=name, was_causality_checked=check_causality)

    def available_at(self, *, reporting_lag_days: int) -> pd.DataFrame:
        """Signal values as they could actually have been used, after reporting lag.

        A lag of ``n`` means a value referring to date ``t`` is first usable on
        date ``t + n``. With the default of one business day, a signal computed
        from the close of Monday is usable from Tuesday, which is the earliest a
        real desk could have acted on it.
        """
        if reporting_lag_days < 0:
            raise ValueError("reporting_lag_days must not be negative.")
        if reporting_lag_days == 0:
            return self.values.copy()
        return self.values.shift(reporting_lag_days)


def assert_causal(
    builder: SignalBuilder,
    closes: pd.DataFrame,
    *,
    seed: int = 0,
    cutoff_fraction: float = 0.6,
) -> None:
    """Recompute the signal with the future overwritten; every past value must match.

    This is the general form of a look-ahead test. It makes no assumption about
    how the signal is computed, so it keeps working when the implementation
    changes — which a hand-written alignment assertion does not.

    The corruption is multiplicative on prices and bounded away from zero, so the
    corrupted panel stays a legal price series: the aim is to prove the past
    cannot see the future, not to test NaN handling.
    """
    if not 0.0 < cutoff_fraction < 1.0:
        raise ValueError("cutoff_fraction must be strictly between 0 and 1.")

    cutoff_position = int(len(closes.index) * cutoff_fraction)
    cutoff = closes.index[cutoff_position]

    rng = np.random.default_rng(seed)
    corrupted = closes.copy()
    future = corrupted.index > cutoff
    multipliers = rng.uniform(0.25, 4.0, size=(int(future.sum()), corrupted.shape[1]))
    corrupted.loc[future, :] = corrupted.loc[future, :].to_numpy() * multipliers

    baseline = builder(closes).loc[:cutoff]
    recomputed = builder(corrupted).loc[:cutoff]

    if not baseline.equals(recomputed):
        difference = (baseline - recomputed).abs()
        worst = float(np.nanmax(difference.to_numpy())) if difference.size else float("nan")
        n_changed = int((difference.fillna(0.0) > 0).to_numpy().sum())
        first_changed = difference.index[(difference.fillna(0.0) > 0).any(axis=1)]
        raise LookaheadError(
            "Signal builder is not causal: overwriting data after "
            f"{cutoff.date()} changed {n_changed} values at or before it "
            f"(largest change {worst:.6g}, earliest affected date "
            f"{first_changed[0].date() if len(first_changed) else 'unknown'}). "
            "The signal is reading the future."
        )


def rank_cross_section(
    values: pd.DataFrame, *, eligible: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Percentile rank within each date's cross-section.

    Ties are broken by ``method="first"``, which is deterministic given column
    order. A nondeterministic tiebreak would make a backtest irreproducible.
    """
    frame = values if eligible is None else values.where(eligible)
    return frame.rank(axis=1, method="first", pct=True)

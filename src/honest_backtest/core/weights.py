"""Turning a ranked cross-section into target weights.

Two constraints here exist because ignoring them is how a backtest quietly
becomes untradeable:

* **Hard-to-borrow names cannot be shorted.** They are removed from the short
  leg and the remaining shorts absorb the exposure. They are not silently
  shorted anyway, and the count of blocked names is reported so the reader can
  see how much of the short leg was unavailable.
* **A per-name weight cap.** Without one, a thin cross-section produces a book
  with three names in it at 33% each, which the metrics will happily annualise
  into a Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from honest_backtest.config import PortfolioConfig


@dataclass(frozen=True)
class TargetWeights:
    weights: pd.DataFrame
    n_long: pd.Series
    n_short: pd.Series
    #: Names that ranked into the short bucket but could not be borrowed.
    n_blocked_shorts: pd.Series


#: Iterations of the water-filling loop. Converges in two or three in practice.
_CAP_ITERATIONS = 20
_CAP_TOLERANCE = 1e-12


def _cap_and_renormalise(leg: pd.DataFrame, *, gross: float, cap: float) -> pd.DataFrame:
    """Scale a leg to ``gross`` without letting any name exceed ``cap``.

    Water-filling: clip the names that breach the cap, then redistribute the
    shortfall proportionally among those with headroom, and repeat.

    **The cap wins.** If the leg has too few names for the cap to be satisfiable
    — five names at a 5% cap cannot add up to 100% — the book is left
    *under-invested* rather than the cap being quietly violated. A backtest that
    silently breaches its own concentration limit to hit a gross target is
    reporting a portfolio nobody would have been allowed to hold.
    """
    signs = np.sign(leg.sum(axis=1)).replace(0.0, 1.0)
    magnitude = leg.abs()

    total = magnitude.sum(axis=1).replace(0.0, np.nan)
    weights = magnitude.div(total, axis=0) * gross

    for _ in range(_CAP_ITERATIONS):
        capped = weights.clip(upper=cap)
        shortfall = gross - capped.sum(axis=1)
        if float(shortfall.max(skipna=True) or 0.0) <= _CAP_TOLERANCE:
            weights = capped
            break

        headroom = (cap - capped).where(magnitude > 0).clip(lower=0.0)
        available = headroom.sum(axis=1)
        if float(available.max(skipna=True) or 0.0) <= _CAP_TOLERANCE:
            weights = capped  # cap is infeasible; stay under-invested
            break

        # Fill proportionally to remaining headroom, never beyond it.
        fillable = shortfall.clip(upper=available)
        weights = capped + headroom.div(available.replace(0.0, np.nan), axis=0).mul(
            fillable, axis=0
        ).fillna(0.0)

    return weights.mul(signs, axis=0).fillna(0.0)


def target_weights(
    ranks: pd.DataFrame,
    cfg: PortfolioConfig,
    *,
    shortable: pd.DataFrame | None = None,
    signal_values: pd.DataFrame | None = None,
) -> TargetWeights:
    """Long the top bucket, short the bottom, subject to borrow and the weight cap.

    ``ranks`` is a percentile rank per date. ``shortable`` is True where a short
    is permitted; pass ``None`` to allow every short, which is what the naive
    configuration does.
    """
    edge = 1.0 / cfg.n_buckets
    long_mask = ranks > (1.0 - edge)
    short_mask = ranks <= edge

    blocked = pd.Series(0, index=ranks.index, dtype="int64")
    if shortable is not None:
        aligned = shortable.reindex_like(ranks).fillna(False).astype(bool)
        blocked = (short_mask & ~aligned).sum(axis=1).astype("int64")
        short_mask = short_mask & aligned

    if cfg.weighting == "equal":
        long_raw = long_mask.astype("float64")
        short_raw = short_mask.astype("float64")
    elif cfg.weighting == "signal_proportional":
        if signal_values is None:
            raise ValueError("signal_proportional weighting needs signal_values")
        centred = signal_values.sub(signal_values.mean(axis=1), axis=0)
        long_raw = centred.where(long_mask).clip(lower=0.0).fillna(0.0)
        short_raw = (-centred).where(short_mask).clip(lower=0.0).fillna(0.0)
    else:
        raise ValueError(f"Unsupported weighting: {cfg.weighting!r}")

    n_long = long_mask.sum(axis=1)
    n_short = short_mask.sum(axis=1)

    # A leg too thin to be a portfolio is emptied rather than concentrated.
    viable = (n_long >= cfg.min_names_per_side) & (n_short >= cfg.min_names_per_side)
    long_raw = long_raw.where(viable, 0.0)
    short_raw = short_raw.where(viable, 0.0)

    longs = _cap_and_renormalise(long_raw, gross=cfg.leg_gross, cap=cfg.max_weight_per_name)
    shorts = _cap_and_renormalise(short_raw, gross=cfg.leg_gross, cap=cfg.max_weight_per_name)

    weights = longs.sub(shorts).fillna(0.0)
    weights = weights.where(viable, 0.0)

    return TargetWeights(
        weights=weights,
        n_long=n_long.where(viable, 0).astype("int64"),
        n_short=n_short.where(viable, 0).astype("int64"),
        n_blocked_shorts=blocked,
    )


def rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """The last trading date of each month or week present in ``dates``."""
    if frequency == "monthly":
        key = dates.to_period("M")
    elif frequency == "weekly":
        key = dates.to_period("W")
    else:
        raise ValueError(f"Unsupported rebalance frequency: {frequency!r}")
    grouped = pd.Series(dates, index=dates).groupby(key).max()
    return pd.DatetimeIndex(grouped.to_numpy(), name="date")


def drift_within_periods(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
    period_id: pd.Series,
) -> pd.DataFrame:
    """Carry target weights forward, letting positions drift with returns.

    Within a holding period each position grows by its own cumulative return; the
    legs are then renormalised separately to hold gross exposure steady. Legs are
    handled separately because a dollar-neutral book sums to roughly zero, and
    dividing by that is meaningless.
    """
    growth = (1.0 + returns.fillna(0.0)).groupby(period_id).cumprod()
    held = targets * growth

    longs = held.clip(lower=0.0)
    shorts = held.clip(upper=0.0)

    target_long_gross = targets.clip(lower=0.0).sum(axis=1)
    target_short_gross = targets.clip(upper=0.0).abs().sum(axis=1)

    long_scale = target_long_gross.div(longs.sum(axis=1).replace(0.0, np.nan))
    short_scale = target_short_gross.div(shorts.abs().sum(axis=1).replace(0.0, np.nan))

    return longs.mul(long_scale.fillna(0.0), axis=0).add(
        shorts.mul(short_scale.fillna(0.0), axis=0)
    )

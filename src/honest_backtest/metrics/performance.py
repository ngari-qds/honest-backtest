"""Return, risk and drawdown statistics for a daily series.

Conventions match ``momentum-stress-study`` deliberately, so numbers from the
two repositories mean the same thing:

* Annualised return is geometric.
* Volatility is the daily standard deviation times ``sqrt(252)``.
* A dollar-neutral long-short book gets no risk-free subtraction — it is
  self-financing, so there is no cash leg to net off.
* Drawdown episodes still under water at the end of the sample are reported as
  unrecovered, with an open recovery date. Closing them at the last observation
  understates how long they last.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).dropna().astype("float64")


def cumulative_growth(returns: pd.Series) -> pd.Series:
    clean = _clean(returns)
    if clean.empty:
        return clean
    return (1.0 + clean).cumprod()


def annualised_return(returns: pd.Series, *, periods_per_year: int) -> float:
    clean = _clean(returns)
    if clean.empty:
        return float("nan")
    total = float((1.0 + clean).prod())
    if total <= 0.0:
        return float("-inf")
    return total ** (periods_per_year / len(clean)) - 1.0


def annualised_volatility(returns: pd.Series, *, periods_per_year: int) -> float:
    clean = _clean(returns)
    if len(clean) < 2:
        return float("nan")
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, *, periods_per_year: int) -> float:
    clean = _clean(returns)
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return float("nan")
    return float(clean.mean() / clean.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    growth = cumulative_growth(returns)
    if growth.empty:
        return float("nan")
    return float((growth / growth.cummax() - 1.0).min())


def drawdown_path(returns: pd.Series) -> pd.Series:
    growth = cumulative_growth(returns)
    if growth.empty:
        return growth
    out = growth / growth.cummax() - 1.0
    out.name = "drawdown"
    return out


def hit_rate(returns: pd.Series) -> float:
    clean = _clean(returns)
    return float("nan") if clean.empty else float((clean > 0).mean())


def drawdown_table(returns: pd.Series, *, threshold: float) -> pd.DataFrame:
    """Every peak-to-recovery episode deeper than ``threshold``."""
    path = drawdown_path(returns)
    columns = [
        "peak",
        "trough",
        "recovery",
        "depth",
        "days_to_trough",
        "days_to_recover",
        "recovered",
    ]
    if path.empty:
        return pd.DataFrame(columns=columns)

    under = path < 0
    episodes: list[tuple[pd.Timestamp, pd.Timestamp, bool]] = []
    start: pd.Timestamp | None = None
    for timestamp, is_under in under.items():
        if is_under and start is None:
            start = timestamp
        elif not is_under and start is not None:
            episodes.append((start, timestamp, True))
            start = None
    if start is not None:
        episodes.append((start, path.index[-1], False))

    rows = []
    for begin, end, recovered in episodes:
        window = path.loc[begin:end]
        depth = float(window.min())
        if depth > -abs(threshold):
            continue
        trough = window.idxmin()
        begin_position = path.index.get_loc(begin)
        peak = path.index[max(int(begin_position) - 1, 0)]
        rows.append(
            {
                "peak": peak,
                "trough": trough,
                "recovery": end if recovered else pd.NaT,
                "depth": depth,
                "days_to_trough": int(path.index.get_loc(trough) - path.index.get_loc(peak)),
                "days_to_recover": (
                    int(path.index.get_loc(end) - path.index.get_loc(trough)) if recovered else -1
                ),
                "recovered": recovered,
            }
        )

    table = pd.DataFrame(rows, columns=columns)
    return table if table.empty else table.sort_values("depth").reset_index(drop=True)


@dataclass(frozen=True)
class Summary:
    n_days: int
    start: str
    end: str
    ann_return: float
    ann_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    skew: float

    def to_series(self, name: str) -> pd.Series:
        return pd.Series(asdict(self), name=name)


def summarise(returns: pd.Series, *, periods_per_year: int) -> Summary:
    clean = _clean(returns)
    return Summary(
        n_days=len(clean),
        start=str(clean.index.min().date()) if len(clean) else "",
        end=str(clean.index.max().date()) if len(clean) else "",
        ann_return=annualised_return(clean, periods_per_year=periods_per_year),
        ann_volatility=annualised_volatility(clean, periods_per_year=periods_per_year),
        sharpe=sharpe_ratio(clean, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(clean),
        hit_rate=hit_rate(clean),
        skew=float(clean.skew()) if len(clean) > 2 else float("nan"),
    )


def summary_table(
    series_by_name: Mapping[str, pd.Series], *, periods_per_year: int
) -> pd.DataFrame:
    rows = {
        name: summarise(series, periods_per_year=periods_per_year).to_series(name)
        for name, series in series_by_name.items()
    }
    return pd.DataFrame(rows).T

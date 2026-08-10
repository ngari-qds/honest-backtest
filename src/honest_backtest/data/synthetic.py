"""SYNTHETIC market generator. Nothing produced here is real market data.

Why the demo is synthetic
-------------------------
This repository is about accounting, not about alpha. Running the naive and the
honest configuration against a *generated* market with a **planted, known**
momentum effect means the gap between them can be checked against ground truth
instead of guessed at: we know exactly how much edge went in, so we can say how
much of it each unrealistic assumption manufactured.

Real prices would make the demo prettier and the claim weaker, because there
would be no true answer to compare against — and a repository whose headline
number is "my backtester says this strategy earns X" is exactly the kind of
thing this repository exists to argue against.

Everything generated here carries ``is_synthetic=True`` on the resulting
``MarketData``, which propagates into every table, chart title and README
figure. Every written file is named ``synthetic_*``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from honest_backtest.config import SyntheticDataConfig
from honest_backtest.data.market import MarketData

BORROW_TIERS = ("general_collateral", "warm", "hot")

#: Formation window of the planted effect, in trading days. These MUST match the
#: signal the demo trades (``study.MOMENTUM_LOOKBACK_DAYS`` / ``MOMENTUM_SKIP_DAYS``),
#: otherwise the demo trades one signal while the market rewards a different one
#: and the naive-versus-honest comparison measures nothing useful.
#: ``tests/test_synthetic.py`` asserts the two stay in step.
PLANTED_LOOKBACK_DAYS = 252
PLANTED_SKIP_DAYS = 21


def _business_days(start: str, n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n_days, name="date")


def generate_market(cfg: SyntheticDataConfig, *, seed: int) -> MarketData:
    """Build a synthetic daily market with a planted cross-sectional momentum effect.

    The data-generating process, stated plainly so the demo's ground truth is
    auditable:

        r[i, t] = mu + k * z[i, t-1] * sigma_daily + eps[i, t]

    where ``z`` is the cross-sectionally standardised trailing 60-day return as
    of the *previous* close, ``k`` converts the configured annual strength into
    daily units, and ``eps`` is i.i.d. normal. The dependence on ``t-1`` is what
    makes the effect real and tradeable rather than contemporaneous — a
    contemporaneous effect would be unexploitable and would make the honest run
    look wrongly bad.

    Names delist by leaving the membership calendar on a chosen date, and a
    terminal return is recorded for that date. They are not deleted from the
    price frames: a backtest should have to *decide* what to do about a
    delisting, not be spared the question.
    """
    rng = np.random.default_rng(seed)
    dates = _business_days(cfg.start, cfg.n_days)
    names = pd.Index([f"SYN{i:04d}" for i in range(cfg.n_names)], name="ticker")
    n_days, n_names = len(dates), len(names)

    daily_sigma = cfg.annual_volatility / np.sqrt(252.0)
    daily_mu = cfg.annual_drift / 252.0
    daily_k = cfg.planted_momentum_strength / 252.0 / daily_sigma if daily_sigma else 0.0

    returns = np.zeros((n_days, n_names))
    log_price = np.zeros((n_days, n_names))
    log_price[0] = np.log(rng.uniform(15.0, 250.0, n_names))

    for t in range(1, n_days):
        # The signal is the one an observer could have computed at the close of
        # t-1: formation ends PLANTED_SKIP_DAYS before that, and begins
        # PLANTED_LOOKBACK_DAYS before that. Everything it touches is strictly in
        # the past, so the planted effect is genuinely tradeable rather than
        # contemporaneous.
        if t > PLANTED_LOOKBACK_DAYS + 1:
            recent = log_price[t - 1 - PLANTED_SKIP_DAYS]
            distant = log_price[t - 1 - PLANTED_LOOKBACK_DAYS]
            trailing = recent - distant
            spread = trailing.std()
            z = (trailing - trailing.mean()) / spread if spread > 0 else np.zeros(n_names)
        else:
            z = np.zeros(n_names)

        shock = rng.normal(0.0, daily_sigma, n_names)
        returns[t] = daily_mu + daily_k * z * daily_sigma + shock
        log_price[t] = log_price[t - 1] + returns[t]

    closes = pd.DataFrame(np.exp(log_price), index=dates, columns=names)

    # Opens gap from the previous close by a fraction of the day's move, so that
    # overnight and intraday returns are distinct and the trade-timing choice
    # actually matters.
    overnight_share = rng.uniform(0.2, 0.6, (n_days, n_names))
    opens = closes.shift(1) * np.exp(overnight_share * returns)
    opens.iloc[0] = closes.iloc[0]

    volumes = pd.DataFrame(
        rng.lognormal(mean=13.0, sigma=1.1, size=(n_days, n_names)),
        index=dates,
        columns=names,
    ).round()

    membership, delisting_returns = _membership_and_delistings(cfg, rng, dates, names)
    borrow_tier = pd.Series(
        rng.choice(BORROW_TIERS, size=n_names, p=[0.75, 0.18, 0.07]),
        index=names,
        name="borrow_tier",
    )

    hard_to_borrow = _hard_to_borrow_flags(cfg, rng, dates, names)

    # Prices exist for every name on every date, including after it leaves the
    # universe. That is deliberate: the engine must decide what to do about a
    # departure rather than being rescued by the data going missing.
    return MarketData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        membership=membership,
        delisting_returns=delisting_returns,
        borrow_tier=borrow_tier,
        hard_to_borrow=hard_to_borrow,
        is_synthetic=True,
    )


def _membership_and_delistings(
    cfg: SyntheticDataConfig,
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    names: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    """Staggered entries, and exits for the delisting cohort.

    Entries are staggered so the universe grows: a backtest that assumes a fixed
    name set from day one is not being asked the awkward question.
    """
    n_days, n_names = len(dates), len(names)
    membership = pd.DataFrame(True, index=dates, columns=names)

    # A third of the names join partway through the sample.
    late_joiners = rng.choice(n_names, size=n_names // 3, replace=False)
    join_positions = rng.integers(1, int(n_days * 0.7), size=len(late_joiners))
    for name_position, join_position in zip(late_joiners, join_positions, strict=True):
        membership.iloc[:join_position, name_position] = False

    n_delisting = round(cfg.delisting_rate * n_names)
    delisting_names = rng.choice(n_names, size=n_delisting, replace=False)
    exit_positions = rng.integers(int(n_days * 0.2), n_days - 1, size=n_delisting)

    terminal: dict[str, float] = {}
    for name_position, exit_position in zip(delisting_names, exit_positions, strict=True):
        membership.iloc[exit_position + 1 :, name_position] = False
        terminal[str(names[name_position])] = cfg.delisting_terminal_return

    delisting_returns = pd.Series(terminal, name="delisting_return", dtype="float64")
    delisting_returns.index.name = "ticker"
    return membership, delisting_returns


def _hard_to_borrow_flags(
    cfg: SyntheticDataConfig,
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    names: pd.Index,
) -> pd.DataFrame:
    """Hard-to-borrow status, which comes and goes rather than being permanent."""
    n_days, n_names = len(dates), len(names)
    flags = pd.DataFrame(False, index=dates, columns=names)

    n_affected = round(cfg.hard_to_borrow_rate * n_names)
    affected = rng.choice(n_names, size=n_affected, replace=False)
    for name_position in affected:
        # One contiguous hard-to-borrow spell of a few months.
        start = int(rng.integers(0, max(n_days - 200, 1)))
        length = int(rng.integers(40, 200))
        flags.iloc[start : start + length, name_position] = True
    return flags

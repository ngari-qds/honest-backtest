"""Shared SYNTHETIC fixtures.

Everything here is generated and seeded. None of it is market data. Fixture
helpers carry ``synthetic`` or ``toy`` in their names so a value from this module
cannot be mistaken for a result in a traceback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from honest_backtest.config import Config, SyntheticDataConfig
from honest_backtest.data.market import MarketData
from honest_backtest.data.synthetic import generate_market

SYNTHETIC_SEED = 4242


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def small_synthetic_config() -> SyntheticDataConfig:
    """A small generated market: fast, and still exercises every code path."""
    return SyntheticDataConfig(
        n_names=40,
        start="2015-01-05",
        n_days=800,
        delisting_rate=0.15,
        hard_to_borrow_rate=0.10,
    )


@pytest.fixture
def synthetic_market(small_synthetic_config: SyntheticDataConfig) -> MarketData:
    return generate_market(small_synthetic_config, seed=SYNTHETIC_SEED)


@pytest.fixture(scope="session")
def demo_synthetic_market() -> MarketData:
    """The market the demo actually runs on: 400 names, ~15 years.

    Session-scoped because it takes a couple of seconds to generate and several
    tests need it. Claims about the demo's behaviour are tested against the demo's
    own configuration rather than against a small fixture where the planted
    effect would be indistinguishable from noise.
    """
    return generate_market(SyntheticDataConfig(), seed=Config().seed)


@pytest.fixture(scope="session")
def strong_effect_market() -> MarketData:
    """A market with an exaggerated planted effect, for testing the generator.

    Used where a test needs the planted effect to be unmistakable rather than
    merely present, so that a failure means "the generator is broken" and not
    "the sample was small".
    """
    return generate_market(
        SyntheticDataConfig(n_names=150, n_days=2_000, planted_momentum_strength=0.40),
        seed=SYNTHETIC_SEED,
    )


def business_days(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n, name="date")


def toy_market(
    *,
    n_days: int = 60,
    names: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    hard_to_borrow: tuple[str, ...] = (),
    delisting: dict[str, int] | None = None,
) -> MarketData:
    """A tiny, fully deterministic market for hand-computed tests. SYNTHETIC.

    Prices rise by a fixed daily amount per name, so any expected value can be
    written down by hand rather than compared against the code that produced it.
    """
    dates = business_days(n_days)
    steps = np.linspace(0.0005, 0.002, len(names))
    closes = pd.DataFrame(
        {name: 100.0 * (1.0 + steps[i]) ** np.arange(n_days) for i, name in enumerate(names)},
        index=dates,
    )
    opens = closes.shift(1).fillna(closes.iloc[0])
    volumes = pd.DataFrame(1_000_000.0, index=dates, columns=list(names))

    membership = pd.DataFrame(True, index=dates, columns=list(names))
    terminal: dict[str, float] = {}
    if delisting:
        for name, position in delisting.items():
            membership.iloc[position + 1 :, membership.columns.get_loc(name)] = False
            terminal[name] = -0.30

    htb = pd.DataFrame(False, index=dates, columns=list(names))
    for name in hard_to_borrow:
        htb[name] = True

    return MarketData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        membership=membership,
        delisting_returns=pd.Series(terminal, dtype="float64"),
        borrow_tier=pd.Series("general_collateral", index=list(names)),
        hard_to_borrow=htb,
        is_synthetic=True,
    )

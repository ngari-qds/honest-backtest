"""honest-backtest: a vectorized cross-sectional backtester that refuses to flatter.

The public API is small on purpose. A caller supplies market data, a signal and a
configuration; it never supplies returns and never chooses an alignment, because
those are the two places a backtest usually starts lying.

    from honest_backtest import Config, MarketData, RealismConfig, Signal, run

    signal = Signal.from_builder(my_builder, market.closes)   # causality-checked
    honest = run(market, signal, cfg, label="honest")
    naive = run(market, signal, cfg.with_realism(RealismConfig.naive()), label="naive")
"""

from honest_backtest.config import Config, RealismConfig, load_config
from honest_backtest.core.engine import BacktestResult, run
from honest_backtest.core.signal import LookaheadError, Signal, assert_causal
from honest_backtest.data.market import MarketData, MarketDataError

__all__ = [
    "BacktestResult",
    "Config",
    "LookaheadError",
    "MarketData",
    "MarketDataError",
    "RealismConfig",
    "Signal",
    "assert_causal",
    "load_config",
    "run",
]
__version__ = "0.1.0"

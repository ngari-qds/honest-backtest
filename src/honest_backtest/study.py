"""The demonstration this repository exists for: the same strategy, run twice.

One strategy, one market, two sets of assumptions. The naive run switches off
every protection at once. The honest run leaves them on. The gap between them is
the number the README leads with.

Because "the gap is large" is not by itself useful, ``realism_ladder`` also
re-enables the protections one at a time, so the gap can be attributed to
specific assumptions rather than left as a single discouraging number. That
ladder is the most useful table in the repository: it says which lie was doing
the work.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from honest_backtest.config import Config, RealismConfig
from honest_backtest.core.engine import BacktestResult, run
from honest_backtest.core.signal import Signal, SignalBuilder
from honest_backtest.data.market import MarketData
from honest_backtest.data.synthetic import generate_market
from honest_backtest.metrics import attribution, performance

#: Formation and skip for the demo signal, in trading days (~12 months, ~1 month).
MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_SKIP_DAYS = 21

#: Order in which protections are switched back on for the ladder. Costs first
#: because they are the ones everybody remembers, so the interesting result is
#: how much damage the later rungs still do after costs are already charged.
LADDER: tuple[tuple[str, str], ...] = (
    ("apply_costs", "charge commission, spread and impact"),
    ("apply_borrow_costs", "charge stock borrow on the short leg"),
    ("enforce_hard_to_borrow", "refuse shorts in hard-to-borrow names"),
    ("apply_reporting_lag", "delay the signal by its reporting lag"),
    ("respect_universe_membership", "trade only that date's actual members"),
    ("honour_delistings", "apply terminal returns on delisting"),
)


def momentum_builder(
    *,
    lookback_days: int = MOMENTUM_LOOKBACK_DAYS,
    skip_days: int = MOMENTUM_SKIP_DAYS,
) -> SignalBuilder:
    """Twelve-month price momentum skipping the most recent month.

    The value at row ``t`` uses closes at ``t - skip`` and ``t - lookback``, both
    strictly in the past, so the builder is causal and ``Signal.from_builder``
    will accept it. The engine applies the reporting lag and the execution delay
    on top of that; this function does not get a say in either.
    """

    def build(closes: pd.DataFrame) -> pd.DataFrame:
        return closes.shift(skip_days) / closes.shift(lookback_days) - 1.0

    return build


def build_market(cfg: Config) -> MarketData:
    """Generate the SYNTHETIC demo market. See data/synthetic.py for why."""
    return generate_market(cfg.synthetic, seed=cfg.seed)


def build_signal(market: MarketData, cfg: Config) -> Signal:
    return Signal.from_builder(
        momentum_builder(),
        market.closes,
        name="momentum_12_1",
        check_causality=True,
        seed=cfg.seed,
    )


@dataclass(frozen=True)
class Comparison:
    naive: BacktestResult
    honest: BacktestResult
    summary: pd.DataFrame
    ladder: pd.DataFrame
    attribution: pd.DataFrame
    capacity: pd.DataFrame
    is_synthetic: bool


def _stats(result: BacktestResult, cfg: Config) -> dict[str, float]:
    periods = cfg.metrics.trading_days_per_year
    net = result.net_return
    return {
        "ann_return": performance.annualised_return(net, periods_per_year=periods),
        "ann_volatility": performance.annualised_volatility(net, periods_per_year=periods),
        "sharpe": performance.sharpe_ratio(net, periods_per_year=periods),
        "max_drawdown": performance.max_drawdown(net),
        "hit_rate": performance.hit_rate(net),
        "ann_turnover": float(result.turnover.sum() / (len(net) / periods)),
        "mean_blocked_shorts": float(result.n_blocked_shorts.mean()),
    }


def run_comparison(market: MarketData, signal: Signal, cfg: Config) -> Comparison:
    """Run the strategy naively and honestly, and account for the difference."""
    naive_cfg = cfg.with_realism(RealismConfig.naive())
    naive = run(market, signal, naive_cfg, label="naive")
    honest = run(market, signal, cfg, label="honest")

    naive_stats = _stats(naive, cfg)
    honest_stats = _stats(honest, cfg)
    summary = pd.DataFrame({"naive": naive_stats, "honest": honest_stats})
    summary["difference"] = summary["honest"] - summary["naive"]
    summary["pct_of_naive_remaining"] = (summary["honest"] / summary["naive"]).where(
        summary["naive"] != 0
    )
    summary.index.name = "statistic"

    return Comparison(
        naive=naive,
        honest=honest,
        summary=summary,
        ladder=realism_ladder(market, signal, cfg),
        attribution=attribution.attribute(
            honest.gross_return,
            honest.net_return,
            honest.costs,
            periods_per_year=cfg.metrics.trading_days_per_year,
        ),
        capacity=attribution.capacity_estimate(
            honest.gross_return,
            honest.costs.impact,
            honest.participation,
            cfg.costs,
            cfg.metrics,
        ),
        is_synthetic=market.is_synthetic,
    )


def realism_ladder(market: MarketData, signal: Signal, cfg: Config) -> pd.DataFrame:
    """Re-enable protections one at a time, recording what each one costs.

    Starts from fully naive and walks to fully honest. The ``sharpe_change``
    column attributes the total gap to individual assumptions, which is the
    difference between "your backtest is optimistic" and "your backtest is
    optimistic *because* it assumes you can short anything you like".

    The attribution is order-dependent — these effects interact, and a different
    ordering would apportion the interactions differently. The column header says
    "cumulative" for that reason, and the ladder is a decomposition rather than a
    set of independent effects.
    """
    settings = RealismConfig.naive().model_dump()
    rows: list[dict[str, object]] = []

    previous_sharpe: float | None = None
    for step, (field, description) in enumerate((("none", "fully naive"), *LADDER)):
        if field != "none":
            settings[field] = True
        realism = RealismConfig(**settings)
        result = run(market, signal, cfg.with_realism(realism), label=field)
        stats = _stats(result, cfg)

        rows.append(
            {
                "step": step,
                "enabled": field,
                "description": description,
                "sharpe": stats["sharpe"],
                "sharpe_change": (
                    float("nan") if previous_sharpe is None else stats["sharpe"] - previous_sharpe
                ),
                "ann_return": stats["ann_return"],
                "max_drawdown": stats["max_drawdown"],
                "ann_turnover": stats["ann_turnover"],
            }
        )
        previous_sharpe = stats["sharpe"]

    table = pd.DataFrame(rows).set_index("step")
    return table


def ground_truth(cfg: Config) -> pd.DataFrame:
    """What was planted in the synthetic market, for comparison against results.

    Stating this alongside the results is the reason the demo uses generated
    data: a backtest that recovers more edge than was put in is broken, and
    without a ground truth there would be no way to notice.
    """
    return pd.DataFrame(
        {
            "planted_momentum_strength_annual": [cfg.synthetic.planted_momentum_strength],
            "annual_volatility": [cfg.synthetic.annual_volatility],
            "annual_drift": [cfg.synthetic.annual_drift],
            "delisting_rate": [cfg.synthetic.delisting_rate],
            "delisting_terminal_return": [cfg.synthetic.delisting_terminal_return],
            "hard_to_borrow_rate": [cfg.synthetic.hard_to_borrow_rate],
            "n_names": [cfg.synthetic.n_names],
            "n_days": [cfg.synthetic.n_days],
        }
    )

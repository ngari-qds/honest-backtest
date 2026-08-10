"""Run the naive-versus-honest comparison and regenerate results/.

    uv run python scripts/run_study.py

Regenerates every figure and table the README refers to. No network access.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from honest_backtest import plotting
from honest_backtest.config import load_config
from honest_backtest.metrics import performance
from honest_backtest.study import build_market, build_signal, ground_truth, run_comparison

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_study")


def _write(frame: pd.DataFrame, tables: Path, name: str) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{name}.csv")
    logger.info("wrote %s.csv", name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    cfg = load_config(args.config)
    figures, tables, dpi = cfg.output.figures_dir, cfg.output.tables_dir, cfg.output.dpi

    logger.info("Building synthetic market")
    market = build_market(cfg)

    logger.info("Building signal (causality-checked)")
    signal = build_signal(market, cfg)

    logger.info("Running naive and honest configurations")
    comparison = run_comparison(market, signal, cfg)

    periods = cfg.metrics.trading_days_per_year
    _write(comparison.summary, tables, "naive_vs_honest_summary")
    _write(comparison.ladder, tables, "realism_ladder")
    _write(comparison.attribution, tables, "cost_attribution")
    _write(comparison.capacity, tables, "capacity_estimate")
    _write(ground_truth(cfg), tables, "synthetic_ground_truth")
    _write(
        performance.drawdown_table(
            comparison.honest.net_return, threshold=cfg.metrics.drawdown_threshold
        ),
        tables,
        "honest_drawdown_episodes",
    )
    _write(
        performance.summary_table(
            {
                "naive": comparison.naive.net_return,
                "honest_gross": comparison.honest.gross_return,
                "honest_net": comparison.honest.net_return,
            },
            periods_per_year=periods,
        ),
        tables,
        "performance_summary",
    )

    naive_growth = performance.cumulative_growth(comparison.naive.net_return)
    honest_growth = performance.cumulative_growth(comparison.honest.net_return)

    plotting.plot_naive_vs_honest(
        naive_growth,
        honest_growth,
        comparison.summary,
        path=figures / "01_naive_vs_honest.png",
        dpi=dpi,
        is_synthetic=comparison.is_synthetic,
    )
    plotting.plot_realism_ladder(
        comparison.ladder,
        path=figures / "02_realism_ladder.png",
        dpi=dpi,
        is_synthetic=comparison.is_synthetic,
    )
    plotting.plot_cost_attribution(
        comparison.attribution,
        path=figures / "03_cost_attribution.png",
        dpi=dpi,
        is_synthetic=comparison.is_synthetic,
    )
    plotting.plot_drawdowns(
        performance.drawdown_path(comparison.naive.net_return),
        performance.drawdown_path(comparison.honest.net_return),
        path=figures / "04_drawdowns.png",
        dpi=dpi,
        is_synthetic=comparison.is_synthetic,
    )

    manifest = {
        "seed": cfg.seed,
        "is_synthetic": comparison.is_synthetic,
        "signal_causality_checked": comparison.honest.signal_causality_checked,
        "n_names": len(market.names),
        "n_days": len(market.dates),
        "start": str(market.dates.min().date()),
        "end": str(market.dates.max().date()),
        "naive_sharpe": float(comparison.summary.loc["sharpe", "naive"]),
        "honest_sharpe": float(comparison.summary.loc["sharpe", "honest"]),
        "honest_caveat": comparison.honest.caveat(),
    }
    # Wall-clock time is deliberately NOT in the manifest. CI reruns the study
    # and fails if results/ changed, so anything that varies between two
    # identical runs would turn that check into noise. Elapsed time is printed
    # below instead, where it informs without being an artefact.
    elapsed = round(time.time() - started, 2)
    (cfg.output.tables_dir.parent / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary = comparison.summary
    print()
    print("SYNTHETIC DATA. Generated market with a planted effect; not a market result.")
    print()
    print(f"  {'':<22}{'naive':>12}{'honest':>12}")
    for row in ("sharpe", "ann_return", "ann_volatility", "max_drawdown", "ann_turnover"):
        print(f"  {row:<22}{summary.loc[row, 'naive']:>12.3f}{summary.loc[row, 'honest']:>12.3f}")
    print()
    print(f"  honest run caveat : {comparison.honest.caveat()}")
    print(f"  elapsed           : {elapsed}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

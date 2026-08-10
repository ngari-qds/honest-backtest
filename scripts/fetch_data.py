"""Generate and cache the SYNTHETIC demo market.

    uv run python scripts/fetch_data.py

This repository's demo does not download anything. The market is generated from
the seed in config/study.yaml, which means the reproduction path has no network
dependency and cannot break because a data provider had a bad morning.

The script is still called fetch_data.py, and still writes to a gitignored
data/ directory, because the shape of the pipeline is the point: swap this one
file for a real download and nothing downstream changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from honest_backtest.config import load_config
from honest_backtest.study import build_market

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_data")

CACHE_NAME = "synthetic_market"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cache = cfg.data_dir / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic market (seed=%d)", cfg.seed)
    market = build_market(cfg)

    # Written with synthetic_ prefixes so a stray file cannot be mistaken for
    # market data if it escapes this directory.
    market.closes.to_parquet(cache / "synthetic_closes.parquet")
    market.opens.to_parquet(cache / "synthetic_opens.parquet")
    market.volumes.to_parquet(cache / "synthetic_volumes.parquet")
    market.membership.to_parquet(cache / "synthetic_membership.parquet")
    market.hard_to_borrow.to_parquet(cache / "synthetic_hard_to_borrow.parquet")
    market.borrow_tier.to_frame().to_parquet(cache / "synthetic_borrow_tier.parquet")
    market.delisting_returns.to_frame().to_parquet(cache / "synthetic_delisting_returns.parquet")

    print()
    print("Synthetic market generated. THIS IS NOT REAL MARKET DATA.")
    print(f"  written to : {cache}")
    print(f"  dates      : {market.dates.min().date()} to {market.dates.max().date()}")
    print(f"  names      : {len(market.names)}")
    print(f"  delistings : {len(market.delisting_events())}")
    print()
    print("Next:  uv run python scripts/run_study.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

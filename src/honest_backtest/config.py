"""Typed configuration.

Two things in here carry the weight of the whole repository.

``RealismConfig`` is a set of switches, each of which turns off one specific
piece of honesty. Every switch defaults to ON. ``RealismConfig.naive()`` turns
them all off at once, and that is the *only* supported way to get the flattering
answer — you cannot arrive at it by accident, and a result produced that way
carries a flag that the reporting layer refuses to drop.

``CostConfig`` is broken into named components rather than a single "cost in bps"
number, because a single number hides which assumption is doing the work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]

Weighting = Literal["equal", "signal_proportional", "inverse_volatility"]
TradeOn = Literal["next_open", "next_close"]
Rebalance = Literal["monthly", "weekly"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RealismConfig(_Base):
    """Each field disables one specific piece of realism when set to False.

    Defaults are all True. The naive preset exists so that the README's central
    comparison is a single, explicit, greppable call rather than six scattered
    keyword arguments — and so nobody can produce the flattering number without
    saying so out loud.
    """

    #: Charge commission, spread and market impact on actual traded notional.
    apply_costs: bool = True
    #: Charge stock-borrow fees on the short leg.
    apply_borrow_costs: bool = True
    #: Refuse to short names on the hard-to-borrow list.
    enforce_hard_to_borrow: bool = True
    #: Delay signal availability by the configured reporting lag.
    apply_reporting_lag: bool = True
    #: Restrict each date's tradeable set to that date's actual membership.
    #: False means trading today's survivors throughout history.
    respect_universe_membership: bool = True
    #: Apply the terminal return of a name that delists mid-period. False means
    #: the position vanishes at its last observed price, which is the single
    #: most common way a backtest quietly deletes its own losses.
    honour_delistings: bool = True

    @classmethod
    def naive(cls) -> RealismConfig:
        """Every assumption at its most flattering. Used only for the comparison."""
        return cls(
            apply_costs=False,
            apply_borrow_costs=False,
            enforce_hard_to_borrow=False,
            apply_reporting_lag=False,
            respect_universe_membership=False,
            honour_delistings=False,
        )

    @property
    def is_naive(self) -> bool:
        return not any(
            (
                self.apply_costs,
                self.apply_borrow_costs,
                self.enforce_hard_to_borrow,
                self.apply_reporting_lag,
                self.respect_universe_membership,
                self.honour_delistings,
            )
        )

    @property
    def disabled_protections(self) -> tuple[str, ...]:
        """Names of every protection currently switched off, for the run label."""
        return tuple(name for name, value in self.model_dump().items() if value is False)


class CostConfig(_Base):
    """Trading costs, split into components that can be reasoned about separately.

    All rates are per side and in basis points of traded notional, except the
    impact coefficient which is defined in the docstring of ``core.costs``.
    """

    #: Broker commission, charged on every share traded, each way.
    commission_bps: float = Field(default=1.0, ge=0.0)
    #: Half the quoted bid-ask spread: what crossing costs on one side.
    half_spread_bps: float = Field(default=4.0, ge=0.0)
    #: Coefficient on the square-root impact term. See core.costs.market_impact.
    impact_coefficient: float = Field(default=10.0, ge=0.0)
    #: Fraction of a name's average daily volume the strategy is assumed able to
    #: consume in one day without extraordinary cost.
    max_participation_rate: float = Field(default=0.10, gt=0.0, le=1.0)
    #: Portfolio notional in currency units. Impact is meaningless without it.
    portfolio_notional: float = Field(default=100_000_000.0, gt=0.0)


class BorrowConfig(_Base):
    """Stock borrow. The short leg is not free and is not always possible."""

    #: Annualised borrow fee for a name with no tier assignment.
    default_annual_bps: float = Field(default=40.0, ge=0.0)
    #: Annualised fee by tier label. Names map to tiers in the borrow file.
    tier_annual_bps: dict[str, float] = Field(
        default_factory=lambda: {"general_collateral": 30.0, "warm": 150.0, "hot": 800.0}
    )
    #: Trading days per year, for converting an annual fee to a daily accrual.
    trading_days_per_year: int = Field(default=252, gt=0)


class SignalConfig(_Base):
    """When a signal value becomes usable."""

    #: Business days between the date a signal value refers to and the date it
    #: could actually have been acted on. Fundamental inputs are the reason this
    #: exists: a quarter ending 31 March is not knowable on 31 March.
    reporting_lag_days: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _lag_is_sane(self) -> SignalConfig:
        if self.reporting_lag_days > 250:
            raise ValueError("reporting_lag_days over 250 business days is almost certainly a typo")
        return self


class PortfolioConfig(_Base):
    n_buckets: int = Field(default=5, ge=2)
    weighting: Weighting = "equal"
    #: Gross exposure per leg. 1.0 each side gives a book with gross 2.0.
    leg_gross: float = Field(default=1.0, gt=0.0)
    #: Hard cap on any single name's absolute weight, applied after weighting.
    max_weight_per_name: float = Field(default=0.05, gt=0.0, le=1.0)
    min_names_per_side: int = Field(default=5, ge=1)


class ExecutionConfig(_Base):
    rebalance: Rebalance = "monthly"
    #: Which price the rebalance trades at. A signal known at the close of t is
    #: never filled at the close of t.
    trade_on: TradeOn = "next_open"


class MetricsConfig(_Base):
    #: Annualisation factor for daily series.
    trading_days_per_year: int = Field(default=252, gt=0)
    #: Drawdown episodes deeper than this appear in the drawdown table.
    drawdown_threshold: float = Field(default=0.05, gt=0.0)
    #: Capacity is reported as the notional at which estimated impact eats this
    #: fraction of gross return.
    capacity_erosion_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)


class SyntheticDataConfig(_Base):
    """Parameters of the GENERATED demo market. Not real market data.

    The demo panel is synthetic on purpose: this repository is about the
    accounting, and synthetic data with a planted effect lets the naive-versus-
    honest gap be measured against a known ground truth instead of guessed at.
    Every artefact built from it is labelled synthetic.
    """

    n_names: int = Field(default=400, ge=10)
    start: str = "2010-01-04"
    n_days: int = Field(default=3_780, ge=250)  # about 15 years of business days
    annual_drift: float = 0.06
    annual_volatility: float = 0.28
    #: Strength of the planted cross-sectional momentum effect, in annualised
    #: return per unit of standardised signal. Set to 0.0 for a null market.
    planted_momentum_strength: float = 0.03
    #: Fraction of names that delist at some point during the sample.
    delisting_rate: float = Field(default=0.15, ge=0.0, lt=1.0)
    #: Terminal return applied to a name on its delisting date.
    delisting_terminal_return: float = Field(default=-0.30, gt=-1.0, le=0.0)
    #: Fraction of names flagged hard to borrow.
    hard_to_borrow_rate: float = Field(default=0.08, ge=0.0, lt=1.0)


class OutputConfig(_Base):
    figures_dir: Path = Path("results/figures")
    tables_dir: Path = Path("results/tables")
    dpi: int = Field(default=150, ge=72)


class Config(_Base):
    realism: RealismConfig = RealismConfig()
    costs: CostConfig = CostConfig()
    borrow: BorrowConfig = BorrowConfig()
    signal: SignalConfig = SignalConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    execution: ExecutionConfig = ExecutionConfig()
    metrics: MetricsConfig = MetricsConfig()
    synthetic: SyntheticDataConfig = SyntheticDataConfig()
    output: OutputConfig = OutputConfig()
    data_dir: Path = Path("data")
    seed: int = 20260810

    def with_realism(self, realism: RealismConfig) -> Config:
        return self.model_copy(update={"realism": realism})

    def resolve(self, root: Path | None = None) -> Config:
        base = (root or REPO_ROOT).resolve()

        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else base / p

        return self.model_copy(
            update={
                "data_dir": _abs(self.data_dir),
                "output": self.output.model_copy(
                    update={
                        "figures_dir": _abs(self.output.figures_dir),
                        "tables_dir": _abs(self.output.tables_dir),
                    }
                ),
            }
        )


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "study.yaml"


def load_config(path: Path | str | None = None, *, root: Path | None = None) -> Config:
    """Load and validate configuration from YAML. Unknown keys are errors."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Expected config/study.yaml; run from the "
            "repository root or pass --config."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")
    return Config(**raw).resolve(root=root)

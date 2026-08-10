"""Figures. matplotlib only.

Every chart drawn from synthetic data says so in its subtitle. That is not
decoration: a chart is the part of a repository most likely to be screenshotted
and shown without its README, so the caveat has to survive the crop.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1a1a1a"
NAIVE = "#b3541e"
HONEST = "#0b6e99"
MUTED = "#8c8c8c"
GRID = "#d9d9d9"

SYNTHETIC_NOTE = "SYNTHETIC DATA — generated market with a planted effect, not a market result"


def _new_axes(figsize: tuple[float, float] = (9.5, 5.6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    return fig, ax


def _title(ax: plt.Axes, title: str, *, is_synthetic: bool) -> None:
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=18)
    if is_synthetic:
        ax.annotate(
            SYNTHETIC_NOTE,
            xy=(0.0, 1.015),
            xycoords="axes fraction",
            fontsize=8.5,
            color=NAIVE,
            annotation_clip=False,
        )


def _save(fig: plt.Figure, path: Path, *, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_naive_vs_honest(
    naive_growth: pd.Series,
    honest_growth: pd.Series,
    summary: pd.DataFrame,
    *,
    path: Path,
    dpi: int,
    is_synthetic: bool,
) -> Path:
    """THE figure. Same strategy, same data, two sets of assumptions."""
    fig, ax = _new_axes()
    ax.plot(
        naive_growth.index,
        naive_growth.to_numpy(),
        color=NAIVE,
        linewidth=2.1,
        label="naive: no costs, no borrow limits, no lag, survivors only",
        zorder=3,
    )
    ax.plot(
        honest_growth.index,
        honest_growth.to_numpy(),
        color=HONEST,
        linewidth=2.1,
        label="honest: everything charged and enforced",
        zorder=4,
    )
    ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=":", zorder=2)
    ax.set_ylabel("Growth of 1 unit of long-leg notional", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    naive_sharpe = summary.loc["sharpe", "naive"]
    honest_sharpe = summary.loc["sharpe", "honest"]
    ax.annotate(
        f"Sharpe {naive_sharpe:.2f}  →  {honest_sharpe:.2f}",
        xy=(0.985, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=11,
        color=INK,
    )
    _title(ax, "The same strategy, run naively and run honestly", is_synthetic=is_synthetic)
    return _save(fig, path, dpi=dpi)


def plot_realism_ladder(
    ladder: pd.DataFrame,
    *,
    path: Path,
    dpi: int,
    is_synthetic: bool,
) -> Path:
    """Waterfall of what each protection costs as it is switched back on."""
    fig, ax = _new_axes(figsize=(10.0, 5.6))
    sharpes = ladder["sharpe"].to_numpy()
    labels = [str(v) for v in ladder["description"]]
    positions = np.arange(len(sharpes))

    ax.step(positions, sharpes, where="mid", color=HONEST, linewidth=2.0, zorder=3)
    ax.scatter(positions, sharpes, color=HONEST, s=34, zorder=4)

    for position, value, change in zip(positions, sharpes, ladder["sharpe_change"], strict=True):
        if np.isfinite(change) and abs(change) > 1e-9:
            ax.annotate(
                f"{change:+.2f}",
                xy=(position, value),
                xytext=(0, -16 if change < 0 else 10),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=NAIVE if change < 0 else INK,
            )

    ax.axhline(0.0, color=INK, linewidth=1.0, zorder=2)
    ax.set_xticks(positions, labels, rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("Net Sharpe ratio", fontsize=10, color=INK)
    _title(
        ax,
        "What each assumption was worth: naive on the left, honest on the right",
        is_synthetic=is_synthetic,
    )
    return _save(fig, path, dpi=dpi)


def plot_cost_attribution(
    table: pd.DataFrame,
    *,
    path: Path,
    dpi: int,
    is_synthetic: bool,
) -> Path:
    """Where the gross return went, as an identity rather than an estimate."""
    fig, ax = _new_axes(figsize=(9.0, 5.0))
    order = ["gross_return", "commission", "spread", "market_impact", "borrow", "net_return"]
    values = table.loc[order, "annualised_contribution"]
    colours = [HONEST if v >= 0 else NAIVE for v in values]

    ax.barh(range(len(values)), values.to_numpy(), color=colours, height=0.62, zorder=3)
    ax.axvline(0.0, color=INK, linewidth=1.0, zorder=4)
    ax.set_yticks(range(len(values)), [label.replace("_", " ") for label in order], fontsize=9.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.set_xlabel("Annualised contribution to return", fontsize=10, color=INK)
    _title(ax, "Gross return, less each cost component, equals net", is_synthetic=is_synthetic)
    return _save(fig, path, dpi=dpi)


def plot_drawdowns(
    naive_drawdown: pd.Series,
    honest_drawdown: pd.Series,
    *,
    path: Path,
    dpi: int,
    is_synthetic: bool,
) -> Path:
    fig, ax = _new_axes(figsize=(9.5, 4.6))
    ax.fill_between(
        honest_drawdown.index, honest_drawdown.to_numpy(), 0.0, color=HONEST, alpha=0.28, zorder=3
    )
    ax.plot(
        naive_drawdown.index,
        naive_drawdown.to_numpy(),
        color=NAIVE,
        linewidth=1.4,
        label="naive",
        zorder=4,
    )
    ax.plot(
        honest_drawdown.index,
        honest_drawdown.to_numpy(),
        color=HONEST,
        linewidth=1.6,
        label="honest",
        zorder=5,
    )
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Drawdown", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _title(ax, "Drawdown under each set of assumptions", is_synthetic=is_synthetic)
    return _save(fig, path, dpi=dpi)

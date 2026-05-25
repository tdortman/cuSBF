#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import plot_utils as pu
import typer

app = typer.Typer(help="Plot Speed of Light (SOL) benchmark results")

SOL_FILTERS = ["cusbf", "cucobloom"]

PLOT_METRICS = [
    ("sm_throughput", "Compute"),
    ("l1_throughput", "L1 Cache"),
    ("l2_throughput", "L2 Cache"),
    ("dram_throughput", "DRAM"),
]

METRIC_MARKERS = {
    "sm_throughput": "o",
    "l1_throughput": "^",
    "l2_throughput": "v",
    "dram_throughput": "D",
}

METRIC_LINESTYLES = {
    "sm_throughput": "-",
    "l1_throughput": "--",
    "l2_throughput": "--",
    "dram_throughput": ":",
}

OPERATION_MARKERS = {
    "insert": "o",
    "query": "s",
}

X_AXIS_LABEL = "Target Filter Capacity [k-mers]"
SOL_MARKER_SIZE = pu.MARKER_SIZE - 3
SOL_LEGEND_MARKER_SIZE = pu.MARKER_SIZE - 2
Y_AXIS_TICKS = [0, 25, 50, 75, 100]


def _filter_legend_handles(filters: list[str]) -> list:
    return [
        plt.Line2D(
            [],
            [],
            color=pu.FILTER_STYLES[f]["color"],
            marker=pu.FILTER_STYLES[f].get("marker", "o"),
            linestyle="-",
            linewidth=pu.LINE_WIDTH,
            markersize=SOL_LEGEND_MARKER_SIZE,
            label=pu.get_filter_display_name(f),
        )
        for f in filters
    ]


def _operation_legend_handles(operations: list[str]) -> list:
    return [
        plt.Line2D(
            [],
            [],
            color="#444444",
            marker=OPERATION_MARKERS[operation],
            linestyle="None",
            markersize=SOL_LEGEND_MARKER_SIZE,
            label=operation.capitalize(),
        )
        for operation in operations
    ]


def _add_comparison_legend(
    fig, axes, filters: list[str], operations: list[str]
) -> None:
    """Place a compact filter/operation legend above the subplot grid."""
    axes_box = axes[0][0].get_position()
    legend_kw = dict(
        fontsize=pu.LEGEND_FONT_SIZE - 2,
        framealpha=pu.LEGEND_FRAME_ALPHA,
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.5,
    )

    legend_handles = _filter_legend_handles(filters) + _operation_legend_handles(
        operations
    )
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, axes_box.y1 + 0.032),
        ncol=len(legend_handles),
        **legend_kw,
    )


def _add_shared_x_label(fig, axes, label: str) -> None:
    """Center a shared x label beneath the bottom subplot span."""
    bottom_y = min(ax.get_position().y0 for ax in axes[-1])
    left_x = min(ax.get_position().x0 for ax in axes[-1])
    right_x = max(ax.get_position().x1 for ax in axes[-1])
    fig.text(
        (left_x + right_x) / 2,
        bottom_y - 0.068,
        label,
        ha="center",
        va="top",
        fontsize=pu.DEFAULT_FONT_SIZE - 1,
        fontweight="bold",
    )


def _add_shared_y_label(fig, axes, label: str) -> None:
    """Center a shared y label beside the left subplot span."""
    bottom_y = min(row[0].get_position().y0 for row in axes)
    top_y = max(row[0].get_position().y1 for row in axes)
    left_x = min(row[0].get_position().x0 for row in axes)
    fig.text(
        left_x - 0.07,
        (bottom_y + top_y) / 2,
        label,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=pu.DEFAULT_FONT_SIZE - 1,
        fontweight="bold",
    )


def _filter_sol_df(df):
    """Keep only cuSBF and cuco rows present in the CSV."""
    present = [f for f in SOL_FILTERS if f in df["filter"].unique()]
    if not present:
        typer.secho(
            f"No rows for {', '.join(SOL_FILTERS)} in the CSV.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    return df[df["filter"].isin(present)], present


def _plot_unified(df, output_dir: Path) -> None:
    """IEEE-friendly 2x2 resource panels with both filters and operations overlaid."""
    operations = [op for op in ["insert", "query"] if op in set(df["operation"])]
    filters = [f for f in SOL_FILTERS if f in set(df["filter"])]
    if not operations:
        typer.secho(
            "No insert/query rows found for cuSBF/cuco.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True, sharey=True)

    for ax, (metric_col, metric_name) in zip(axes.flat, PLOT_METRICS):
        for filter_type in filters:
            filter_subset = df[df["filter"] == filter_type]
            style = pu.FILTER_STYLES.get(filter_type, {})

            for operation in operations:
                series = filter_subset[
                    filter_subset["operation"] == operation
                ].sort_values("capacity")
                if series.empty or metric_col not in series.columns:
                    continue

                ax.plot(
                    series["capacity"].values,
                    series[metric_col].values,
                    label="_nolegend_",
                    color=style.get("color", "#333333"),
                    marker=OPERATION_MARKERS[operation],
                    linestyle="-",
                    linewidth=pu.LINE_WIDTH,
                    markersize=SOL_MARKER_SIZE,
                )

        ax.set_title(metric_name, fontweight="bold", pad=4)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 105)
        ax.set_yticks(Y_AXIS_TICKS)
        ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)

    for ax in axes[0]:
        ax.tick_params(axis="x", labelbottom=True)

    plt.tight_layout(rect=(0.08, 0.04, 1, 0.88))
    _add_shared_y_label(fig, axes, "Throughput (% of Peak)")
    _add_shared_x_label(fig, axes, X_AXIS_LABEL)
    _add_comparison_legend(fig, axes, filters, operations)

    output_file = output_dir / "sol_benchmark.pdf"
    plt.savefig(
        output_file, bbox_inches="tight", transparent=True, format="pdf", dpi=600
    )
    typer.secho(f"Saved {output_file}", fg=typer.colors.GREEN)
    plt.close()


def _plot_per_metric_comparison(df, output_dir: Path) -> None:
    """One plot per metric comparing cuSBF and cuco (insert/query separate)."""
    for metric_col, metric_name in PLOT_METRICS:
        if metric_col not in df.columns:
            continue

        for operation in sorted(df["operation"].unique()):
            op_subset = df[df["operation"] == operation]
            if op_subset.empty:
                continue

            _, ax = plt.subplots(figsize=(12, 7))

            for filter_type in sorted(op_subset["filter"].unique()):
                filter_subset = op_subset[
                    op_subset["filter"] == filter_type
                ].sort_values("capacity")
                if filter_subset.empty:
                    continue

                style = pu.FILTER_STYLES.get(filter_type, {})
                ax.plot(
                    filter_subset["capacity"].values,
                    filter_subset[metric_col].values,
                    label=pu.get_filter_display_name(filter_type),
                    linewidth=pu.LINE_WIDTH,
                    markersize=pu.MARKER_SIZE,
                    color=style.get("color"),
                    marker=METRIC_MARKERS.get(metric_col, style.get("marker", "o")),
                    linestyle="-",
                )

            ax.set_xlabel(
                X_AXIS_LABEL,
                fontsize=pu.AXIS_LABEL_FONT_SIZE,
                fontweight="bold",
            )
            ax.set_ylabel(
                f"{metric_name} Throughput (% of Peak)",
                fontsize=pu.AXIS_LABEL_FONT_SIZE,
                fontweight="bold",
            )
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 105)
            ax.set_yticks(Y_AXIS_TICKS)
            ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
            ax.legend(
                fontsize=pu.LEGEND_FONT_SIZE,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=2,
                framealpha=pu.LEGEND_FRAME_ALPHA,
            )

            plt.tight_layout(rect=(0, 0, 1, 0.92))

            output_file = output_dir / f"sol_compare_{metric_col}_{operation}.pdf"
            plt.savefig(
                output_file,
                bbox_inches="tight",
                transparent=True,
                format="pdf",
                dpi=600,
            )
            typer.secho(f"Saved {output_file}", fg=typer.colors.GREEN)
            plt.close()


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to CSV file with SOL benchmark results",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
    per_metric: bool = typer.Option(
        False,
        "--per-metric",
        help="Also emit per-metric filter comparison plots.",
    ),
):
    """
    Generate Speed of Light (SOL) throughput plots from benchmark CSV results.

    Plots cuSBF and cuco only. The main figure compares the two filters directly
    within each resource panel (Compute, L1, L2, DRAM): filter is encoded by
    color and operation by marker shape. Memory is collected in the CSV but not
    plotted.
    """
    df = pu.load_csv(csv_file)
    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    df, _ = _filter_sol_df(df)

    _plot_unified(df, output_dir)

    if per_metric:
        _plot_per_metric_comparison(df, output_dir)


if __name__ == "__main__":
    app()

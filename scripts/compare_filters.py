#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Clustered-bar throughput plots for gpu-filter-comparison FASTX benchmarks."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.axes import Axes
from matplotlib.patches import Patch

app = typer.Typer(help="Plot gpu-filter-comparison throughput as clustered bars")

# Left-to-right bar order; short legend labels.
THROUGHPUT_GROUP_ORDER: list[str] = [
    "cuckoogpu",
    "cucobloom",
    "gqf",
    "tcf",
    "cusbf",
    "superbloom_cpu",
]

THROUGHPUT_LEGEND_LABELS: dict[str, str] = {
    "cuckoogpu": "Cuckoo-GPU",
    "cucobloom": "GPU Blocked Bloom",
    "gqf": "GQF",
    "tcf": "TCF",
    "cusbf": "cuSBF",
    "superbloom_cpu": "Super Bloom",
}

OPERATION_LABELS = {
    "Insert": "Insert",
    "Query": "Query",
}

_BAR_WIDTH = 0.12
_BAR_VALUE_FONT_SIZE = 8
_GROUP_STRIDE = _BAR_WIDTH * 1.05
_CATEGORY_GAP = 0.15
_X_MARGIN = 0.1
_SUBPLOT_FIGSIZE = (7.2, 4.25)
# Log-scale headroom so bar value labels above the tallest bar are not clipped.
_YLIM_TOP_FACTOR = 1.45
_OUTPUT_BASENAME = "benchmark_throughput_comparison"


def normalize_filter_key(fixture_base: str) -> str:
    """Map fixture base names to canonical filter keys."""
    key = fixture_base.lower()
    if key.startswith("cuckoogpu"):
        return "cuckoogpu"
    if key.startswith("cucobloom"):
        return "cucobloom"
    if key == "gqf" or key.startswith("gqffixture"):
        return "gqf"
    if key == "tcf" or key.startswith("tcffixture"):
        return "tcf"
    if key.startswith("cusbf"):
        return "cusbf"
    if key.startswith("superbloom"):
        return "superbloom_cpu"
    return key


def sort_groups(
    throughput_data: dict[str, dict[str, float]],
    preferred_order: list[str],
) -> list[str]:
    ordered = [g for g in preferred_order if g in throughput_data]
    extras = sorted(
        set(throughput_data.keys()) - set(ordered),
        key=lambda g: -max(throughput_data[g].values(), default=0.0),
    )
    return ordered + extras


def workload_scale(row: pd.Series) -> float | None:
    """Return a scalar workload size for ordering small vs large inputs."""
    for column in ("memory_bytes", "filter_bits", "sequence_bases"):
        value = row.get(column)
        if pd.notna(value):
            return float(value)
    return None


def load_throughput_data(
    csv_path: Path,
) -> tuple[dict[str, dict[str, float]], float | None]:
    """Load Insert/Query throughput [GKmer/s] from a gpu-filter-comparison CSV."""
    df = pu.load_csv(csv_path)
    df = df[df["name"].str.endswith("_median", na=False)]

    throughput_data: dict[str, dict[str, float]] = defaultdict(dict)
    scale: float | None = None

    for _, row in df.iterrows():
        parsed = pu.parse_fixture_benchmark_name(row["name"])
        if parsed is None:
            continue

        fixture_base, operation, _size = parsed
        if operation not in OPERATION_LABELS:
            continue

        items_per_second = row.get("items_per_second")
        if pd.isna(items_per_second):
            continue

        row_scale = workload_scale(row)
        if row_scale is not None:
            scale = row_scale if scale is None else max(scale, row_scale)

        filter_key = normalize_filter_key(fixture_base)
        op_label = OPERATION_LABELS[operation]
        throughput_data[filter_key][op_label] = pu.to_gkmers_per_sec(items_per_second)

    return dict(throughput_data), scale


def expand_throughput_ylim(ylim: tuple[float, float]) -> tuple[float, float]:
    """Add log-scale headroom above the tallest bar for value labels."""
    ymin, ymax = ylim
    if ymax <= 0:
        return ylim
    return ymin, ymax * _YLIM_TOP_FACTOR


def relabel_chart_data(
    throughput_data: dict[str, dict[str, float]],
    groups: list[str],
) -> dict[str, dict[str, float]]:
    categories = [OPERATION_LABELS["Insert"], OPERATION_LABELS["Query"]]
    return {
        group: {cat: throughput_data.get(group, {}).get(cat, 0.0) for cat in categories}
        for group in groups
    }


def plot_throughput_on_axis(
    ax: Axes,
    chart_data: dict[str, dict[str, float]],
    groups: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    show_ylabel: bool,
) -> list[Patch]:
    """Plot one workload on *ax* and return filter legend patches."""
    categories = [OPERATION_LABELS["Insert"], OPERATION_LABELS["Query"]]
    n_groups = len(groups)
    cluster_span = max(n_groups - 1, 0) * _GROUP_STRIDE + _BAR_WIDTH
    category_stride = cluster_span + _CATEGORY_GAP

    pu.clustered_bar_chart(
        ax,
        categories=categories,
        groups=groups,
        data=chart_data,
        colors=colors,
        labels=labels,
        bar_width=_BAR_WIDTH,
        group_stride=_GROUP_STRIDE,
        category_stride=category_stride,
        show_values=True,
        value_decimals=2,
        value_fontsize=_BAR_VALUE_FONT_SIZE,
    )

    pu.format_axis(
        ax,
        "",
        pu.THROUGHPUT_LABEL if show_ylabel else "",
        xscale=None,
        yscale="log",
    )
    if show_ylabel:
        ax.set_ylabel(
            pu.THROUGHPUT_LABEL,
            fontsize=pu.AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
            labelpad=10,
        )
    else:
        ax.set_ylabel("")
    half_span = ((n_groups - 1) / 2) * _GROUP_STRIDE + (_BAR_WIDTH / 2)
    ax.set_xlim(-half_span - _X_MARGIN, category_stride + half_span + _X_MARGIN)
    ax.tick_params(axis="x", labelsize=pu.TICK_LABEL_FONT_SIZE)

    return [
        Patch(
            facecolor=colors.get(group, "#333333"),
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label=labels.get(group, group),
        )
        for group in groups
    ]


def save_throughput_subplot(
    output_path: Path,
    chart_data: dict[str, dict[str, float]],
    groups: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    show_ylabel: bool,
    ylim: tuple[float, float],
    message: str,
) -> list[Patch]:
    """Render and save one independent throughput subplot PDF."""
    fig, ax = plt.subplots(1, 1, figsize=_SUBPLOT_FIGSIZE)
    legend_patches = plot_throughput_on_axis(
        ax,
        chart_data,
        groups,
        colors,
        labels,
        show_ylabel=show_ylabel,
    )

    ax.set_ylim(*expand_throughput_ylim(ylim))
    left_margin = 0.16 if show_ylabel else 0.07
    fig.subplots_adjust(left=left_margin, right=0.99, bottom=0.14, top=0.94)
    fig.savefig(
        output_path,
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(message, fg=typer.colors.GREEN)
    plt.close(fig)
    return legend_patches


def save_throughput_legend_figure(
    output_path: Path,
    filter_handles: list[Patch],
    fig_width: float,
    message: str,
) -> None:
    """Render and save a legend-only figure for the split throughput charts."""
    legend_fig_height = 0.55
    fig = plt.figure(figsize=(fig_width, legend_fig_height))
    fig.legend(
        handles=filter_handles,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=len(filter_handles),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.5,
    )
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(message, fg=typer.colors.GREEN)
    plt.close(fig)


@app.command()
def main(
    csv_file_small: Path = typer.Argument(
        ...,
        help="Throughput CSV for the smaller FASTX workload (e.g. C. elegans)",
    ),
    csv_file_large: Path = typer.Argument(
        ...,
        help="Throughput CSV for the larger FASTX workload (e.g. human genome)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Plot Insert/Query throughput [GKmer/s] from two gpu-filter-comparison CSVs.

    Writes three PDFs for LaTeX composition (see external/cuckoo-gpu-paper/main.tex):
    benchmark_throughput_comparison_legend.pdf,
    benchmark_throughput_comparison_left.pdf (smaller workload),
    benchmark_throughput_comparison_right.pdf (larger workload).
    """
    small_data, scale_small = load_throughput_data(csv_file_small)
    large_data, scale_large = load_throughput_data(csv_file_large)

    if not small_data or not large_data:
        typer.secho(
            "No throughput data found in one or both CSV files",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    if scale_small is not None and scale_large is not None and scale_small != scale_large:
        if scale_small > scale_large:
            small_data, large_data = large_data, small_data
    else:
        typer.secho(
            "Warning: Could not infer unique small/large workloads from CSV counters; "
            "using first CSV as left (small) and second CSV as right (large).",
            fg=typer.colors.YELLOW,
            err=True,
        )

    all_groups = set(small_data.keys()) | set(large_data.keys())
    groups = sort_groups(
        {g: {**small_data.get(g, {}), **large_data.get(g, {})} for g in all_groups},
        THROUGHPUT_GROUP_ORDER,
    )

    colors = {g: pu.FILTER_COLORS.get(g, "#555555") for g in groups}
    labels = {
        g: THROUGHPUT_LEGEND_LABELS.get(g, pu.get_filter_display_name(g)) for g in groups
    }
    small_chart = relabel_chart_data(small_data, groups)
    large_chart = relabel_chart_data(large_data, groups)

    fig_width = max(8.0, 1.2 * len(groups) + 1.5)

    preview_fig, (preview_left, preview_right) = plt.subplots(
        1,
        2,
        figsize=(fig_width, _SUBPLOT_FIGSIZE[1]),
        sharey=True,
        gridspec_kw={"wspace": 0.04},
    )
    legend_left = plot_throughput_on_axis(
        preview_left,
        small_chart,
        groups,
        colors,
        labels,
        show_ylabel=True,
    )
    legend_right = plot_throughput_on_axis(
        preview_right,
        large_chart,
        groups,
        colors,
        labels,
        show_ylabel=False,
    )
    y_min_left, y_max_left = preview_left.get_ylim()
    y_min_right, y_max_right = preview_right.get_ylim()
    common_ylim = expand_throughput_ylim(
        (min(y_min_left, y_min_right), max(y_max_left, y_max_right))
    )
    plt.close(preview_fig)

    filter_handles = legend_left if len(legend_left) >= len(legend_right) else legend_right

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    left_output = output_dir / f"{_OUTPUT_BASENAME}_left.pdf"
    right_output = output_dir / f"{_OUTPUT_BASENAME}_right.pdf"
    legend_output = output_dir / f"{_OUTPUT_BASENAME}_legend.pdf"

    save_throughput_subplot(
        left_output,
        small_chart,
        groups,
        colors,
        labels,
        show_ylabel=True,
        ylim=common_ylim,
        message=f"Left throughput subplot saved to {left_output}",
    )
    save_throughput_subplot(
        right_output,
        large_chart,
        groups,
        colors,
        labels,
        show_ylabel=False,
        ylim=common_ylim,
        message=f"Right throughput subplot saved to {right_output}",
    )
    save_throughput_legend_figure(
        legend_output,
        filter_handles,
        fig_width=fig_width,
        message=f"Throughput legend saved to {legend_output}",
    )


if __name__ == "__main__":
    app()

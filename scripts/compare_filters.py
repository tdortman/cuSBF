#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Clustered-bar throughput plot for gpu-filter-comparison FASTX benchmarks."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

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
    if key.startswith("superbloomcpufastx") or key.startswith("superbloomcpu"):
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


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to CSV file, or '-' to read from stdin",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Plot Insert/Query throughput [GKmer/s] from gpu-filter-comparison CSV.

    Expects a single FASTX workload (one benchmark point per filter/operation).
    """
    df = pu.load_csv(csv_file)
    df = df[df["name"].str.endswith("_median", na=False)]

    throughput_data: dict[str, dict[str, float]] = defaultdict(dict)

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

        filter_key = normalize_filter_key(fixture_base)
        op_label = OPERATION_LABELS[operation]
        throughput_data[filter_key][op_label] = pu.to_gkmers_per_sec(items_per_second)

    if not throughput_data:
        typer.secho("No throughput data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    groups = sort_groups(throughput_data, THROUGHPUT_GROUP_ORDER)
    categories = [OPERATION_LABELS["Insert"], OPERATION_LABELS["Query"]]
    chart_data = {
        g: {cat: throughput_data[g].get(cat, 0.0) for cat in categories} for g in groups
    }

    colors = {g: pu.FILTER_COLORS.get(g, "#555555") for g in groups}
    labels = {
        g: THROUGHPUT_LEGEND_LABELS.get(g, pu.get_filter_display_name(g)) for g in groups
    }

    n_groups = len(groups)
    bar_width = min(0.12, 0.85 / max(n_groups, 1))
    group_stride = bar_width * 1.05

    fig, ax = pu.setup_figure(figsize=(14, 7))
    # Tight spacing between Insert and Query clusters (default category_stride=1 is too wide).
    cluster_span = max(n_groups - 1, 0) * group_stride + bar_width
    category_stride = cluster_span + 0.15

    pu.clustered_bar_chart(
        ax,
        categories=categories,
        groups=groups,
        data=chart_data,
        colors=colors,
        labels=labels,
        bar_width=bar_width,
        group_stride=group_stride,
        category_stride=category_stride,
        show_values=True,
        value_decimals=2,
    )

    pu.format_axis(
        ax,
        "",
        pu.THROUGHPUT_LABEL,
        xscale=None,
        yscale="log",
    )
    ax.set_xlim(-cluster_span / 2 - 0.1, category_stride + cluster_span / 2 + 0.1)
    ax.tick_params(axis="x", labelsize=pu.TICK_LABEL_FONT_SIZE)

    plt.tight_layout(rect=(0, 0.02, 1, 0.90))
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        fontsize=20,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(groups),
        framealpha=pu.LEGEND_FRAME_ALPHA_SOLID,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.6,
    )

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    output_file = output_dir / "benchmark_throughput_comparison.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(
        f"Throughput comparison plot saved to {output_file}",
        fg=typer.colors.GREEN,
    )
    plt.close()


if __name__ == "__main__":
    app()

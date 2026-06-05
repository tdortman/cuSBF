#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Plot byte-sequence vs dense-packed cuSBF throughput (device and host input)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.patches import Patch

app = typer.Typer(help="Plot byte vs dense-packed cuSBF throughput from benchmark CSV")

_OPERATIONS = ["Insert", "Query"]
_ENCODINGS = ["Byte sequence", "Dense packed"]
_RESIDENCIES = ["device", "host"]
_ENCODING_COLORS = {
    "Byte sequence": pu.FILTER_COLORS["cusbf"],
    "Dense packed": "#F18F01",
}
_OPERATION_MARKERS = {
    "Insert": "o",
    "Query": "s",
}
_RESIDENCY_LABELS = {
    "device": "Device-resident input",
    "host": "Host-resident input",
}
_LEGEND_MARKER_SIZE = pu.MARKER_SIZE - 2
_PLOT_MARKER_SIZE = pu.MARKER_SIZE
_LEGEND_ROW_STEP = 0.08
_YLABEL_LEFT_MARGIN = 0.09
_X_AXIS_LABEL = "Sequence length (symbols)"


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    lines = csv_path.read_text().splitlines()
    header_idx = next(
        (idx for idx, line in enumerate(lines) if line.startswith("name,")),
        0,
    )
    data = io.StringIO("\n".join(lines[header_idx:]))
    header_df = pd.read_csv(data, nrows=0)
    data.seek(0)
    return pd.read_csv(data, usecols=list(header_df.columns))


def parse_benchmark_row(name: str) -> tuple[str, str, str, int] | None:
    parsed = pu.parse_fixture_benchmark_name(name)
    if parsed is None:
        return None
    fixture_base, benchmark_name, size = parsed
    if fixture_base.startswith("densepackeddevicethroughput"):
        residency = "device"
    elif fixture_base.startswith("densepackedhostthroughput"):
        residency = "host"
    elif fixture_base.startswith("densepackedthroughput"):
        residency = "device"
    else:
        return None

    if benchmark_name.endswith("Insert"):
        operation = "Insert"
        encoding = (
            "Byte sequence" if benchmark_name.startswith("Byte") else "Dense packed"
        )
    elif benchmark_name.endswith("Query"):
        operation = "Query"
        encoding = (
            "Byte sequence" if benchmark_name.startswith("Byte") else "Dense packed"
        )
    else:
        return None
    return residency, operation, encoding, size


def load_throughput_series(csv_path: Path) -> pd.DataFrame:
    df = load_benchmark_csv(csv_path)
    df = df[df["name"].str.endswith("_median", na=False)]

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        parsed = parse_benchmark_row(str(row["name"]))
        if parsed is None:
            continue
        residency, operation, encoding, size = parsed
        items_per_second = row.get("items_per_second")
        if pd.isna(items_per_second):
            continue

        rows.append(
            {
                "residency": residency,
                "operation": operation,
                "encoding": encoding,
                "num_symbols": size,
                "throughput_gkmers_per_sec": pu.to_gkmers_per_sec(
                    float(items_per_second)
                ),
            }
        )

    if not rows:
        raise typer.BadParameter(f"No dense-packed throughput rows found in {csv_path}")

    return pd.DataFrame(rows)


def _encoding_legend_handles(encodings: list[str]) -> list:
    return [
        Patch(
            facecolor=_ENCODING_COLORS[encoding],
            edgecolor="black",
            linewidth=pu.BAR_EDGE_WIDTH,
            label=encoding,
        )
        for encoding in encodings
    ]


def _operation_legend_handles(operations: list[str]) -> list:
    return [
        plt.Line2D(
            [],
            [],
            color="#444444",
            marker=_OPERATION_MARKERS[operation],
            linestyle="None",
            markersize=_LEGEND_MARKER_SIZE,
            label=operation,
        )
        for operation in operations
    ]


def _add_residency_label(fig: plt.Figure, ax: plt.Axes, residency: str) -> None:
    legend_right_x = ax.get_position().x1
    operation_y = ax.get_position().y1 + 0.045
    residency_y = operation_y + _LEGEND_ROW_STEP
    fig.text(
        legend_right_x,
        residency_y,
        _RESIDENCY_LABELS[residency],
        ha="right",
        va="bottom",
        fontsize=pu.LEGEND_FONT_SIZE - 2,
        fontweight="bold",
    )


def _add_comparison_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    encodings: list[str],
    operations: list[str],
    residency: str,
) -> None:
    """Place encoding legend left; residency label and operation markers right."""
    legend_y = ax.get_position().y1 + 0.045
    legend_left_x = ax.get_position().x0
    legend_right_x = ax.get_position().x1
    legend_kw = dict(
        fontsize=pu.LEGEND_FONT_SIZE - 2,
        framealpha=pu.LEGEND_FRAME_ALPHA,
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.5,
    )

    encoding_handles = _encoding_legend_handles(encodings)
    if encoding_handles:
        fig.legend(
            encoding_handles,
            [h.get_label() for h in encoding_handles],
            loc="lower left",
            bbox_to_anchor=(legend_left_x, legend_y),
            ncol=1,
            **legend_kw,
        )

    _add_residency_label(fig, ax, residency)

    operation_handles = _operation_legend_handles(operations)
    if operation_handles:
        fig.legend(
            operation_handles,
            [h.get_label() for h in operation_handles],
            loc="lower right",
            bbox_to_anchor=(legend_right_x, legend_y),
            ncol=len(operation_handles),
            **legend_kw,
        )


def _add_shared_x_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    bottom_y = ax.get_position().y0
    left_x = ax.get_position().x0
    right_x = ax.get_position().x1
    fig.text(
        (left_x + right_x) / 2,
        bottom_y - 0.068,
        label,
        ha="center",
        va="top",
        fontsize=pu.DEFAULT_FONT_SIZE - 1,
        fontweight="bold",
    )


def _add_shared_y_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    bottom_y = ax.get_position().y0
    top_y = ax.get_position().y1
    left_x = ax.get_position().x0
    fig.text(
        left_x - _YLABEL_LEFT_MARGIN,
        (bottom_y + top_y) / 2,
        label,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=pu.DEFAULT_FONT_SIZE - 1,
        fontweight="bold",
    )


def plot_throughput(data: pd.DataFrame, residency: str, output_pdf: Path) -> None:
    subset = data[data["residency"] == residency]
    if subset.empty:
        typer.secho(
            f"No {residency} rows in CSV; skipping {output_pdf.name}",
            fg=typer.colors.YELLOW,
        )
        return

    operations = [op for op in _OPERATIONS if op in set(subset["operation"])]
    encodings = [enc for enc in _ENCODINGS if enc in set(subset["encoding"])]
    if not operations or not encodings:
        typer.secho(
            f"Incomplete {residency} rows in CSV; skipping {output_pdf.name}",
            fg=typer.colors.YELLOW,
        )
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))

    for encoding in encodings:
        encoding_subset = subset[subset["encoding"] == encoding]
        for operation in operations:
            series = encoding_subset[
                encoding_subset["operation"] == operation
            ].sort_values("num_symbols")
            if series.empty:
                continue
            ax.plot(
                series["num_symbols"],
                series["throughput_gkmers_per_sec"],
                label="_nolegend_",
                color=_ENCODING_COLORS[encoding],
                marker=_OPERATION_MARKERS[operation],
                linestyle="-",
                linewidth=pu.LINE_WIDTH,
                markersize=_PLOT_MARKER_SIZE,
            )

    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)

    plt.tight_layout(rect=(0.08, 0.04, 1, 0.80))
    _add_shared_y_label(fig, ax, pu.THROUGHPUT_LABEL)
    _add_shared_x_label(fig, ax, _X_AXIS_LABEL)
    _add_comparison_legend(fig, ax, encodings, operations, residency)

    fig.savefig(
        output_pdf, bbox_inches="tight", transparent=True, format="pdf", dpi=600
    )
    plt.close(fig)
    typer.secho(
        f"Dense-packed throughput figure saved to {output_pdf}", fg=typer.colors.GREEN
    )


@app.command()
def main(
    csv_path: Path = typer.Argument(..., help="dense-packed-throughput benchmark CSV"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """Plot device- and host-resident byte vs dense-packed throughput."""
    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))
    data = load_throughput_series(csv_path)

    for residency in _RESIDENCIES:
        plot_throughput(
            data,
            residency,
            output_dir / f"dense_packed_throughput_{residency}.pdf",
        )


if __name__ == "__main__":
    app()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Plot cuSBF host-sequence transfer overhead vs device-resident kernels"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

app = typer.Typer(help="Plot cuSBF host-sequence throughput from benchmark CSVs")

_OPERATIONS = ["Insert", "Query"]
_OUTPUT_BASENAME = "host_overhead"
_SUBPLOT_FIGSIZE = (4.5, 2.5)
_SUBPLOT_LEFT_MARGIN = 0.17
_SUBPLOT_RIGHT_MARGIN = 0.99
_SUBPLOT_BOTTOM_MARGIN = 0.05
_SUBPLOT_TOP_MARGIN = 0.76
_FIGURE_PAD_INCHES = 0.04
_YLIM_TOP_FACTOR = 1.45

_YLABEL_FONT_SIZE = 10
_YLABEL_COORD_X = -0.12
_LEGEND_FONT_SIZE = 10

_LEGEND_ABOVE_AXES_Y = 1.1
_LEGEND_ROW_STEP = 0.11
_DEVICE_RESIDENT_LABEL = "Device-resident"

_TICK_LABEL_FONT_SIZE = 9

_HBM3_LABEL = "GH200"
_GDDR7_LABEL = "RTX PRO 6000"
_HBM3_COLOR = pu.FILTER_COLORS["cusbf"]
_GDDR7_COLOR = "#F18F01"
_KERNEL_COLOR = _HBM3_COLOR
_OVERHEAD_COLOR = "#D1D5DB"
_OVERHEAD_HATCH = "//"
_BAR_WIDTH = 0.22
_GROUP_SPACING = 1.0
_OP_STRIDE = 0.28
_PLATFORM_OFFSET = 0.52
_ANNOTATION_FONT_SIZE = 7
_PLATFORM_SHORT_LABELS = {
    _GDDR7_LABEL: "System A",
    _HBM3_LABEL: "System B",
}
_PLATFORM_LEGEND_LABELS = {
    _GDDR7_LABEL: _GDDR7_LABEL,
    _HBM3_LABEL: _HBM3_LABEL,
}
_PLATFORM_ORDER = [_GDDR7_LABEL, _HBM3_LABEL]
_BAR_OPERATIONS = [
    ("Insert", "//"),
    ("Query", None),
]


@dataclass(frozen=True)
class PipelineThroughput:
    host: dict[str, float]
    device: dict[str, float]


def parse_pipeline_mode(row: pd.Series, fixture_base: str) -> Optional[str]:
    """Return ``host`` or ``device`` from fixture name or pipeline_mode counter."""
    base = fixture_base.lower()
    if "host" in base:
        return "host"
    if "device" in base:
        return "device"

    pipeline_mode = row.get("pipeline_mode")
    if pd.notna(pipeline_mode):
        if float(pipeline_mode) < 0.5:
            return "host"
        return "device"
    return None


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    """Load a Google Benchmark CSV, skipping optional console metadata preamble."""
    lines = csv_path.read_text().splitlines()
    header_idx = next(
        (idx for idx, line in enumerate(lines) if line.startswith("name,")),
        0,
    )
    data = io.StringIO("\n".join(lines[header_idx:]))
    header_df = pd.read_csv(data, nrows=0)
    data.seek(0)
    return pd.read_csv(data, usecols=list(header_df.columns))


def load_pipeline_throughput(csv_path: Path) -> PipelineThroughput:
    """Load host/device Insert/Query throughput [GKmer/s] from a benchmark CSV."""
    df = load_benchmark_csv(csv_path)
    df = df[df["name"].str.endswith("_median", na=False)]

    host: dict[str, float] = {}
    device: dict[str, float] = {}

    for _, row in df.iterrows():
        parsed = pu.parse_fixture_benchmark_name(row["name"])
        if parsed is None:
            continue

        fixture_base, operation, _size = parsed
        if operation not in _OPERATIONS:
            continue

        items_per_second = row.get("items_per_second")
        if pd.isna(items_per_second):
            continue

        mode = parse_pipeline_mode(row, fixture_base)
        if mode is None:
            continue

        throughput = pu.to_gkmers_per_sec(float(items_per_second))
        if mode == "host":
            host[operation] = throughput
        else:
            device[operation] = throughput

    return PipelineThroughput(host=host, device=device)


def apply_throughput_ylabel(ax: Axes) -> None:
    """Bold throughput y-label sized to fit this short panel without clipping."""
    ax.set_ylabel(
        pu.THROUGHPUT_LABEL,
        fontsize=_YLABEL_FONT_SIZE,
        fontweight="bold",
        labelpad=3,
    )
    ax.yaxis.set_label_coords(_YLABEL_COORD_X, 0.5)


def compute_overhead(host_tp: float, device_tp: float) -> tuple[float, float]:
    """Return (kernel_fraction, overhead_pct) from host/device throughput."""
    if device_tp <= 0.0 or host_tp <= 0.0:
        return 0.0, 0.0
    kernel_fraction = min(1.0, host_tp / device_tp)
    overhead_pct = max(0.0, (1.0 - kernel_fraction) * 100.0)
    return kernel_fraction, overhead_pct


def plot_overhead_panel(
    ax: Axes,
    hbm3: PipelineThroughput,
    gddr7: PipelineThroughput,
    show_ylabel: bool,
    ylim: Optional[tuple[float, float]] = None,
) -> list[Artist]:
    """Draw host-sequence throughput bars for one workload subplot."""
    legend_handles: list[Artist] = []
    data_by_platform = {
        _GDDR7_LABEL: gddr7,
        _HBM3_LABEL: hbm3,
    }
    platforms = [(platform, data_by_platform[platform]) for platform in _PLATFORM_ORDER]
    min_positive = float("inf")
    max_throughput = 0.0

    for platform_idx, (platform, data) in enumerate(platforms):
        platform_center = platform_idx * _GROUP_SPACING
        platform_color = _HBM3_COLOR if platform == _HBM3_LABEL else _GDDR7_COLOR
        for op_idx, (operation, hatch) in enumerate(_BAR_OPERATIONS):
            host_tp = data.host.get(operation, 0.0)
            device_tp = data.device.get(operation, 0.0)
            _, overhead_pct = compute_overhead(host_tp, device_tp)
            if host_tp > 0.0:
                min_positive = min(min_positive, host_tp)
            if device_tp > 0.0:
                min_positive = min(min_positive, device_tp)
            max_throughput = max(max_throughput, host_tp, device_tp)
            x = platform_center + (op_idx - (len(_BAR_OPERATIONS) - 1) / 2) * _OP_STRIDE

            ax.bar(
                x,
                host_tp,
                _BAR_WIDTH,
                color=platform_color,
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                hatch=hatch,
                zorder=3,
            )
            if device_tp > 0.0:
                ax.hlines(
                    device_tp,
                    x - _BAR_WIDTH / 2,
                    x + _BAR_WIDTH / 2,
                    colors="black",
                    linewidth=pu.REFERENCE_LINE_WIDTH,
                    zorder=4,
                )
            if host_tp > 0.0 and device_tp > 0.0:
                label_y = (
                    host_tp * 1.10 if device_tp / host_tp > 1.35 else device_tp * 1.08
                )
                ax.text(
                    x,
                    label_y,
                    f"{overhead_pct:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=_ANNOTATION_FONT_SIZE,
                    clip_on=False,
                )

    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    if ylim is not None:
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
    elif max_throughput > 0.0 and min_positive < float("inf"):
        ax.set_yscale("log")
        ax.set_ylim(max(0.1, min_positive * 0.55), max_throughput * _YLIM_TOP_FACTOR)
    else:
        ax.set_ylim(0.1, 1.0)
    pair_half_span = ((len(_BAR_OPERATIONS) - 1) / 2) * _OP_STRIDE + (_BAR_WIDTH / 2)
    ax.set_xlim(
        -pair_half_span - 0.18,
        (len(platforms) - 1) * _GROUP_SPACING + pair_half_span + 0.18,
    )

    ax.tick_params(axis="y", labelsize=_TICK_LABEL_FONT_SIZE)
    if show_ylabel:
        apply_throughput_ylabel(ax)
    ax.grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)

    platform_colors = {
        _GDDR7_LABEL: _GDDR7_COLOR,
        _HBM3_LABEL: _HBM3_COLOR,
    }
    for platform in _PLATFORM_ORDER:
        legend_handles.append(
            Patch(
                facecolor=platform_colors[platform],
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                label=_PLATFORM_LEGEND_LABELS[platform],
            )
        )
    for operation, hatch in _BAR_OPERATIONS:
        legend_handles.append(
            Patch(
                facecolor="gray",
                edgecolor="black",
                linewidth=pu.BAR_EDGE_WIDTH,
                hatch=hatch,
                label=operation,
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=pu.REFERENCE_LINE_WIDTH,
            label=_DEVICE_RESIDENT_LABEL,
        )
    )
    return legend_handles


def split_overhead_legend_handles(
    legend_handles: list[Artist],
) -> tuple[list[Artist], list[Artist], list[Artist]]:
    return legend_handles[:2], legend_handles[2:4], legend_handles[4:]


def add_overhead_figure_legend(
    ax: Axes,
    platform_handles: list[Artist],
    operation_handles: list[Artist],
    marker_handles: list[Artist],
) -> None:
    """2x2 legend grid: platforms left; Insert/Query and device-resident marker right."""
    legend_kwargs = {
        "fontsize": _LEGEND_FONT_SIZE,
        "frameon": False,
        "handlelength": 1.2,
        "handletextpad": 0.4,
    }
    top_y = _LEGEND_ABOVE_AXES_Y
    bottom_y = top_y - _LEGEND_ROW_STEP

    if len(platform_handles) >= 1:
        top_platform_legend = ax.legend(
            handles=[platform_handles[0]],
            loc="lower left",
            bbox_to_anchor=(0.0, top_y),
            ncol=1,
            **legend_kwargs,
        )
        ax.add_artist(top_platform_legend)

    if operation_handles:
        operation_legend = ax.legend(
            handles=operation_handles,
            loc="lower right",
            bbox_to_anchor=(1.0, top_y),
            ncol=2,
            **legend_kwargs,
        )
        ax.add_artist(operation_legend)

    if len(platform_handles) >= 2:
        bottom_platform_legend = ax.legend(
            handles=[platform_handles[1]],
            loc="lower left",
            bbox_to_anchor=(0.0, bottom_y),
            ncol=1,
            **legend_kwargs,
        )
        ax.add_artist(bottom_platform_legend)

    if marker_handles:
        ax.legend(
            handles=marker_handles,
            loc="lower right",
            bbox_to_anchor=(1.0, bottom_y),
            ncol=1,
            **legend_kwargs,
        )


def plot_absolute_panel(
    ax: Axes,
    hbm3: PipelineThroughput,
    gddr7: PipelineThroughput,
    show_ylabel: bool,
) -> None:
    """Grouped throughput bars (host vs device) for optional absolute comparison."""
    modes = [("Host sequence", "host"), ("Device sequence", "device")]
    n_modes = len(modes)
    bar_width = 0.11
    for op_idx, operation in enumerate(_OPERATIONS):
        group_center = op_idx * _GROUP_SPACING
        data_by_platform = {
            _GDDR7_LABEL: gddr7,
            _HBM3_LABEL: hbm3,
        }
        for plat_idx, platform in enumerate(_PLATFORM_ORDER):
            data = data_by_platform[platform]
            plat_center = group_center + (plat_idx - 0.5) * _PLATFORM_OFFSET
            for mode_idx, (mode_label, mode_key) in enumerate(modes):
                tp_dict = data.host if mode_key == "host" else data.device
                tp = tp_dict.get(operation, 0.0)
                x = plat_center + (mode_idx - (n_modes - 1) / 2) * bar_width
                alpha = 1.0 if mode_key == "device" else 0.55
                hatch = _OVERHEAD_HATCH if mode_key == "host" else None
                ax.bar(
                    x,
                    tp,
                    bar_width,
                    color=_KERNEL_COLOR,
                    edgecolor="black",
                    linewidth=pu.BAR_EDGE_WIDTH,
                    alpha=alpha,
                    hatch=hatch,
                    zorder=3,
                )

    ax.set_xticks([idx * _GROUP_SPACING for idx in range(len(_OPERATIONS))])
    ax.set_xticklabels(_OPERATIONS, fontsize=pu.TICK_LABEL_FONT_SIZE)
    ax.set_xlim(-0.55, (len(_OPERATIONS) - 1) * _GROUP_SPACING + 0.55)

    if show_ylabel:
        ax.set_ylabel(pu.THROUGHPUT_LABEL, fontsize=pu.AXIS_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=pu.TITLE_FONT_SIZE - 2, pad=8)
    ax.grid(True, axis="y", ls="--", alpha=pu.GRID_ALPHA)


def save_overhead_figure(
    output_pdf: Path,
    output_png: Path,
    small_hbm3: PipelineThroughput,
    small_gddr7: PipelineThroughput,
    large_hbm3: PipelineThroughput,
    large_gddr7: PipelineThroughput,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=_SUBPLOT_FIGSIZE,
        gridspec_kw={"wspace": _SUBPLOT_WSPACE},
    )
    legend_handles = plot_overhead_panel(
        ax_left,
        small_hbm3,
        small_gddr7,
        title="Small workload",
        show_ylabel=True,
    )
    plot_overhead_panel(
        ax_right,
        large_hbm3,
        large_gddr7,
        title="Large workload",
        show_ylabel=False,
    )

    fig.subplots_adjust(
        left=_SUBPLOT_LEFT_MARGIN,
        right=_SUBPLOT_RIGHT_MARGIN,
        bottom=_SUBPLOT_BOTTOM_MARGIN,
        top=_SUBPLOT_TOP_MARGIN,
    )
    platform_handles, operation_handles, marker_handles = split_overhead_legend_handles(
        legend_handles
    )
    add_overhead_figure_legend(
        ax, platform_handles, operation_handles, marker_handles
    )

    fig.savefig(
        output_pdf,
        pad_inches=_FIGURE_PAD_INCHES,
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(pdf_message, fg=typer.colors.GREEN)
    plt.close(fig)


def save_absolute_figure(
    output_path: Path,
    hbm3: PipelineThroughput,
    gddr7: PipelineThroughput,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=_SUBPLOT_FIGSIZE)
    plot_absolute_panel(
        ax,
        hbm3,
        gddr7,
        show_ylabel=False,
    )

    fig.subplots_adjust(
        left=_SUBPLOT_LEFT_MARGIN,
        right=_SUBPLOT_RIGHT_MARGIN,
        bottom=_SUBPLOT_BOTTOM_MARGIN,
        top=_SUBPLOT_TOP_MARGIN,
    )
    pu.save_figure(
        fig,
        output_path,
        message=f"Absolute throughput figure saved to {output_path}",
    )


def validate_pipeline_data(data: PipelineThroughput, label: str) -> bool:
    if not data.host and not data.device:
        typer.secho(
            f"No cuSBF host/device throughput rows found in {label}",
            fg=typer.colors.RED,
            err=True,
        )
        return False
    return True


@app.command()
def main(
    csv_hbm3_small: Path = typer.Argument(
        ...,
        help="GH200 (HBM3) cusbf-host-overhead CSV for small workload",
    ),
    csv_gddr7_small: Path = typer.Argument(
        ...,
        help="RTX PRO 6000 (GDDR7) cusbf-host-overhead CSV for small workload",
    ),
    csv_hbm3_large: Path = typer.Argument(
        ...,
        help="GH200 (HBM3) cusbf-host-overhead CSV for large workload",
    ),
    csv_gddr7_large: Path = typer.Argument(
        ...,
        help="RTX PRO 6000 (GDDR7) cusbf-host-overhead CSV for large workload",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots and summary CSV (default: build/)",
    ),
    absolute: bool = typer.Option(
        False,
        "--absolute",
        help="Also emit grouped absolute-throughput PDF (host vs device GKmer/s)",
    ),
):
    """Plot host-sequence transfer overhead from four benchmark CSV files."""
    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    small_hbm3 = load_pipeline_throughput(csv_hbm3_small)
    small_gddr7 = load_pipeline_throughput(csv_gddr7_small)
    large_hbm3 = load_pipeline_throughput(csv_hbm3_large)
    large_gddr7 = load_pipeline_throughput(csv_gddr7_large)

    if not all(
        validate_pipeline_data(data, str(path))
        for data, path in (
            (small_hbm3, csv_hbm3_small),
            (small_gddr7, csv_gddr7_small),
            (large_hbm3, csv_hbm3_large),
            (large_gddr7, csv_gddr7_large),
        )
    ):
        raise typer.Exit(1)

    summary_rows = build_summary_rows(small_hbm3, small_gddr7, "small")
    summary_rows.extend(build_summary_rows(large_hbm3, large_gddr7, "large"))
    write_summary_csv(summary_rows, output_dir / f"{_OUTPUT_BASENAME}_summary.csv")

    overhead_pdf = output_dir / f"{_OUTPUT_BASENAME}.pdf"
    overhead_png = output_dir / f"{_OUTPUT_BASENAME}.png"
    save_overhead_figure(
        overhead_pdf,
        overhead_png,
        small_hbm3,
        small_gddr7,
        large_hbm3,
        large_gddr7,
    )

    if absolute:
        save_absolute_figure(
            output_dir / f"{_OUTPUT_BASENAME}_absolute.pdf",
            small_hbm3,
            small_gddr7,
            large_hbm3,
            large_gddr7,
        )


if __name__ == "__main__":
    app()

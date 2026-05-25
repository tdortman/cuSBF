#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot FPR benchmark results")


CUSBF_FIXTURE_PATTERN = re.compile(
    r"^CuSbfFixture(?P<s>\d+)?$", re.IGNORECASE
)
CUSBF_CONFIG_FIXTURE_PATTERN = re.compile(
    r"^CuSBF_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_Fixture$",
    re.IGNORECASE,
)
SUPERBLOOM_CPU_CONFIG_FIXTURE_PATTERN = re.compile(
    r"^SuperBloomCpu_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_Fixture$",
    re.IGNORECASE,
)
CUCO_FIXTURE_PATTERN = re.compile(r"^CucoBloomFixture$", re.IGNORECASE)
CUCKOO_GPU_FIXTURE_PATTERN = re.compile(r"^CuckooGpuFixture$", re.IGNORECASE)
GQF_FIXTURE_PATTERN = re.compile(r"^GqfFixture$", re.IGNORECASE)
TCF_FIXTURE_PATTERN = re.compile(r"^TcfFixture$", re.IGNORECASE)
# Cuckoo-GPU-style fair FPR benchmarks (gpu-filter-fair-comparison).
CUCO_FAIR_FPR_PATTERN = re.compile(r"^CucoBloom_FPR$", re.IGNORECASE)
CUCKOO_FAIR_FPR_PATTERN = re.compile(r"^CuckooGpu_FPR$", re.IGNORECASE)
GQF_FAIR_FPR_PATTERN = re.compile(r"^Gqf_FPR$", re.IGNORECASE)
TCF_FAIR_FPR_PATTERN = re.compile(r"^Tcf_FPR$", re.IGNORECASE)
GCF_FAIR_FPR_PATTERN = re.compile(r"^GCF_FPR$", re.IGNORECASE)
BBF_FAIR_FPR_PATTERN = re.compile(r"^BBF_FPR$", re.IGNORECASE)
PROTEIN_GQF_FIXTURE_PATTERN = re.compile(r"^ProteinGqfFixture$", re.IGNORECASE)
PROTEIN_TCF_FIXTURE_PATTERN = re.compile(r"^ProteinTcfFixture$", re.IGNORECASE)

# Same colour as cuSBF; marker distinguishes s (blocked-window size).
CUSBF_S_MARKERS: dict[int, str] = {
    31: "x",
    30: "o",
    28: "s",
    27: "^",
    26: "D",
    24: "P",
    22: "v",
    20: "<",
    16: ">",
}
DEFAULT_CUSBF_S_MARKER = "o"


def parse_cusbf_variant(fixture_name: str, row: pd.Series) -> Optional[int]:
    """Parse cuSBF variant ``s`` value from fixture name or CSV counters."""
    match = CUSBF_FIXTURE_PATTERN.match(fixture_name)
    if match is not None:
        fixture_suffix = match.group("s")
        if fixture_suffix is not None:
            return int(fixture_suffix)

        row_s = row.get("s")
        if pd.notna(row_s):
            try:
                return int(float(row_s))
            except (TypeError, ValueError):
                return None

        return None

    config_match = CUSBF_CONFIG_FIXTURE_PATTERN.match(fixture_name)
    if config_match is not None:
        return int(config_match.group("s"))

    cpu_match = SUPERBLOOM_CPU_CONFIG_FIXTURE_PATTERN.match(fixture_name)
    if cpu_match is not None:
        return int(cpu_match.group("s"))

    return None


def extract_filter_series(name: str, row: pd.Series) -> Optional[tuple[str, str, str]]:
    """Extract filter series key, base style key, and display label."""
    stripped_name = str(name).strip('"')
    parts = stripped_name.split("/")

    if len(parts) >= 2:
        fixture_name = parts[0]
        operation = parts[1]
        cusbf_variant = parse_cusbf_variant(fixture_name, row)
        is_cpu = SUPERBLOOM_CPU_CONFIG_FIXTURE_PATTERN.match(fixture_name) is not None
        is_gpu = (
            CUSBF_CONFIG_FIXTURE_PATTERN.match(fixture_name) is not None
            or CUSBF_FIXTURE_PATTERN.match(fixture_name) is not None
        )
        is_cuco = CUCO_FIXTURE_PATTERN.match(fixture_name) is not None
        is_cuckoo = CUCKOO_GPU_FIXTURE_PATTERN.match(fixture_name) is not None
        is_gqf = GQF_FIXTURE_PATTERN.match(fixture_name) is not None
        is_tcf = TCF_FIXTURE_PATTERN.match(fixture_name) is not None
        is_protein_gqf = PROTEIN_GQF_FIXTURE_PATTERN.match(fixture_name) is not None
        is_protein_tcf = PROTEIN_TCF_FIXTURE_PATTERN.match(fixture_name) is not None
        is_cuco_fair = CUCO_FAIR_FPR_PATTERN.match(fixture_name) is not None
        is_cuckoo_fair = CUCKOO_FAIR_FPR_PATTERN.match(fixture_name) is not None
        is_gqf_fair = GQF_FAIR_FPR_PATTERN.match(fixture_name) is not None
        is_tcf_fair = TCF_FAIR_FPR_PATTERN.match(fixture_name) is not None
        is_gcf_fair = GCF_FAIR_FPR_PATTERN.match(fixture_name) is not None
        is_bbf_fair = BBF_FAIR_FPR_PATTERN.match(fixture_name) is not None

        if (
            is_gpu
            or is_cpu
            or is_cuco
            or is_cuckoo
            or is_gqf
            or is_tcf
            or is_protein_gqf
            or is_protein_tcf
            or is_cuco_fair
            or is_cuckoo_fair
            or is_gqf_fair
            or is_tcf_fair
            or is_gcf_fair
            or is_bbf_fair
        ):
            if operation.upper() != "FPR":
                return None

            if is_cpu and cusbf_variant is not None:
                series_key = f"superbloom_cpu_s{cusbf_variant}"
                display_name = f"{pu.get_filter_display_name('superbloom_cpu')} (s={cusbf_variant})"
                return series_key, "superbloom_cpu", display_name

            if is_gpu and cusbf_variant is not None:
                series_key = f"cusbf_s{cusbf_variant}"
                display_name = f"{pu.get_filter_display_name('cusbf')} (s={cusbf_variant})"
                return series_key, "cusbf", display_name

            if is_cuco or is_cuco_fair or is_bbf_fair:
                return (
                    "cucobloom",
                    "cucobloom",
                    pu.get_filter_display_name("cucobloom"),
                )

            if is_cuckoo or is_cuckoo_fair or is_gcf_fair:
                return (
                    "cuckoogpu",
                    "cuckoogpu",
                    pu.get_filter_display_name("cuckoogpu"),
                )

            if is_gqf or is_gqf_fair:
                return ("gqf", "gqf", pu.get_filter_display_name("gqf"))

            if is_tcf or is_tcf_fair:
                return ("tcf", "tcf", pu.get_filter_display_name("tcf"))

            if is_protein_gqf:
                return (
                    "proteingqf",
                    "proteingqf",
                    pu.get_filter_display_name("proteingqf"),
                )

            if is_protein_tcf:
                return (
                    "proteintcf",
                    "proteintcf",
                    pu.get_filter_display_name("proteintcf"),
                )

    return None


def parse_s_from_series_key(filter_type: str) -> Optional[int]:
    """Extract blocked-window ``s`` from a cuSBF series key like ``cusbf_s30``."""
    match = re.search(r"_s(\d+)$", filter_type)
    if match is None:
        return None
    return int(match.group(1))


def get_plot_style(filter_type: str, base_filter: str) -> dict[str, str]:
    """Style a series; cuSBF/superbloom_cpu s-variants share colour, differ by marker."""
    style = dict(
        pu.FILTER_STYLES.get(
            filter_type,
            pu.FILTER_STYLES.get(base_filter, {}),
        )
    )

    if base_filter in ("cusbf", "superbloom_cpu"):
        s_value = parse_s_from_series_key(filter_type)
        if s_value is not None:
            style["marker"] = CUSBF_S_MARKERS.get(s_value, DEFAULT_CUSBF_S_MARKER)
            style["linestyle"] = "-"
            style["color"] = pu.FILTER_STYLES[base_filter]["color"]

    return style


def sort_filters_by_descending_fpr(
    fpr_data: dict[str, dict[float, float]],
    filter_display_names: dict[str, str],
) -> list[str]:
    """Sort filters by highest observed FPR (descending)."""

    def fpr_sort_key(filter_type: str) -> tuple[float, str]:
        values = fpr_data.get(filter_type, {}).values()
        max_fpr = max(values) if values else float("-inf")
        display_name = filter_display_names.get(
            filter_type,
            pu.get_filter_display_name(filter_type),
        )
        return (-max_fpr, display_name)

    return sorted(fpr_data.keys(), key=fpr_sort_key)


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to the CSV file containing benchmark results",
    ),
    output_dir: Path = typer.Option(
        Path("./build"),
        help="Directory to save output plots",
    ),
):
    """
    Plot FPR vs filter memory from gpu-filter-comparison (and related) CSV output.

    cuSBF ``s`` variants share colour; marker shape encodes ``s``.
    """
    if not csv_file.exists():
        typer.secho(f"CSV file not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.secho(f"Reading CSV from: {csv_file}", fg=typer.colors.CYAN)

    df = pd.read_csv(csv_file)

    # Filter for median records only
    df = df[df["name"].str.endswith("_median")]

    # Dictionary structure: filter_type -> {memory_size: metric_value}
    fpr_data = defaultdict(dict)
    bits_per_item_data = defaultdict(dict)
    filter_display_names: dict[str, str] = {}
    filter_base_types: dict[str, str] = {}

    for _, row in df.iterrows():
        name = row["name"]
        filter_series = extract_filter_series(name, row)

        if filter_series is None:
            continue

        filter_type, base_filter_type, display_name = filter_series
        filter_display_names[filter_type] = display_name
        filter_base_types[filter_type] = base_filter_type

        memory_bytes = row.get("memory_bytes")
        fpr_percentage = row.get("fpr_percentage")
        bits_per_item = row.get("bits_per_item")

        if pd.notna(memory_bytes):
            if pd.notna(fpr_percentage):
                fpr_data[filter_type][memory_bytes] = fpr_percentage
            if pd.notna(bits_per_item):
                bits_per_item_data[filter_type][memory_bytes] = bits_per_item

    if not fpr_data:
        typer.secho("No FPR data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    fpr_sorted_filters = sort_filters_by_descending_fpr(
        dict(fpr_data),
        filter_display_names,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: FPR vs Memory Size
    fig, ax = plt.subplots(figsize=(12, 8))

    for filter_type in fpr_sorted_filters:
        memory_sizes = sorted(fpr_data[filter_type].keys())
        fpr_values = [fpr_data[filter_type][mem] for mem in memory_sizes]

        style = get_plot_style(
            filter_type,
            filter_base_types.get(filter_type, filter_type),
        )
        ax.plot(
            memory_sizes,
            fpr_values,
            label=filter_display_names.get(
                filter_type,
                pu.get_filter_display_name(filter_type),
            ),
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            **style,  # type: ignore
        )

    ax.set_xlabel(
        "Filter memory [bytes]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_ylabel(
        "False Positive Rate [%]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=pu.AXIS_LABEL_FONT_SIZE)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    axes_box = ax.get_position()
    legend_center_x = (axes_box.x0 + axes_box.x1) / 2
    legend_y = axes_box.y1 + 0.01
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(legend_center_x, legend_y),
        ncol=max(1, min(4, len(labels))),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )

    output_file = output_dir / "fpr_vs_memory.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(
        f"FPR vs memory plot saved to {output_file.absolute()}",
        fg=typer.colors.GREEN,
    )
    plt.close()

    # Plot 2: Bits per Item vs Memory Size
    fig, ax = plt.subplots(figsize=(12, 8))

    bits_filter_order = [
        f for f in fpr_sorted_filters if f in bits_per_item_data
    ] + sorted(set(bits_per_item_data.keys()) - set(fpr_sorted_filters))

    for filter_type in bits_filter_order:
        memory_sizes = sorted(bits_per_item_data[filter_type].keys())
        bits_values = [bits_per_item_data[filter_type][mem] for mem in memory_sizes]

        style = get_plot_style(
            filter_type,
            filter_base_types.get(filter_type, filter_type),
        )
        ax.plot(
            memory_sizes,
            bits_values,
            label=filter_display_names.get(
                filter_type,
                pu.get_filter_display_name(filter_type),
            ),
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            **style,  # type: ignore
        )

    ax.set_xlabel(
        "Memory Size [bytes]", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold"
    )
    ax.set_ylabel("Bits per Item", fontsize=pu.AXIS_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.tick_params(axis="both", labelsize=pu.AXIS_LABEL_FONT_SIZE)
    ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    axes_box = ax.get_position()
    legend_center_x = (axes_box.x0 + axes_box.x1) / 2
    legend_y = axes_box.y1 + 0.01
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=pu.LEGEND_FONT_SIZE,
        loc="lower center",
        bbox_to_anchor=(legend_center_x, legend_y),
        ncol=max(1, min(4, len(labels))),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )

    output_file = output_dir / "bits_per_item_vs_memory.pdf"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        transparent=True,
        format="pdf",
        dpi=600,
    )
    typer.secho(
        f"Bits per item plot saved to {output_file.absolute()}",
        fg=typer.colors.GREEN,
    )
    plt.close()

    typer.secho("\nAll plots generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()

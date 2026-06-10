#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "typer",
# ]
# ///
"""Line plots of false-positive hits vs filter size from gpu-filter-fpr-fastx CSV."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot gpu-filter-fpr-fastx benchmark results")

CUSBF_FPR_FASTX_PATTERN = re.compile(
    r"^CuSBF_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_FprFastxFixture$",
    re.IGNORECASE,
)
SUPERBLOOM_CPU_FPR_FASTX_PATTERN = re.compile(
    r"^SuperBloomCpuFprFastxFixture(?P<s>\d+)$",
    re.IGNORECASE,
)
CUCO_FPR_FASTX_PATTERN = re.compile(r"^CucoBloomFprFastxFixture$", re.IGNORECASE)
CUCKOO_FPR_FASTX_PATTERN = re.compile(r"^CuckooGpuFprFastxFixture$", re.IGNORECASE)
GQF_FPR_FASTX_PATTERN = re.compile(r"^GqfFprFastxFixture$", re.IGNORECASE)
TCF_FPR_FASTX_PATTERN = re.compile(r"^TcfFprFastxFixture$", re.IGNORECASE)

CUSBF_S_MARKERS: dict[int, str] = {
    28: "s",
    30: "o",
    31: "x",
    27: "^",
}
DEFAULT_CUSBF_S_MARKER = "o"


def extract_filter_series(name: str, row: pd.Series) -> Optional[tuple[str, str, str]]:
    """Return (series_key, base_filter, display_label)."""
    stripped_name = str(name).strip('"')
    parts = stripped_name.split("/")
    if len(parts) < 2 or parts[1].upper() != "FPR":
        return None

    fixture_name = parts[0]

    if CUCKOO_FPR_FASTX_PATTERN.match(fixture_name):
        return ("cuckoogpu", "cuckoogpu", pu.get_filter_display_name("cuckoogpu"))
    if CUCO_FPR_FASTX_PATTERN.match(fixture_name):
        return ("cucobloom", "cucobloom", pu.get_filter_display_name("cucobloom"))
    if GQF_FPR_FASTX_PATTERN.match(fixture_name):
        return ("gqf", "gqf", pu.get_filter_display_name("gqf"))
    if TCF_FPR_FASTX_PATTERN.match(fixture_name):
        return ("tcf", "tcf", pu.get_filter_display_name("tcf"))

    cusbf_match = CUSBF_FPR_FASTX_PATTERN.match(fixture_name)
    if cusbf_match is not None:
        s = int(cusbf_match.group("s"))
        key = f"cusbf_s{s}"
        return (
            key,
            "cusbf",
            f"{pu.get_filter_display_name('cusbf')} (s={s})",
        )

    sb_match = SUPERBLOOM_CPU_FPR_FASTX_PATTERN.match(fixture_name)
    if sb_match is not None:
        s = int(sb_match.group("s"))
        key = f"superbloom_cpu_s{s}"
        return (
            key,
            "superbloom_cpu",
            f"{pu.get_filter_display_name('superbloom_cpu')} (s={s})",
        )

    if fixture_name.lower() == "superbloomcpufprfastxfixture":
        row_s = row.get("s")
        if pd.notna(row_s):
            s = int(float(row_s))
            key = f"superbloom_cpu_s{s}"
            return (
                key,
                "superbloom_cpu",
                f"{pu.get_filter_display_name('superbloom_cpu')} (s={s})",
            )

    return None


def parse_s_from_series_key(filter_type: str) -> Optional[int]:
    match = re.search(r"_s(\d+)$", filter_type)
    return int(match.group(1)) if match else None




def filter_size_bits(row: pd.Series) -> float | None:
    """Allocated filter capacity in bits (x-axis)."""
    filter_bits = row.get("filter_bits")
    if pd.notna(filter_bits):
        return float(filter_bits)
    memory_bytes = row.get("memory_bytes")
    if pd.notna(memory_bytes):
        return float(memory_bytes) * 8.0
    return None

def get_plot_style(filter_type: str, base_filter: str) -> dict[str, str]:
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


def sort_filters_by_descending_fp(
    fp_data: dict[str, dict[float, float]],
    filter_display_names: dict[str, str],
) -> list[str]:
    def sort_key(filter_type: str) -> tuple[float, str]:
        values = fp_data.get(filter_type, {}).values()
        max_fp = max(values) if values else float("-inf")
        return (-max_fp, filter_display_names.get(filter_type, filter_type))

    return sorted(fp_data.keys(), key=sort_key)


@app.command()
def main(
    csv_file: Path = typer.Argument(
        ...,
        help="Path to gpu-filter-fpr-fastx CSV output",
    ),
    output_dir: Path = typer.Option(
        Path("./build"),
        help="Directory to save output plots",
    ),
):
    """
    Plot false-positive hit counts vs filter size [bits].

    Each series is one filter; disjoint random k-mer queries (default 1B) count as FPs.
    """
    df = pu.load_csv(csv_file)
    df = df[df["name"].str.endswith("_median", na=False)]

    fp_data: dict[str, dict[float, float]] = defaultdict(dict)
    filter_display_names: dict[str, str] = {}
    filter_base_types: dict[str, str] = {}

    for _, row in df.iterrows():
        series = extract_filter_series(row["name"], row)
        if series is None:
            continue

        filter_type, base_filter_type, display_name = series
        filter_display_names[filter_type] = display_name
        filter_base_types[filter_type] = base_filter_type

        size_bits = filter_size_bits(row)
        false_positives = row.get("false_positives")
        if size_bits is not None and pd.notna(false_positives):
            fp_data[filter_type][size_bits] = float(false_positives)

    if not fp_data:
        typer.secho("No false-positive data found in CSV", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    all_sizes = {x for sizes in fp_data.values() for x in sizes}
    if len(all_sizes) <= 1:
        only = next(iter(all_sizes)) if all_sizes else float("nan")
        typer.secho(
            f"Filter size is constant ({only:g} bits) across all series; "
            "re-run gpu-filter-fpr-fastx (filter_bits_exp sweep 22–31) to plot FPR vs space.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    sorted_filters = sort_filters_by_descending_fp(dict(fp_data), filter_display_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for filter_type in sorted_filters:
        filter_bits_list = sorted(fp_data[filter_type].keys())
        fp_values = [fp_data[filter_type][bits] for bits in filter_bits_list]
        style = get_plot_style(
            filter_type,
            filter_base_types.get(filter_type, filter_type),
        )
        ax.plot(
            filter_bits_list,
            fp_values,
            label=filter_display_names.get(
                filter_type,
                pu.get_filter_display_name(filter_type),
            ),
            linewidth=pu.LINE_WIDTH,
            markersize=pu.MARKER_SIZE,
            **style,  # type: ignore[arg-type]
        )

    ax.set_xlabel(
        pu.paper_text("Filter size [bits]", bold=True),
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
    )
    ax.set_ylabel(
        pu.paper_text("False positives (hits / 1B random k-mers)", bold=True),
        fontsize=pu.AXIS_LABEL_FONT_SIZE,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=pu.TICK_LABEL_FONT_SIZE)
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
        ncol=max(1, min(5, len(labels))),
        framealpha=pu.LEGEND_FRAME_ALPHA,
    )

    output_file = output_dir / "fpr_fastx.pdf"
    pu.save_figure(
        fig,
        output_file,
        f"FPR FASTX plot saved to {output_file.absolute()}",
    )


if __name__ == "__main__":
    app()

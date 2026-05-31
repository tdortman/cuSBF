#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "typer",
# ]
# ///

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import plot_utils as pu
import typer

app = typer.Typer(help="Plot S/M/H parameter sweep benchmark results")

FIXTURE_PATTERN = re.compile(
    r"^CuSBF_K(?P<k>\d+)_S(?P<s>\d+)_M(?P<m>\d+)_H(?P<h>\d+)_Fixture",
    re.IGNORECASE,
)

M_CMAP = plt.colormaps["viridis"]
HEATMAP_CMAP = plt.colormaps["magma"].copy()
HEATMAP_CMAP.set_bad("#e6e6e6")
H_MARKERS = {4: "o", 8: "s", 12: "^", 16: "D"}
PARETO_LABEL_COLOR = "#1a5276"
PARETO_OUTLINE_COLOR = "#f7f9f9"
PARETO_EDGE_COLOR = "#17202a"
# Above minor grid (~1.5) and axis spines (~2.5) so outer Pareto edges stay visible.
# clip_on=False keeps edge-row halos from being clipped by the axes boundary.
PARETO_PATCH_ZORDER = 4
DOUBLE_COLUMN_WIDTH_IN = 7.1
PARETO_FIG_HEIGHT_IN = 4.4
HEATMAP_FIG_HEIGHT_IN = 5.2
PAPER_AXIS_LABEL_SIZE = 11
PAPER_TICK_LABEL_SIZE = 9
PAPER_LEGEND_SIZE = 9
PAPER_TITLE_SIZE = 10
PAPER_ANNOTATION_SIZE = 8


def parse_csv(csv_file: Path) -> pd.DataFrame:
    """Load and parse benchmark CSV, extracting S, M, H, and operation."""
    df = pu.load_csv(csv_file)
    df = df[df["name"].str.endswith("_median")].copy()

    extracted = df["name"].str.extract(FIXTURE_PATTERN)
    df["s"] = extracted["s"].astype(float)
    df["m"] = extracted["m"].astype(float)
    df["h"] = extracted["h"].astype(float)
    df["operation"] = df["name"].str.split("/").str[1]
    df = df.dropna(subset=["s", "m", "h", "operation"])

    df["s"] = df["s"].astype(int)
    df["m"] = df["m"].astype(int)
    df["h"] = df["h"].astype(int)
    df["real_time"] = pd.to_numeric(df["real_time"], errors="coerce")
    df["fpr_percentage"] = pd.to_numeric(
        df.get("fpr_percentage", np.nan), errors="coerce"
    )
    return df


def build_merged_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (s, m, h) with Insert/Query/FPR metrics."""
    insert_df = df[df["operation"] == "Insert"][["s", "m", "h", "real_time"]].rename(
        columns={"real_time": "insert_time"}
    )
    query_df = df[df["operation"] == "Query"][["s", "m", "h", "real_time"]].rename(
        columns={"real_time": "query_time"}
    )
    fpr_df = df[df["operation"] == "FPR"][
        ["s", "m", "h", "real_time", "fpr_percentage"]
    ].rename(columns={"real_time": "fpr_time", "fpr_percentage": "fpr"})

    merged = insert_df.merge(query_df, on=["s", "m", "h"], how="outer")
    merged = merged.merge(fpr_df, on=["s", "m", "h"], how="outer")
    merged = merged.dropna(subset=["fpr"])
    merged["total_time"] = merged["insert_time"] + merged["query_time"]
    return merged


def compute_pareto_mask(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Boolean mask for Pareto-optimal points (minimize both x and y)."""
    values = df[[x_col, y_col]].to_numpy()
    mask = np.ones(len(df), dtype=bool)

    for i, (x, y) in enumerate(values):
        dominated = (
            (values[:, 0] <= x)
            & (values[:, 1] <= y)
            & ((values[:, 0] < x) | (values[:, 1] < y))
        )
        mask[i] = not dominated.any()

    return pd.Series(mask, index=df.index)


def build_log_norm(series: pd.Series) -> LogNorm:
    """Log normalization resilient to near-constant data."""
    positive = series.dropna()
    positive = positive[positive > 0]
    if positive.empty:
        raise ValueError("Heatmap metric must contain at least one positive value")

    vmin = float(positive.min())
    vmax = float(positive.max())
    if np.isclose(vmin, vmax):
        vmax = vmin * 1.0001
    return LogNorm(vmin=vmin, vmax=vmax)


def scaled_tick_label_size(
    num_ticks: int, base_size: int = PAPER_TICK_LABEL_SIZE
) -> int:
    """Shrink tick labels as categorical axes become denser."""
    if num_ticks <= 7:
        return base_size
    return max(6, int(round(base_size - 0.45 * (num_ticks - 7))))


def categorical_tick_labels(values: list[int]) -> list[str]:
    """Show a readable subset of dense categorical tick labels."""
    num_ticks = len(values)
    if num_ticks <= 9:
        return [str(value) for value in values]

    step = 2 if num_ticks <= 17 else 3
    labels = []
    for idx, value in enumerate(values):
        show_label = (idx % step == 0) or (idx == num_ticks - 1)
        labels.append(str(value) if show_label else "")
    return labels


def plot_pareto_scatters(merged: pd.DataFrame, output_dir: Path) -> None:
    m_values_sorted = sorted(merged["m"].unique())
    h_values_sorted = sorted(merged["h"].unique())
    m_colors = {
        m: M_CMAP(i / max(len(m_values_sorted) - 1, 1))
        for i, m in enumerate(m_values_sorted)
    }

    pareto_configs = [
        ("insert_time", "Insert Time [ms]"),
        ("query_time", "Query Time [ms]"),
        ("total_time", "Insert + Query Time [ms]"),
    ]

    for time_col, y_label in pareto_configs:
        subset = merged.dropna(subset=[time_col, "fpr"]).copy()
        if subset.empty:
            continue

        subset["pareto"] = compute_pareto_mask(subset, "fpr", time_col)
        fig, ax = plt.subplots(
            figsize=(DOUBLE_COLUMN_WIDTH_IN, PARETO_FIG_HEIGHT_IN),
            constrained_layout=True,
        )

        for is_pareto in [False, True]:
            group = subset[subset["pareto"] == is_pareto]
            if group.empty:
                continue

            for h_val in h_values_sorted:
                h_group = group[group["h"] == h_val]
                if h_group.empty:
                    continue

                ax.scatter(
                    h_group["fpr"],
                    h_group[time_col],
                    c=[m_colors[m] for m in h_group["m"]],
                    marker=H_MARKERS.get(h_val, "o"),
                    s=115 if is_pareto else 35,
                    alpha=0.95 if is_pareto else 0.35,
                    edgecolors="black" if is_pareto else "none",
                    linewidths=0.8 if is_pareto else 0.0,
                    zorder=4 if is_pareto else 2,
                )

        pareto = subset[subset["pareto"]].sort_values("fpr")
        if len(pareto) > 1:
            ax.plot(
                pareto["fpr"],
                pareto[time_col],
                "--",
                color="#2E86AB",
                alpha=0.65,
                linewidth=1.6,
                zorder=3,
            )

        offsets = [(8, 8), (-8, -8), (8, -8), (-8, 8)]
        for _, row in pareto.iterrows():
            ox, oy = offsets[(int(row["s"]) + int(row["m"])) % len(offsets)]
            ax.annotate(
                f"S{int(row['s'])},M{int(row['m'])},H{int(row['h'])}",
                (row["fpr"], row[time_col]),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=PAPER_ANNOTATION_SIZE,
                fontweight="bold",
                color=PARETO_LABEL_COLOR,
                zorder=5,
            )

        ax.set_xlabel("FPR [%]", fontsize=PAPER_AXIS_LABEL_SIZE, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=PAPER_AXIS_LABEL_SIZE, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=PAPER_TICK_LABEL_SIZE)
        ax.grid(True, which="both", ls="--", alpha=pu.GRID_ALPHA)

        m_legend = [
            Patch(facecolor=m_colors[m], edgecolor="black", label=f"M={m}")
            for m in m_values_sorted
        ]
        h_legend = [
            Line2D(
                [0],
                [0],
                marker=H_MARKERS.get(h, "o"),
                color="black",
                markerfacecolor="white",
                linestyle="None",
                markersize=7,
                label=f"H={h}",
            )
            for h in h_values_sorted
        ]
        style_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="white",
                linestyle="None",
                markersize=7,
                label="Pareto config",
            ),
            Line2D(
                [0],
                [0],
                color="#2E86AB",
                linestyle="--",
                linewidth=1.6,
                label="Pareto frontier",
            ),
        ]

        legend_font = PAPER_LEGEND_SIZE
        m_legend_artist = ax.legend(
            handles=m_legend,
            title="M",
            fontsize=legend_font,
            title_fontsize=legend_font,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            framealpha=0,
        )
        ax.add_artist(m_legend_artist)
        h_legend_artist = ax.legend(
            handles=h_legend,
            title="H",
            fontsize=legend_font,
            title_fontsize=legend_font,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.01),
            framealpha=0,
        )
        ax.add_artist(h_legend_artist)
        ax.legend(
            handles=style_legend,
            fontsize=legend_font,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            framealpha=0,
        )

        suffix = time_col.replace("_time", "")
        pareto_path = output_dir / f"param_sweep_pareto_{suffix}.pdf"
        pu.save_figure(
            fig, pareto_path, f"Pareto plot ({suffix}) saved to {pareto_path}"
        )


def plot_heatmap_summary(merged: pd.DataFrame, output_dir: Path) -> None:
    subset = merged.dropna(subset=["fpr", "total_time"]).copy()
    if subset.empty:
        return

    subset["pareto_total"] = compute_pareto_mask(subset, "fpr", "total_time")
    s_values_sorted = sorted(subset["s"].unique())
    m_values_sorted = sorted(subset["m"].unique())
    h_values_sorted = sorted(subset["h"].unique())
    s_index = {s: idx for idx, s in enumerate(s_values_sorted)}
    m_index = {m: idx for idx, m in enumerate(m_values_sorted)}

    metric_configs = [
        ("fpr", "FPR [%]"),
        ("total_time", "Insert + Query Time [ms]"),
    ]
    norms = {
        metric_col: build_log_norm(subset[metric_col])
        for metric_col, _ in metric_configs
    }

    fig, axes = plt.subplots(
        len(metric_configs),
        len(h_values_sorted),
        figsize=(DOUBLE_COLUMN_WIDTH_IN, HEATMAP_FIG_HEIGHT_IN),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(len(metric_configs), len(h_values_sorted))

    row_images = []
    for row_idx, (metric_col, metric_label) in enumerate(metric_configs):
        row_image = None
        for col_idx, h_val in enumerate(h_values_sorted):
            ax = axes[row_idx, col_idx]
            ax.set_axisbelow(True)
            h_subset = subset[subset["h"] == h_val]
            pivot = h_subset.pivot(index="s", columns="m", values=metric_col).reindex(
                index=s_values_sorted, columns=m_values_sorted
            )

            image = ax.imshow(
                pivot.to_numpy(dtype=float),
                origin="lower",
                aspect="auto",
                cmap=HEATMAP_CMAP,
                norm=norms[metric_col],
            )
            if row_image is None:
                row_image = image

            x_tick_label_size = scaled_tick_label_size(len(m_values_sorted))
            x_tick_labels = categorical_tick_labels(m_values_sorted)
            y_tick_label_size = scaled_tick_label_size(len(s_values_sorted))

            ax.set_title(f"H={h_val}", fontsize=PAPER_TITLE_SIZE, fontweight="bold")
            ax.set_xticks(range(len(m_values_sorted)))
            ax.set_xticklabels(m_values_sorted)
            ax.set_xticklabels(
                x_tick_labels,
                fontsize=x_tick_label_size,
                rotation=0,
                ha="center",
            )
            ax.set_yticks(range(len(s_values_sorted)))
            ax.set_yticklabels(s_values_sorted, fontsize=y_tick_label_size)
            ax.set_xticks(np.arange(-0.5, len(m_values_sorted), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(s_values_sorted), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.45)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_zorder(0.5)

            for _, row in h_subset[h_subset["pareto_total"]].iterrows():
                x = m_index[int(row["m"])] - 0.5
                y = s_index[int(row["s"])] - 0.5
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        fill=False,
                        edgecolor=PARETO_OUTLINE_COLOR,
                        linewidth=1.8,
                        zorder=PARETO_PATCH_ZORDER,
                        clip_on=False,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        fill=False,
                        edgecolor=PARETO_EDGE_COLOR,
                        linewidth=0.8,
                        zorder=PARETO_PATCH_ZORDER + 0.1,
                        clip_on=False,
                    )
                )

            if col_idx == 0:
                ax.set_ylabel("S", fontsize=PAPER_AXIS_LABEL_SIZE, fontweight="bold")
            if row_idx == len(metric_configs) - 1:
                ax.set_xlabel("M", fontsize=PAPER_AXIS_LABEL_SIZE, fontweight="bold")

        row_images.append((row_image, metric_label))

    for row_idx, (image, metric_label) in enumerate(row_images):
        colorbar = fig.colorbar(
            image,
            ax=axes[row_idx, :].tolist(),
            shrink=0.92,
            pad=0.02,
        )
        colorbar.set_label(
            metric_label, fontsize=PAPER_AXIS_LABEL_SIZE, fontweight="bold"
        )
        colorbar.ax.tick_params(labelsize=PAPER_TICK_LABEL_SIZE)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color=PARETO_EDGE_COLOR,
                markerfacecolor="none",
                markeredgewidth=0.8,
                linestyle="None",
                markersize=7,
                label="Pareto config",
            )
        ],
        loc="upper right",
        bbox_to_anchor=(1.035, 1.0),
        framealpha=1.0,
        facecolor="white",
        edgecolor="#dddddd",
        fontsize=PAPER_LEGEND_SIZE - 1,
        handletextpad=0.3,
        borderpad=0.2,
    )

    heatmap_path = output_dir / "param_sweep_heatmap_summary.pdf"
    pu.save_figure(
        fig,
        heatmap_path,
        f"Heatmap summary saved to {heatmap_path}",
    )


@app.command()
def main(
    csv_file: Path = typer.Argument(..., help="Path to benchmark CSV results"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Output directory (default: build/)"
    ),
):
    if not csv_file.exists():
        typer.secho(f"CSV not found: {csv_file}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    df = parse_csv(csv_file)
    if df.empty:
        typer.secho("No data", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    merged = build_merged_table(df)
    if merged.empty:
        typer.secho("Not enough data for plots", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(0)

    output_dir = pu.resolve_output_dir(output_dir, Path(__file__))

    typer.secho("Generating Pareto plots...", fg=typer.colors.CYAN)
    plot_pareto_scatters(merged, output_dir)

    typer.secho("Generating heatmap summary...", fg=typer.colors.CYAN)
    plot_heatmap_summary(merged, output_dir)

    typer.secho("\nAll plots generated successfully!", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()

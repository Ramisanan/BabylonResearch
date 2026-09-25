"""CLI: z-score normalize each commodity separately --

    z_i(t) = (x_i(t) - mean(x_i)) / std(x_i)

-- for both the X and 1/X formats of the DataverseNL Appendix dataset
(Data/appendix.xlsx). Same methodology and X/1-X convention as
run_benford_appendix.py: the sheet's raw columns are silver-per-quantity
(the price), so they are 1/X; X is their reciprocal (quantity per unit
silver). Each commodity's mean/std/CDF is computed only from that
commodity's own values -- commodities are never mixed together.

Saves:
  - zscore_X.csv, zscore_1overX.csv           (per-row commodity, value, zscore, cdf)
  - graphs/zscore_overlay_X.png               (normalized PDFs, X)
  - graphs/zscore_overlay_1overX.png          (normalized PDFs, 1/X)
  - graphs/zscore_overlay_side_by_side.png    (normalized PDFs, X vs 1/X)
  - graphs/zscore_cdf_X.png                   (normalized CDFs, X)
  - graphs/zscore_cdf_1overX.png              (normalized CDFs, 1/X)
  - graphs/zscore_cdf_side_by_side.png        (normalized CDFs, X vs 1/X)
into "Zscore Normalized Data Output for Appendix Data/".

Usage:
    python scripts/zscore_normalize_appendix.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import zscore_normalize, empirical_cdf
from babylon_prices.io import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data" / "appendix.xlsx"
OUT_DIR = ROOT / "Zscore Normalized Data Output for Appendix Data"
GRAPH_DIR = OUT_DIR / "graphs"

COMMODITY_COLUMNS = {
    "BARLEY": "barley",
    "DATES": "dates",
    "CUSCUTA": "cuscuta",
    "CRESS": "cress",
    "SESAME": "sesame",
    "WOOL": "wool",
}


def build_long_table(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """column_map: {commodity_name: column_to_read_raw_values_from}

    Each commodity's mean/std (for the z-score) and its CDF are computed
    independently, from that commodity's own values only -- never mixed
    with any other commodity."""
    rows = []
    for commodity, column in column_map.items():
        values = pd.to_numeric(df[column], errors="coerce")
        z = zscore_normalize(values).dropna()
        values = values.loc[z.index]

        # rank each commodity's own z-scores to get its own empirical CDF
        # (ordinal ranks so ties don't collide -- each row still gets a value)
        ranks = pd.Series(z.values).rank(method="first")
        cdf = (ranks / len(z)).values

        table = pd.DataFrame({
            "commodity": commodity,
            "value": values.values,
            "zscore": z.values,
            "cdf": cdf,
        })
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def plot_overlay_on_axis(ax, long_table: pd.DataFrame, title: str):
    x_grid = np.linspace(-3, 6, 400)
    for commodity, group in long_table.groupby("commodity", sort=False):
        z = group["zscore"].dropna()
        if len(z) < 2 or z.std() == 0:
            continue
        kde = gaussian_kde(z)
        ax.plot(x_grid, kde(x_grid), label=commodity, linewidth=1.8)

    ax.plot(x_grid, norm.pdf(x_grid), color="black", linestyle="--", linewidth=1.5, label="standard normal")

    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_overlay(long_table: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_overlay_on_axis(ax, long_table, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_side_by_side(long_x: pd.DataFrame, long_1overx: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_overlay_on_axis(axes[0], long_x, "X format")
    plot_overlay_on_axis(axes[1], long_1overx, "1/X format")
    fig.suptitle("Z-score normalized distributions, per commodity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cdf_on_axis(ax, long_table: pd.DataFrame, title: str):
    """Each commodity's CDF is computed from that commodity's own z-scores only."""
    for commodity, group in long_table.groupby("commodity", sort=False):
        z_sorted, cdf = empirical_cdf(group["zscore"].values)
        ax.step(z_sorted, cdf, where="post", label=commodity, linewidth=1.6)

    ax.set_xlabel("z-score")
    ax.set_ylabel("F(z)  =  P(Z <= z)")
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_cdf_overlay(long_table: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_cdf_on_axis(ax, long_table, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cdf_side_by_side(long_x: pd.DataFrame, long_1overx: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_cdf_on_axis(axes[0], long_x, "X format")
    plot_cdf_on_axis(axes[1], long_1overx, "1/X format")
    fig.suptitle("Z-score normalized cumulative distributions, per commodity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df_1overx = load_dataset(SRC).iloc[1:]  # raw columns are silver per quantity, i.e. 1/X
    df_x = df_1overx.copy()
    for column in COMMODITY_COLUMNS.values():
        df_x[column] = 1 / pd.to_numeric(df_1overx[column], errors="coerce")

    long_x = build_long_table(df_x, COMMODITY_COLUMNS)
    long_x.to_csv(OUT_DIR / "zscore_X.csv", index=False)
    print(f"Saved: {OUT_DIR / 'zscore_X.csv'}")

    long_1overx = build_long_table(df_1overx, COMMODITY_COLUMNS)
    long_1overx.to_csv(OUT_DIR / "zscore_1overX.csv", index=False)
    print(f"Saved: {OUT_DIR / 'zscore_1overX.csv'}")

    plot_overlay(long_x, "Z-score normalized distributions -- X format", GRAPH_DIR / "zscore_overlay_X.png")
    plot_overlay(long_1overx, "Z-score normalized distributions -- 1/X format", GRAPH_DIR / "zscore_overlay_1overX.png")
    plot_side_by_side(long_x, long_1overx, GRAPH_DIR / "zscore_overlay_side_by_side.png")

    plot_cdf_overlay(long_x, "Z-score normalized CDFs -- X format", GRAPH_DIR / "zscore_cdf_X.png")
    plot_cdf_overlay(long_1overx, "Z-score normalized CDFs -- 1/X format", GRAPH_DIR / "zscore_cdf_1overX.png")
    plot_cdf_side_by_side(long_x, long_1overx, GRAPH_DIR / "zscore_cdf_side_by_side.png")


if __name__ == "__main__":
    main()

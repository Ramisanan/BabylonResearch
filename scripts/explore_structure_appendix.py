"""CLI: look at the basic distribution structure of the X and 1/X data
(range, mean, std, skew, kurtosis) for every commodity in the appendix
dataset (Data/appendix.xlsx), before deciding on a normalization technique.

Same X/1-X convention as run_benford_appendix.py: the sheet's raw columns
are silver-per-quantity (the price, e.g. "gramme silver /100 litre"), so
they are 1/X; X is their reciprocal (quantity per unit silver).

Saves:
  - structure_summary.csv           (stats table)
  - graphs/<commodity>.png          (X vs 1/X histograms, side by side)
into "Data Structure Analysis for Appendix Data/".

Usage:
    python scripts/explore_structure_appendix.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import summary_stats
from babylon_prices.io import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data" / "appendix.xlsx"
OUT_DIR = ROOT / "Data Structure Analysis for Appendix Data"
GRAPH_DIR = OUT_DIR / "graphs"

COMMODITY_COLUMNS = {
    "BARLEY": "barley",
    "DATES": "dates",
    "CUSCUTA": "cuscuta",
    "CRESS": "cress",
    "SESAME": "sesame",
    "WOOL": "wool",
}


def plot_histogram(ax, values: pd.Series, title: str):
    values = pd.to_numeric(values, errors="coerce").dropna()
    ax.hist(values, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df_1overx = load_dataset(SRC).iloc[1:]  # first row holds unit labels, not data

    rows = []
    for commodity, column in COMMODITY_COLUMNS.items():
        over_x_values = pd.to_numeric(df_1overx[column], errors="coerce")
        x_values = 1 / over_x_values

        rows.append({"commodity": commodity, "format": "X (quantity per silver)", **summary_stats(x_values)})
        rows.append({"commodity": commodity, "format": "1/X (silver per quantity)", **summary_stats(over_x_values)})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plot_histogram(axes[0], x_values, "X (quantity per silver)")
        plot_histogram(axes[1], over_x_values, "1/X (silver per quantity)")
        fig.suptitle(commodity)
        fig.tight_layout()

        safe_name = commodity.replace("/", "-").replace(" ", "_")
        out_path = GRAPH_DIR / f"{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "structure_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'structure_summary.csv'}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

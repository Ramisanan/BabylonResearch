"""CLI: look at the basic distribution structure of the X and 1/X data
(range, mean, std, skew, kurtosis) for every commodity, before deciding
on a normalization technique.

Saves:
  - structure_summary.csv           (stats table)
  - graphs/<commodity>.png          (X vs 1/X histograms, side by side)
into "Data Structure Analysis for Vanderspek Data/".

Usage:
    python scripts/explore_structure.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import COMMODITIES, convert_x_to_1_over_x, summary_stats
from babylon_prices.io import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data" / "Babylon_vanderspek_X.xlsx"
OUT_DIR = ROOT / "Data Structure Analysis for Vanderspek Data"
GRAPH_DIR = OUT_DIR / "graphs"


def plot_histogram(ax, values: pd.Series, title: str):
    values = pd.to_numeric(values, errors="coerce").dropna()
    ax.hist(values, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df_x = load_dataset(SRC)
    df_1overx = convert_x_to_1_over_x(df_x)

    rows = []
    for commodity, (x_col, unit, over_x_col) in COMMODITIES.items():
        x_values = df_x[x_col]
        over_x_values = df_1overx[over_x_col]

        rows.append({"commodity": commodity, "format": "X (qty per shekel)", **summary_stats(x_values)})
        rows.append({"commodity": commodity, "format": f"1/X ({unit})", **summary_stats(over_x_values)})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plot_histogram(axes[0], x_values, "X format (quantity per shekel)")
        plot_histogram(axes[1], over_x_values, f"1/X format ({unit})")
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

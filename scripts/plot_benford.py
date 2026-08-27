"""CLI: plot observed vs. Benford-expected leading-digit distributions for
every commodity (and the ALL aggregate), X format and 1/X format side by side.

Reads the CSVs produced by run_benford.py and saves one PNG per
commodity into a "graphs" subfolder of the output folder.

Usage:
    python scripts/run_benford.py   # first, to (re)generate the CSVs
    python scripts/plot_benford.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import conformity_from_distribution

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "Benford Law Data Output for Vanderspek Data"
GRAPH_DIR = OUT_DIR / "graphs"

FORMATS = [
    ("benford_X_digit_distribution.csv", "X format (quantity per shekel)"),
    ("benford_1overX_digit_distribution.csv", "1/X format (shekel per unit)"),
]


def plot_on_axis(ax, dist: pd.DataFrame, title: str):
    stats = conformity_from_distribution(dist)

    ax.bar(dist["digit"], dist["observed_pct"], color="#4C72B0", label="Observed")
    ax.plot(dist["digit"], dist["expected_pct"], color="#C44E52",
            marker="o", linewidth=2, label="Benford expected")
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Leading digit")
    ax.set_ylabel("Proportion")
    ax.set_title(f"{title}\n$R^2$={stats['r_squared']:.3f}, MAD={stats['mad']:.4f} ({stats['mad_conformity']}), n={stats['n']}")
    ax.legend()


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    tables = [pd.read_csv(OUT_DIR / csv_name) for csv_name, _label in FORMATS]
    commodities = list(tables[0]["commodity"].drop_duplicates())

    for commodity in commodities:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        for ax, table, (_csv_name, format_label) in zip(axes, tables, FORMATS):
            dist = table[table["commodity"] == commodity]
            plot_on_axis(ax, dist, format_label)

        display_name = "ALL 6 COMMODITIES (POOLED)" if commodity == "ALL" else commodity
        fig.suptitle(display_name)
        fig.tight_layout()

        safe_name = commodity.replace("/", "-").replace(" ", "_")
        out_path = GRAPH_DIR / f"{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""CLI: plot observed vs. Benford-expected leading-digit distributions for
every commodity (and the ALL aggregate), X format and 1/X format side by side.

Reads the CSVs produced by run_benford.py and saves one PNG per
commodity into a "graphs" subfolder of the output folder.

Usage:
    python scripts/run_benford.py   # first, to (re)generate the CSVs
    python scripts/plot_benford.py [output_dir]

output_dir defaults to "Benford Law Data Output for Vanderspek Data"; pass
another Benford output folder (e.g. one produced by run_benford_appendix.py)
to plot that dataset instead.
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
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Benford Law Data Output for Vanderspek Data"
GRAPH_DIR = OUT_DIR / "graphs"

DIST_CSV_NAMES = ["benford_X_digit_distribution.csv", "benford_1overX_digit_distribution.csv"]


def format_labels() -> list[str]:
    """The two format labels (X, then 1/X), as written into benford_summary.csv
    by whichever run_benford*.py script produced this output folder."""
    summary = pd.read_csv(OUT_DIR / "benford_summary.csv")
    return list(summary["format"].drop_duplicates())


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

    tables = [pd.read_csv(OUT_DIR / csv_name) for csv_name in DIST_CSV_NAMES]
    labels = format_labels()
    commodities = list(tables[0]["commodity"].drop_duplicates())

    for commodity in commodities:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        for ax, table, format_label in zip(axes, tables, labels):
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

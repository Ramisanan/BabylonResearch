"""CLI: residual analysis (observed proportion - Benford expected proportion)
for every commodity and leading digit 1-9, X format and 1/X format.

Uses the exact counts/proportions already computed from the data by
run_benford.py (not estimates read off a graph). Residual near 0 = good
match, positive = observed more than expected, negative = observed less
than expected.

Reads the CSVs produced by run_benford.py and saves the residual tables
and bar charts into a "residual" subfolder of the graphs directory.

Usage:
    python scripts/run_benford.py   # first, to (re)generate the CSVs
    python scripts/residual_benford.py [output_dir]

output_dir defaults to "Benford Law Data Output for Vanderspek Data"; pass
another Benford output folder (e.g. one produced by run_benford_appendix.py)
to compute residuals for that dataset instead.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import residual_table

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Benford Law Data Output for Vanderspek Data"
RESIDUAL_DIR = OUT_DIR / "graphs" / "residual"

DIST_FILES = [
    ("benford_X_digit_distribution.csv", "residual_X.csv"),
    ("benford_1overX_digit_distribution.csv", "residual_1overX.csv"),
]


def format_labels() -> list[str]:
    """The two format labels (X, then 1/X), as written into benford_summary.csv
    by whichever run_benford*.py script produced this output folder."""
    summary = pd.read_csv(OUT_DIR / "benford_summary.csv")
    return list(summary["format"].drop_duplicates())


def plot_on_axis(ax, dist: pd.DataFrame, title: str):
    colors = ["#55A868" if r > 0 else "#C44E52" for r in dist["residual"]]
    ax.bar(dist["digit"], dist["residual"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Leading digit")
    ax.set_ylabel("Residual (observed - expected)")
    ax.set_title(title)


def main():
    RESIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    residual_tables = []
    for (csv_name, residual_csv_name), format_label in zip(DIST_FILES, format_labels()):
        dist = pd.read_csv(OUT_DIR / csv_name)
        residual = residual_table(dist)
        residual.insert(0, "format", format_label)
        residual.to_csv(RESIDUAL_DIR / residual_csv_name, index=False)
        residual_tables.append((residual, format_label))
        print(f"Saved: {RESIDUAL_DIR / residual_csv_name}")

    combined = pd.concat([t for t, _ in residual_tables], ignore_index=True)
    combined_path = RESIDUAL_DIR / "residual_summary.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path}")

    commodities = list(residual_tables[0][0]["commodity"].drop_duplicates())
    for commodity in commodities:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        for ax, (residual, format_label) in zip(axes, residual_tables):
            dist = residual[residual["commodity"] == commodity]
            plot_on_axis(ax, dist, format_label)

        display_name = "ALL 6 COMMODITIES (POOLED)" if commodity == "ALL" else commodity
        fig.suptitle(f"Benford Residuals -- {display_name}")
        fig.tight_layout()

        safe_name = commodity.replace("/", "-").replace(" ", "_")
        out_path = RESIDUAL_DIR / f"{safe_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print()
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()

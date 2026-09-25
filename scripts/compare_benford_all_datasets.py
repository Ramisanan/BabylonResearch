"""CLI: compare the pooled ("ALL" commodities) Benford leading-digit
distribution across all three price datasets -- Vanderspek, Appendix, and
Pirngruber -- X format and 1/X format side by side.

Reads each dataset's benford_summary.csv (for format labels) and
benford_X_digit_distribution.csv / benford_1overX_digit_distribution.csv
(for the "ALL" row per digit), and saves a comparison table + grouped bar
chart into "All Data Comparison/".

Usage:
    python scripts/compare_benford_all_datasets.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices.benford import expected_proportions

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "All Data Comparison"

DATASETS = [
    ("Vanderspek", ROOT / "Benford Law Data Output for Vanderspek Data"),
    ("Appendix", ROOT / "Benford Law Data Output for Appendix Data"),
    ("Pirngruber", ROOT / "Benford Law Data Output for Pirngruber Data"),
]

DIST_FILES = ["benford_X_digit_distribution.csv", "benford_1overX_digit_distribution.csv"]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_all_row(dataset_dir: Path, dist_file: str) -> pd.DataFrame:
    dist = pd.read_csv(dataset_dir / dist_file)
    return dist[dist["commodity"] == "ALL"].sort_values("digit")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    per_format = [[], []]  # per_format[0] = X across datasets, per_format[1] = 1/X across datasets

    for dataset_name, dataset_dir in DATASETS:
        format_labels = list(pd.read_csv(dataset_dir / "benford_summary.csv")["format"].drop_duplicates())
        for i, dist_file in enumerate(DIST_FILES):
            all_row = load_all_row(dataset_dir, dist_file)
            tagged = all_row.copy()
            tagged.insert(0, "dataset", dataset_name)
            tagged.insert(1, "format", format_labels[i])
            rows.append(tagged)
            per_format[i].append((dataset_name, all_row))

    combined = pd.concat(rows, ignore_index=True)
    combined_path = OUT_DIR / "benford_all_datasets_ALL_commodities.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved: {combined_path}")

    digits = list(range(1, 10))
    expected = expected_proportions()
    n_datasets = len(DATASETS)
    bar_width = 0.8 / n_datasets

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["X format (pooled 'ALL' commodities)", "1/X format (pooled 'ALL' commodities)"]

    for ax, title, series in zip(axes, titles, per_format):
        for i, (dataset_name, all_row) in enumerate(series):
            offset = (i - (n_datasets - 1) / 2) * bar_width
            positions = [d + offset for d in digits]
            ax.bar(positions, all_row["observed_pct"], width=bar_width,
                   color=COLORS[i % len(COLORS)], label=f"{dataset_name} (n={int(all_row['count'].sum())})")

        ax.plot(digits, [expected[d] for d in digits], color="black",
                marker="o", linewidth=2, linestyle="--", label="Benford expected")
        ax.set_xticks(digits)
        ax.set_xlabel("Leading digit")
        ax.set_ylabel("Proportion")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Benford's Law -- Observed Leading-Digit Distribution Across Datasets")
    fig.tight_layout()

    out_path = OUT_DIR / "benford_all_datasets_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

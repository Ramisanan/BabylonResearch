"""CLI: run Benford's Law (first-digit) analysis on both the X format
(quantity per shekel) and 1/X format (shekel per unit) price data, and
save the results to "Benford Law Data Output for Vanderspek Data/".

Usage:
    python scripts/run_benford.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import COMMODITIES, convert_x_to_1_over_x, digit_distribution, conformity_stats
from babylon_prices.io import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data" / "Babylon_vanderspek_X.xlsx"
OUT_DIR = ROOT / "Benford Law Data Output for Vanderspek Data"


def run_for_all_commodities(df: pd.DataFrame, value_columns: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """value_columns: {commodity_name: column_to_read_values_from}
    Returns (digit_distribution_table, summary_table) across all commodities + an ALL aggregate."""
    dist_rows = []
    summary_rows = []
    all_values = []

    for commodity, column in value_columns.items():
        values = pd.to_numeric(df[column], errors="coerce")
        all_values.append(values)

        dist = digit_distribution(values)
        dist.insert(0, "commodity", commodity)
        dist_rows.append(dist)

        stats = conformity_stats(values)
        summary_rows.append({"commodity": commodity, **stats})

    combined = pd.concat(all_values)
    dist = digit_distribution(combined)
    dist.insert(0, "commodity", "ALL")
    dist_rows.append(dist)

    stats = conformity_stats(combined)
    summary_rows.append({"commodity": "ALL", **stats})

    return pd.concat(dist_rows, ignore_index=True), pd.DataFrame(summary_rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df_x = load_dataset(SRC)
    df_1overx = convert_x_to_1_over_x(df_x)

    x_columns = {commodity: interp_col for commodity, (interp_col, _unit, _out_col) in COMMODITIES.items()}
    dist_x, summary_x = run_for_all_commodities(df_x, x_columns)
    dist_x.to_csv(OUT_DIR / "benford_X_digit_distribution.csv", index=False)

    over_x_columns = {commodity: out_col for commodity, (_interp_col, _unit, out_col) in COMMODITIES.items()}
    dist_1overx, summary_1overx = run_for_all_commodities(df_1overx, over_x_columns)
    dist_1overx.to_csv(OUT_DIR / "benford_1overX_digit_distribution.csv", index=False)

    summary_x.insert(0, "format", "X (qty per shekel)")
    summary_1overx.insert(0, "format", "1/X (shekel per unit)")
    summary = pd.concat([summary_x, summary_1overx], ignore_index=True)
    summary.to_csv(OUT_DIR / "benford_summary.csv", index=False)

    print(f"Saved to: {OUT_DIR}")
    print(" - benford_X_digit_distribution.csv")
    print(" - benford_1overX_digit_distribution.csv")
    print(" - benford_summary.csv")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

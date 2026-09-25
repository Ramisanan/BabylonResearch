"""CLI: run Benford's Law (first-digit) analysis on the appendix commodity
price data (Data/appendix.xlsx), same methodology and X/1-X convention as
run_benford.py -- X = quantity of goods per unit silver, 1/X = silver per
unit quantity (the price) -- and save the results to
"Benford Law Data Output for Appendix Data/".

The sheet's raw columns are already silver-per-quantity (e.g. "gramme
silver /100 litre"), i.e. the 1/X side; X is their reciprocal. This matches
the direction used for the Vanderspek data, where X is quantity-per-shekel
and 1/X is shekel-per-quantity.

The sheet's first data row holds unit labels ("gramme silver /100 litre",
etc.), not a value, so it's dropped before analysis.

Usage:
    python scripts/run_benford_appendix.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import digit_distribution, conformity_stats
from babylon_prices.io import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data" / "appendix.xlsx"
OUT_DIR = ROOT / "Benford Law Data Output for Appendix Data"

COMMODITY_COLUMNS = {
    "BARLEY": "barley",
    "DATES": "dates",
    "CUSCUTA": "cuscuta",
    "CRESS": "cress",
    "SESAME": "sesame",
    "WOOL": "wool",
}


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

    df_1overx = load_dataset(SRC).iloc[1:]  # raw columns are silver per quantity, i.e. 1/X

    dist_1overx, summary_1overx = run_for_all_commodities(df_1overx, COMMODITY_COLUMNS)
    dist_1overx.to_csv(OUT_DIR / "benford_1overX_digit_distribution.csv", index=False)

    df_x = df_1overx.copy()
    for column in COMMODITY_COLUMNS.values():
        df_x[column] = 1 / pd.to_numeric(df_1overx[column], errors="coerce")
    dist_x, summary_x = run_for_all_commodities(df_x, COMMODITY_COLUMNS)
    dist_x.to_csv(OUT_DIR / "benford_X_digit_distribution.csv", index=False)

    summary_x.insert(0, "format", "X (quantity per silver)")
    summary_1overx.insert(0, "format", "1/X (silver per quantity)")
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

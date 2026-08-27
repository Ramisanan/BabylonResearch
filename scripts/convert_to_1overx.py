"""CLI: convert Data/Babylon_vanderspek_X.xlsx (X format) to 1/X format.

Usage:
    python scripts/convert_to_1overx.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import COMMODITIES, convert_x_to_1_over_x
from babylon_prices.io import load_dataset, save_dataset

SRC = Path(__file__).resolve().parent.parent / "Data" / "Babylon_vanderspek_X.xlsx"
OUT = Path(__file__).resolve().parent.parent / "Data" / "Babylon_vanderspek_1overX.xlsx"


def main():
    df = load_dataset(SRC)
    df = convert_x_to_1_over_x(df)
    save_dataset(df, OUT)

    print(f"Saved: {OUT}")
    for raw_col, (_interp_col, unit, out_col) in COMMODITIES.items():
        n = df[out_col].notna().sum()
        print(f"{raw_col:25s} -> {out_col:22s} ({unit}) : {n} values")


if __name__ == "__main__":
    main()

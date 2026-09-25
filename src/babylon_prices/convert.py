"""Convert Babylonian price data from X format (quantity per shekel)
to 1/X format (shekel per unit quantity)."""

import pandas as pd

# For each commodity: which column holds the clean X value, what unit
# the converted price will be in, and what to name the new column.
COMMODITIES = {
    "BARLEY": ("Bar-interpretatie", "shekel per liter", "Bar-shekel_per_l"),
    "DATES": ("Dat-interpretatie", "shekel per liter", "Dat-shekel_per_l"),
    "MUSTARD": ("Mus-interpretatie", "shekel per liter", "Mus-shekel_per_l"),
    "CRESS": ("Cre-interpretatie", "shekel per liter", "Cre-shekel_per_l"),
    "SESAME": ("Ses-interpretatie", "shekel per liter", "Ses-shekel_per_l"),
    "WOOL 0.5 kg p. Shekel": ("Wo-interpretatie", "shekel per 0.5kg (mina)", "Wo-shekel_per_mina"),
}


def convert_x_to_1_over_x(df: pd.DataFrame, commodities: dict = None) -> pd.DataFrame:
    """Return a copy of df with a 1/X (shekel per unit) column added
    next to each commodity's interpretation column.

    commodities defaults to COMMODITIES; pass another {commodity: (x_column,
    unit, new_column)} mapping to reuse this on a workbook with different
    column names but the same X/1-X structure."""
    df = df.copy()
    commodities = COMMODITIES if commodities is None else commodities

    for x_column, unit, new_column in commodities.values():
        # Read the X values (e.g. 27 = 27 liters per shekel) as numbers.
        # Anything that isn't a clean number (blank, "?", etc.) becomes NaN.
        x_values = pd.to_numeric(df[x_column], errors="coerce")

        # 1/X = shekels needed to buy ONE unit (e.g. 1/27 = 0.037 shekel per liter).
        prices = 1 / x_values

        # Place the new column right after the X column it came from,
        # instead of tacking it onto the far right of the sheet.
        position = df.columns.get_loc(x_column) + 1
        df.insert(position, new_column, prices)

    return df

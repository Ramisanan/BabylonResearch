"""Basic distribution structure of a numeric series -- the stats you'd
look at before picking a normalization technique (range, skew, spread)."""

import pandas as pd


def summary_stats(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()

    return {
        "n": len(values),
        "min": values.min(),
        "max": values.max(),
        "range": values.max() - values.min() if len(values) else float("nan"),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "skew": values.skew(),
        "kurtosis": values.kurt(),
    }

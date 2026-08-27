"""Per-commodity z-score normalization:

    S_tilde_i(t) = (S_i(t) - mean(S_i)) / std(S_i)

Each commodity is normalized against its OWN mean and std (not a global
mean/std), so this is done separately per commodity."""

import pandas as pd


def zscore_normalize(values: pd.Series) -> pd.Series:
    """Normalize one commodity's values against its own mean and std."""
    values = pd.to_numeric(values, errors="coerce")
    return (values - values.mean()) / values.std()

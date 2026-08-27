"""Benford's Law (first-digit) analysis for a series of numeric values."""

import math

import pandas as pd
from scipy.stats import chisquare

DIGITS = list(range(1, 10))


def expected_proportions() -> dict:
    """Benford's expected proportion for each leading digit 1-9."""
    return {d: math.log10(1 + 1 / d) for d in DIGITS}


def leading_digit(x: float):
    """First significant digit of a positive number, or None if x isn't usable
    (NaN, zero, or negative -- Benford's law only applies to positive magnitudes)."""
    if pd.isna(x) or x <= 0:
        return None
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def digit_distribution(values: pd.Series) -> pd.DataFrame:
    """Observed vs. Benford-expected leading-digit distribution for values."""
    digits = values.apply(leading_digit).dropna()
    n = len(digits)
    counts = digits.value_counts().reindex(DIGITS, fill_value=0)
    expected = expected_proportions()

    return pd.DataFrame({
        "digit": DIGITS,
        "count": [int(counts[d]) for d in DIGITS],
        "observed_pct": [counts[d] / n if n else float("nan") for d in DIGITS],
        "expected_pct": [expected[d] for d in DIGITS],
    })


def mad_conformity(mad: float) -> str:
    """Nigrini's mean-absolute-deviation conformity bands for first-digit tests."""
    if mad < 0.006:
        return "close conformity"
    if mad < 0.012:
        return "acceptable conformity"
    if mad < 0.015:
        return "marginally acceptable"
    return "nonconformity"


def conformity_from_distribution(dist: pd.DataFrame) -> dict:
    """n, mean absolute deviation (+ conformity band), R^2 of observed vs.
    expected proportions, and a chi-square goodness-of-fit test -- computed
    from an already-built digit_distribution() table."""
    n = int(dist["count"].sum())

    observed = dist["observed_pct"]
    expected = dist["expected_pct"]
    mad = (observed - expected).abs().mean()

    if n > 0:
        ss_res = ((observed - expected) ** 2).sum()
        ss_tot = ((observed - observed.mean()) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        chi2, p_value = chisquare(dist["count"], f_exp=expected * n)
    else:
        r_squared, chi2, p_value = float("nan"), float("nan"), float("nan")

    return {
        "n": n,
        "mad": mad,
        "mad_conformity": mad_conformity(mad) if n else "no data",
        "r_squared": r_squared,
        "chi2": chi2,
        "p_value": p_value,
    }


def conformity_stats(values: pd.Series) -> dict:
    """Same as conformity_from_distribution(), built directly from raw values."""
    return conformity_from_distribution(digit_distribution(values))

"""Second-order Detrended Fluctuation Analysis (DFA-2), as used in Romero
et al. (2010) to quantify long-term power-law correlations in Babylonian
commodity price time series.

Method (matches the paper):
  1. Build the "profile": the cumulative sum of (price - mean price).
  2. Split the profile into non-overlapping windows of length n (from both
     ends of the series, to use all the data).
  3. In each window, fit and subtract a 2nd-order (quadratic) polynomial
     trend -- this is what makes it DFA-2, removing constant AND linear
     trends, per Kantelhardt et al.
  4. F(n) = root-mean-square of the detrended residuals, averaged over all
     windows.
  5. Repeat for a range of window sizes n. A power law F(n) ~ n^alpha
     indicates scale-invariant (self-similar) correlations:
       alpha = 0.5 -> uncorrelated (white noise)
       alpha < 0.5 -> anti-correlated
       alpha > 0.5 -> positively correlated (the stronger, the larger alpha)
  6. alpha is estimated as the slope of log F(n) vs log n."""

import numpy as np
import pandas as pd

ROMAN_MONTHS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}


def parse_month(raw) -> float:
    """Babylonian month label -> a sortable number within the year.
    Leap months (trailing 'B', e.g. 'VIB', 'XIIB') are placed just after
    their base month. Ambiguous entries (ranges like 'I-IV', or 'I of
    XIIB') return None and are dropped -- there's no single month to
    assign them to."""
    s = str(raw).strip()
    if not s or "-" in s or " of " in s:
        return None
    s = s.rstrip("?").strip()
    if s.endswith("B") and s[:-1] in ROMAN_MONTHS:
        return ROMAN_MONTHS[s[:-1]] + 0.5
    return ROMAN_MONTHS.get(s)


def build_monthly_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Chronologically ordered, monthly-averaged price series for one
    commodity (matches the paper's "monthly averaged prices")."""
    year = pd.to_numeric(df["YEAR"], errors="coerce")
    month = df["MON"].apply(parse_month)
    value = pd.to_numeric(df[column], errors="coerce")

    table = pd.DataFrame({"year": year, "month": month, "value": value}).dropna()
    monthly = table.groupby(["year", "month"])["value"].mean()
    return monthly.sort_index()


def build_annual_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Chronologically ordered, annually-averaged price series for one
    commodity (matches the paper's "annually averaged prices")."""
    year = pd.to_numeric(df["YEAR"], errors="coerce")
    value = pd.to_numeric(df[column], errors="coerce")

    table = pd.DataFrame({"year": year, "value": value}).dropna()
    annual = table.groupby("year")["value"].mean()
    return annual.sort_index()


def _window_fluctuation(profile: np.ndarray, n: int, order: int) -> float:
    N = len(profile)
    n_segments = N // n
    if n_segments < 1:
        return float("nan")

    t = np.arange(n)
    variances = []
    for v in range(n_segments):  # forward segments
        segment = profile[v * n:(v + 1) * n]
        trend = np.polyval(np.polyfit(t, segment, order), t)
        variances.append(np.mean((segment - trend) ** 2))
    for v in range(n_segments):  # backward segments, so no data is wasted
        segment = profile[N - (v + 1) * n: N - v * n]
        trend = np.polyval(np.polyfit(t, segment, order), t)
        variances.append(np.mean((segment - trend) ** 2))

    return float(np.sqrt(np.mean(variances)))


def dfa(x: np.ndarray, order: int = 2, n_min: int = None, n_max: int = None, n_points: int = 20) -> pd.DataFrame:
    """Fluctuation function F(n) for a range of window sizes n."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    profile = np.cumsum(x - x.mean())

    if n_min is None:
        n_min = 4 * (order + 1)  # need enough points per window to fit a stable polynomial
    if n_max is None:
        n_max = N // 4

    candidates = np.unique(np.round(np.logspace(np.log10(n_min), np.log10(n_max), n_points)).astype(int))
    candidates = candidates[(candidates >= n_min) & (candidates <= n_max)]

    f_values = [_window_fluctuation(profile, n, order) for n in candidates]
    return pd.DataFrame({"n": candidates, "F": f_values}).dropna()


def fit_alpha(n_values: np.ndarray, f_values: np.ndarray) -> dict:
    """Scaling exponent alpha = slope of log F(n) vs log n (least squares)."""
    n_values, f_values = np.asarray(n_values, dtype=float), np.asarray(f_values, dtype=float)
    mask = np.isfinite(f_values) & (f_values > 0) & (n_values > 0)
    log_n, log_f = np.log10(n_values[mask]), np.log10(f_values[mask])

    alpha, intercept = np.polyfit(log_n, log_f, 1)
    log_f_fit = alpha * log_n + intercept
    ss_res = np.sum((log_f - log_f_fit) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"alpha": alpha, "r_squared": r_squared, "n_windows": int(mask.sum())}

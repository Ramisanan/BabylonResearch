"""Check whether the recorded price quantities carry a base-60 (sexagesimal)
fingerprint: a preference for "regular" numbers (only prime factors 2, 3, 5)
and fractional parts that are clean halves/quarters/eighths/thirds rather
than arbitrary decimals."""

import numpy as np
import pandas as pd

from .convert import COMMODITIES

REGULAR_PRIMES = (2, 3, 5)
# scale factors used to catch halves/quarters/fifths/eighths/tenths/twentieths
# before testing whether what's left is 2,3,5-smooth
REGULAR_SCALES = (1, 2, 4, 5, 8, 10, 20, 60)

# fractions Babylonian scribes commonly produced via halving/thirding a
# sexagesimal quantity; used to classify the fractional part of a value
BASE60_FRACTION_DENOMS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 60)


def is_regular_number(n: float, tol: float = 1e-9) -> bool:
    """True if n (after clearing halves/quarters/fifths/etc.) has only
    2, 3, 5 as prime factors -- i.e. its reciprocal terminates in base 60."""
    if n <= 0:
        return False
    for scale in REGULAR_SCALES:
        m = n * scale
        if abs(m - round(m)) < tol:
            m = round(m)
            if m == 0:
                continue
            x = m
            for p in REGULAR_PRIMES:
                while x % p == 0:
                    x //= p
            if x == 1:
                return True
    return False


def nearest_base60_fraction(frac: float, tol: float = 0.01):
    """Return (numerator, denominator) if frac matches a base-60-friendly
    fraction within tol, else None. Picks the simplest (smallest denominator) match."""
    best = None
    for d in BASE60_FRACTION_DENOMS:
        num = round(frac * d)
        if num == 0:
            continue
        if abs(num / d - frac) < tol:
            if best is None or d < best[1]:
                best = (num, d)
    return best


def _all_interpretation_values(df: pd.DataFrame) -> pd.Series:
    cols = [interp_col for interp_col, _unit, _out_col in COMMODITIES.values()]
    return pd.concat([pd.to_numeric(df[c], errors="coerce") for c in cols]).dropna()


def analyze(df: pd.DataFrame) -> dict:
    """Run the regular-number and base-60-fraction checks and return a report dict."""
    report = {"by_commodity": {}, "fraction_breakdown": {}}

    all_vals = []
    for raw_col, (interp_col, _unit, _out_col) in COMMODITIES.items():
        vals = pd.to_numeric(df[interp_col], errors="coerce").dropna()
        all_vals.append(vals)
        n = len(vals)
        n_regular = int(vals.apply(is_regular_number).sum()) if n else 0
        report["by_commodity"][raw_col] = {
            "n": n,
            "pct_regular": n_regular / n if n else float("nan"),
        }

    all_vals = pd.concat(all_vals)
    report["overall_pct_regular"] = float(all_vals.apply(is_regular_number).mean())

    frac_parts = (all_vals - np.floor(all_vals)).round(4)
    frac_parts = frac_parts[frac_parts != 0]
    matches = frac_parts.apply(nearest_base60_fraction)

    report["n_fractional"] = len(frac_parts)
    report["n_fractional_matched"] = int(matches.notna().sum())
    report["fraction_breakdown"] = (
        matches.dropna().apply(lambda t: f"{t[0]}/{t[1]}").value_counts().to_dict()
    )

    return report

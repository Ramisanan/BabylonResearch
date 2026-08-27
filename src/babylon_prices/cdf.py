"""Empirical cumulative distribution function of a 1-D array of values."""

import numpy as np


def empirical_cdf(values: np.ndarray):
    """Sorted values and F(x) = P(X <= x), computed from values alone."""
    values = np.sort(np.asarray(values))
    n = len(values)
    cdf = (np.arange(1, n + 1)) / n
    return values, cdf

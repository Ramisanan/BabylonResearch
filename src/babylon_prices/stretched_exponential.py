"""Fit a stretched exponential to the cumulative tail distribution of
z-score normalized values:

    P(z) = P(Z >= z)  ~  exp( -(z/tau)^delta )

P(z) is the empirical survival function, computed from the FULL sample
(negative z-scores included, never dropped or renormalized away). The
stretched-exponential model is only defined for z >= 0, so fitting is
restricted to that range -- but the P(z) values being fit still come from
the full-sample CCDF, not a CCDF recomputed on the z >= 0 subset alone."""

import numpy as np
from scipy.optimize import curve_fit


def empirical_ccdf(values: np.ndarray):
    """Sorted values (all of them, negative included) and their empirical
    survival probability P(Z >= z), computed against the full sample size."""
    values = np.sort(np.asarray(values))
    n = len(values)
    ccdf = (n - np.arange(n)) / n  # fraction of the FULL sample >= values[i]
    return values, ccdf


def restrict_to_fit_range(s: np.ndarray, p: np.ndarray, min_z: float = 0.0):
    """Keep only the points with s >= min_z for fitting. The P values are
    left untouched (still computed from the full sample) -- only which
    points participate in the fit is restricted."""
    mask = s >= min_z
    return s[mask], p[mask]


def stretched_exponential(s, tau, delta):
    return np.exp(-((s / tau) ** delta))


def stretched_exponential_with_amplitude(s, amplitude, tau, delta):
    """P(z) ~ A * exp(-(z/tau)^delta) -- the free-amplitude version of eq. (1)'s
    "~" (proportional to), which doesn't force P(0) = 1 the way the plain
    model does. Useful when the fit range's starting P(z) isn't 1."""
    return amplitude * np.exp(-((s / tau) ** delta))


def _r_squared_log(p: np.ndarray, p_fit: np.ndarray) -> float:
    log_p, log_p_fit = np.log(p), np.log(p_fit)
    ss_res = np.sum((log_p - log_p_fit) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_to_points(s: np.ndarray, p: np.ndarray) -> dict:
    """Fit tau, delta to already-computed (s, P(s)) tail points.
    Returns fitted params plus R^2 of the fit in log(P) space."""
    tau0 = np.median(s) if len(s) else 1.0
    (tau, delta), _ = curve_fit(
        stretched_exponential, s, p,
        p0=[tau0, 1.0], bounds=([1e-6, 1e-6], [np.inf, np.inf]), maxfev=10000,
    )

    p_fit = stretched_exponential(s, tau, delta)
    return {"tau": tau, "delta": delta, "n": len(s), "r_squared": _r_squared_log(p, p_fit)}


def fit_to_points_with_amplitude(s: np.ndarray, p: np.ndarray) -> dict:
    """Same as fit_to_points, but also fits a free amplitude A in
    P(z) ~ A * exp(-(z/tau)^delta), so the curve isn't forced through P=1
    at the start of the fit range."""
    tau0 = np.median(s) if len(s) else 1.0
    amplitude0 = p[0] if len(p) else 1.0
    (amplitude, tau, delta), _ = curve_fit(
        stretched_exponential_with_amplitude, s, p,
        p0=[amplitude0, tau0, 1.0],
        bounds=([1e-6, 1e-6, 1e-6], [np.inf, np.inf, np.inf]), maxfev=10000,
    )

    p_fit = stretched_exponential_with_amplitude(s, amplitude, tau, delta)
    return {
        "amplitude": amplitude, "tau": tau, "delta": delta,
        "n": len(s), "r_squared": _r_squared_log(p, p_fit),
    }


def fit_stretched_exponential(values: np.ndarray, fit_min_z: float = 0.0) -> dict:
    """CCDF computed from the full sample; only points with z >= fit_min_z
    are used for the fit itself."""
    s, p = empirical_ccdf(values)
    s_fit, p_fit = restrict_to_fit_range(s, p, fit_min_z)
    result = fit_to_points(s_fit, p_fit)
    result["fit_min_z"] = fit_min_z
    return result

"""CLI: for each commodity SEPARATELY, z-score normalize (own mean/std) and
compute its own empirical CCDF P(z) = P(Z >= z) from its FULL sample --
negative z-scores included, never dropped or renormalized away. Plot all
six commodities' full curves together (negative and positive z), then fit
ONE common stretched exponential:

    P(z) ~ exp( -(z/tau)^delta )

restricted only to the z >= 0 portion of those points, since the model
isn't defined for negative z. The P(z) values used in that fit still come
from the full-sample CCDF -- only which points participate in the fit is
restricted, not how P(z) itself was calculated.

Raw values are never pooled/mixed across commodities -- each commodity's
z-score and CCDF are computed independently. The "common" fit is obtained
by fitting to the union of the six commodities' z >= 0 points, i.e. a
data-collapse fit across the six independent curves. Done separately for
the X and 1/X formats.

Saves into "Zscore Normalized Data Output for Vanderspek Data/":
  - stretched_exponential_fit.csv            (tau, delta, n, R^2, fit_min_z per format)
  - stretched_exponential_tail_points_X.csv       (per-commodity z, P(z), full sample)
  - stretched_exponential_tail_points_1overX.csv
  - graphs/stretched_exponential_fit.png     (full per-commodity CCDFs + fit, side by side)

Usage:
    python scripts/fit_stretched_exponential.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import (
    COMMODITIES, convert_x_to_1_over_x, zscore_normalize,
    empirical_ccdf, stretched_exponential, stretched_exponential_with_amplitude,
    fit_to_points, fit_to_points_with_amplitude, restrict_to_fit_range,
)
from babylon_prices.io import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data" / "Babylon_vanderspek_X.xlsx"
OUT_DIR = ROOT / "Zscore Normalized Data Output for Vanderspek Data"
GRAPH_DIR = OUT_DIR / "graphs"

FIT_MIN_Z = 0.0  # stretched exponential is only defined for z >= 0; this is
                 # the ONLY place the sample gets restricted -- P(z) itself
                 # is always computed from the full (all-z) sample.


def commodity_tail_points(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """For each commodity: its own z-score (own mean/std) and its own
    empirical CCDF P(z), computed from that commodity's FULL sample
    (negative z included) -- never mixed with any other commodity's raw
    values, and never renormalized to a subset."""
    rows = []
    for commodity, column in column_map.items():
        values = pd.to_numeric(df[column], errors="coerce")
        z = zscore_normalize(values).dropna()
        s, p = empirical_ccdf(z.values)  # this commodity's own full-sample CCDF
        rows.append(pd.DataFrame({"commodity": commodity, "z": s, "P": p}))
    return pd.concat(rows, ignore_index=True)


def plot_format(ax, tail_points: pd.DataFrame, fit: dict, fit_a: dict, title: str):
    for commodity, group in tail_points.groupby("commodity", sort=False):
        ax.semilogy(group["z"], group["P"], ".", markersize=4, label=commodity, alpha=0.7)

    # fit curves are only drawn (and were only fit) over z >= fit_min_z
    z_max = tail_points["z"].max()
    z_grid = np.linspace(fit["fit_min_z"], z_max, 200)

    fit_curve = stretched_exponential(z_grid, fit["tau"], fit["delta"])
    ax.semilogy(z_grid, fit_curve, color="black", linewidth=2,
                label=fr"A=1 fit: $\tau$={fit['tau']:.2f}, $\delta$={fit['delta']:.2f}, "
                      fr"$R^2$={fit['r_squared']:.2f}")

    fit_a_curve = stretched_exponential_with_amplitude(z_grid, fit_a["amplitude"], fit_a["tau"], fit_a["delta"])
    ax.semilogy(z_grid, fit_a_curve, color="#C44E52", linewidth=2, linestyle="--",
                label=fr"free-A fit: A={fit_a['amplitude']:.2f}, $\tau$={fit_a['tau']:.2f}, "
                      fr"$\delta$={fit_a['delta']:.2f}, $R^2$={fit_a['r_squared']:.2f}")

    ax.axvline(fit["fit_min_z"], color="gray", linestyle=":", linewidth=1)

    ax.set_xlabel("z-score  (full sample, negative included)")
    ax.set_ylabel("P(z) = P(Z >= z)  (log scale)")
    ax.set_title(f"{title}  (fit range: $z\\geq${fit['fit_min_z']:.0f}, n_fit={fit['n']})")
    ax.legend(fontsize=7)


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df_x = load_dataset(SRC)
    df_1overx = convert_x_to_1_over_x(df_x)

    x_columns = {commodity: interp_col for commodity, (interp_col, _unit, _out_col) in COMMODITIES.items()}
    over_x_columns = {commodity: out_col for commodity, (_interp_col, _unit, out_col) in COMMODITIES.items()}

    tail_x = commodity_tail_points(df_x, x_columns)
    tail_1overx = commodity_tail_points(df_1overx, over_x_columns)

    tail_x.to_csv(OUT_DIR / "stretched_exponential_tail_points_X.csv", index=False)
    tail_1overx.to_csv(OUT_DIR / "stretched_exponential_tail_points_1overX.csv", index=False)
    print(f"Saved: {OUT_DIR / 'stretched_exponential_tail_points_X.csv'} "
          f"(full sample, z range [{tail_x['z'].min():.2f}, {tail_x['z'].max():.2f}])")
    print(f"Saved: {OUT_DIR / 'stretched_exponential_tail_points_1overX.csv'} "
          f"(full sample, z range [{tail_1overx['z'].min():.2f}, {tail_1overx['z'].max():.2f}])")

    # common fit = fit to the union of the six independently-computed tail
    # curves, restricted to z >= FIT_MIN_Z (P(z) itself still comes from
    # the full, unrestricted sample computed above). Two variants:
    #   A=1  : plain eq. (1), forces P(fit_min_z) = 1
    #   free A : P(z) ~ A * exp(-(z/tau)^delta), lets the curve start at
    #            whatever P(fit_min_z) actually is -- the "~" (proportional
    #            to) in eq. (1) read literally, rather than "=".
    s_x, p_x = restrict_to_fit_range(tail_x["z"].values, tail_x["P"].values, FIT_MIN_Z)
    fit_x = fit_to_points(s_x, p_x)
    fit_x["fit_min_z"] = FIT_MIN_Z
    fit_x_a = fit_to_points_with_amplitude(s_x, p_x)
    fit_x_a["fit_min_z"] = FIT_MIN_Z

    s_1overx, p_1overx = restrict_to_fit_range(tail_1overx["z"].values, tail_1overx["P"].values, FIT_MIN_Z)
    fit_1overx = fit_to_points(s_1overx, p_1overx)
    fit_1overx["fit_min_z"] = FIT_MIN_Z
    fit_1overx_a = fit_to_points_with_amplitude(s_1overx, p_1overx)
    fit_1overx_a["fit_min_z"] = FIT_MIN_Z

    summary = pd.DataFrame([
        {"format": "X (qty per shekel)", "model": "A=1", **fit_x},
        {"format": "X (qty per shekel)", "model": "free A", **fit_x_a},
        {"format": "1/X (shekel per unit)", "model": "A=1", **fit_1overx},
        {"format": "1/X (shekel per unit)", "model": "free A", **fit_1overx_a},
    ])
    summary.to_csv(OUT_DIR / "stretched_exponential_fit.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'stretched_exponential_fit.csv'}")
    print(f"Fitting range: z >= {FIT_MIN_Z} (CCDF values themselves computed from the full sample)")
    print()
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_format(axes[0], tail_x, fit_x, fit_x_a, "X format")
    plot_format(axes[1], tail_1overx, fit_1overx, fit_1overx_a, "1/X format")
    fig.suptitle(r"Per-commodity tails, common stretched-exponential fit: $P(z) \sim A \cdot \exp(-(z/\tau)^\delta)$")
    fig.tight_layout()

    out_path = GRAPH_DIR / "stretched_exponential_fit.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

"""CLI: second-order Detrended Fluctuation Analysis (DFA-2) on Babylonian
commodity price time series, following Romero et al. (2010):

  - Build monthly-averaged and annually-averaged price series per commodity
    (chronologically ordered, gaps simply skipped -- the paper works with
    the sequence of available data points, not a fixed calendar grid, and
    shows in their fig. 5 that DFA-2 is robust to heavy data loss).
  - DFA-2 (quadratic detrending) on each series -> F(n) vs window size n.
  - alpha = slope of log F(n) vs log n. alpha=0.5: uncorrelated;
    alpha>0.5: positively correlated (persistent); alpha<0.5: anti-correlated.

The paper ran this on the raw commodity price (our X format: quantity per
shekel) -- that's the primary, paper-matching output here. The 1/X format
is included alongside for comparison, since the rest of this project has
consistently reported both.

Saves into "DFA/":
  - dfa_summary.csv               (alpha, R^2, n_windows per commodity/scale/format)
  - dfa_fluctuation_curves.csv    (raw F(n) vs n points, for reproducibility)
  - graphs/dfa_scaling_X.png      (F(n) vs n, log-log, matches paper fig. 4a)
  - graphs/dfa_scaling_1overX.png
  - graphs/dfa_scaling_side_by_side.png

Usage:
    python scripts/run_dfa.py
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
    COMMODITIES, convert_x_to_1_over_x,
    build_monthly_series, build_annual_series, dfa, fit_alpha,
)
from babylon_prices.io import load_dataset

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Data" / "Babylon_vanderspek_X.xlsx"
OUT_DIR = ROOT / "DFA"
GRAPH_DIR = OUT_DIR / "graphs"

DFA_ORDER = 2


def analyze_format(df: pd.DataFrame, column_map: dict, format_label: str):
    """Returns (summary_rows, curve_rows) for every commodity, monthly and annual."""
    summary_rows = []
    curve_rows = []

    for commodity, column in column_map.items():
        for scale, series in [
            ("monthly", build_monthly_series(df, column)),
            ("annual", build_annual_series(df, column)),
        ]:
            curve = dfa(series.values, order=DFA_ORDER)
            fit = fit_alpha(curve["n"], curve["F"])

            summary_rows.append({
                "format": format_label, "commodity": commodity, "scale": scale,
                "series_length": len(series), **fit,
            })
            for _, row in curve.iterrows():
                curve_rows.append({
                    "format": format_label, "commodity": commodity, "scale": scale,
                    "n": row["n"], "F": row["F"],
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def plot_scaling(ax, curves: pd.DataFrame, summary: pd.DataFrame, scale: str, marker: str, fill: bool):
    for commodity, group in curves[curves["scale"] == scale].groupby("commodity", sort=False):
        alpha_row = summary[(summary["commodity"] == commodity) & (summary["scale"] == scale)].iloc[0]
        facecolor = None if fill else "none"
        line, = ax.plot(group["n"], group["F"], marker=marker, linestyle="none",
                         markersize=5, markerfacecolor=facecolor,
                         label=f"{commodity} ({scale}, $\\alpha$={alpha_row['alpha']:.2f})")

        n_grid = np.array([group["n"].min(), group["n"].max()])
        intercept = np.log10(group["F"].iloc[0]) - alpha_row["alpha"] * np.log10(group["n"].iloc[0])
        f_grid = 10 ** (alpha_row["alpha"] * np.log10(n_grid) + intercept)
        ax.plot(n_grid, f_grid, color=line.get_color(), linewidth=1, alpha=0.6)


def plot_scaling_on_axis(ax, curves: pd.DataFrame, summary: pd.DataFrame, title: str):
    plot_scaling(ax, curves, summary, "monthly", marker="o", fill=True)
    plot_scaling(ax, curves, summary, "annual", marker="s", fill=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time scale n")
    ax.set_ylabel("Fluctuation function F(n)")
    ax.set_title(title)
    ax.legend(fontsize=5, loc="lower right", markerscale=0.7)


def plot_format(curves: pd.DataFrame, summary: pd.DataFrame, format_label: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_scaling_on_axis(ax, curves, summary, f"DFA-2 scaling -- {format_label}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_side_by_side(curves_x, summary_x, curves_1overx, summary_1overx, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_scaling_on_axis(axes[0], curves_x, summary_x, "X format (qty per shekel)")
    plot_scaling_on_axis(axes[1], curves_1overx, summary_1overx, "1/X format (shekel per unit)")
    fig.suptitle("DFA-2 scaling")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df_x = load_dataset(SRC)
    df_1overx = convert_x_to_1_over_x(df_x)

    x_columns = {commodity: interp_col for commodity, (interp_col, _unit, _out_col) in COMMODITIES.items()}
    over_x_columns = {commodity: out_col for commodity, (_interp_col, _unit, out_col) in COMMODITIES.items()}

    summary_x, curves_x = analyze_format(df_x, x_columns, "X (qty per shekel) -- matches paper")
    summary_1overx, curves_1overx = analyze_format(df_1overx, over_x_columns, "1/X (shekel per unit)")

    summary = pd.concat([summary_x, summary_1overx], ignore_index=True)
    curves = pd.concat([curves_x, curves_1overx], ignore_index=True)

    summary.to_csv(OUT_DIR / "dfa_summary.csv", index=False)
    curves.to_csv(OUT_DIR / "dfa_fluctuation_curves.csv", index=False)
    print(f"Saved: {OUT_DIR / 'dfa_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'dfa_fluctuation_curves.csv'}")
    print()
    print(summary.to_string(index=False))

    print()
    for format_label, fmt_summary in summary.groupby("format"):
        for scale, group in fmt_summary.groupby("scale"):
            print(f"{format_label} | {scale}: alpha = {group['alpha'].mean():.2f} +/- {group['alpha'].std():.2f} "
                  f"(mean +/- std over {len(group)} commodities)")

    plot_format(curves_x, summary_x, "X format (qty per shekel)", GRAPH_DIR / "dfa_scaling_X.png")
    plot_format(curves_1overx, summary_1overx, "1/X format (shekel per unit)", GRAPH_DIR / "dfa_scaling_1overX.png")
    plot_side_by_side(curves_x, summary_x, curves_1overx, summary_1overx, GRAPH_DIR / "dfa_scaling_side_by_side.png")


if __name__ == "__main__":
    main()

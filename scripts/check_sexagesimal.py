"""CLI: check whether the recorded X-format quantities show a base-60
(sexagesimal) fingerprint -- a preference for "regular" numbers and
base-60-friendly fractional parts.

Usage:
    python scripts/check_sexagesimal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import analyze
from babylon_prices.io import load_dataset

SRC = Path(__file__).resolve().parent.parent / "Data" / "Babylon_vanderspek_X.xlsx"


def main():
    df = load_dataset(SRC)
    report = analyze(df)

    print("Regular numbers (only prime factors 2, 3, 5) by commodity:")
    for commodity, stats in report["by_commodity"].items():
        print(f"  {commodity:25s} n={stats['n']:4d}  regular={stats['pct_regular']:.1%}")
    print(f"  {'OVERALL':25s} regular={report['overall_pct_regular']:.1%}")
    print()

    print(f"Values with a fractional part: {report['n_fractional']}")
    print(f"  matching a base-60-friendly fraction: {report['n_fractional_matched']} "
          f"({report['n_fractional_matched'] / report['n_fractional']:.1%})")
    print("  breakdown:")
    for frac, count in sorted(report["fraction_breakdown"].items(), key=lambda kv: -kv[1]):
        print(f"    {frac:>6s} : {count}")


if __name__ == "__main__":
    main()

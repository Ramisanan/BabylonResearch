import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices.benford import leading_digit, digit_distribution, expected_proportions, conformity_stats


def test_leading_digit():
    assert leading_digit(27) == 2
    assert leading_digit(0.037037) == 3
    assert leading_digit(9.99) == 9
    assert leading_digit(100) == 1
    assert leading_digit(0) is None
    assert leading_digit(-5) is None
    assert leading_digit(float("nan")) is None


def test_expected_proportions_sum_to_one():
    total = sum(expected_proportions().values())
    assert abs(total - 1.0) < 1e-9


def test_digit_distribution_shape():
    values = pd.Series([27, 30, 96, 120, 0.5, None])
    dist = digit_distribution(values)
    assert list(dist["digit"]) == list(range(1, 10))
    assert dist["count"].sum() == 5  # None dropped


def test_conformity_stats_perfect_fit_has_r_squared_one():
    # values whose leading digits exactly match Benford's expected proportions
    values = []
    for digit, pct in expected_proportions().items():
        values += [digit] * round(pct * 1000)
    stats = conformity_stats(pd.Series(values))
    assert abs(stats["r_squared"] - 1.0) < 1e-4
    assert "r_squared" in stats


if __name__ == "__main__":
    test_leading_digit()
    test_expected_proportions_sum_to_one()
    test_digit_distribution_shape()
    test_conformity_stats_perfect_fit_has_r_squared_one()
    print("ok")

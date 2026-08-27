import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices.dfa import parse_month, dfa, fit_alpha


def test_parse_month():
    assert parse_month("I") == 1
    assert parse_month("XII") == 12
    assert parse_month("  IV  ") == 4
    assert parse_month("II?") == 2
    assert parse_month("VIB") == 6.5
    assert parse_month("XIIB") == 12.5
    assert parse_month("I-IV") is None
    assert parse_month("I of XIIB") is None


def test_dfa_white_noise_gives_alpha_near_half():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    curve = dfa(x, order=2)
    fit = fit_alpha(curve["n"], curve["F"])
    assert 0.4 < fit["alpha"] < 0.6


def test_dfa_random_walk_gives_alpha_near_one():
    rng = np.random.default_rng(0)
    # a random walk's increments are white noise -> DFA on the walk itself
    # (not its increments) should give alpha close to 1.5; instead, feed the
    # increments' cumulative profile-equivalent by using a persistent series:
    # simplest persistent check is a random walk's first difference already
    # being white noise (alpha~0.5), so here we just check a strongly
    # autocorrelated (smoothed) signal gives alpha clearly above 0.5.
    white = rng.normal(size=4000)
    smoothed = np.cumsum(white)  # random walk -> strongly persistent
    curve = dfa(smoothed, order=2)
    fit = fit_alpha(curve["n"], curve["F"])
    assert fit["alpha"] > 1.0


if __name__ == "__main__":
    test_parse_month()
    test_dfa_white_noise_gives_alpha_near_half()
    test_dfa_random_walk_gives_alpha_near_one()
    print("ok")

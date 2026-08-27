import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import is_regular_number, nearest_base60_fraction


def test_is_regular_number():
    assert is_regular_number(60)        # 2^2 * 3 * 5
    assert is_regular_number(27)        # 3^3
    assert is_regular_number(30)        # 2 * 3 * 5
    assert is_regular_number(0.5)       # 1/2
    assert not is_regular_number(52.5)  # 105/2 = (3*5*7)/2 -> factor of 7
    assert not is_regular_number(7)     # prime, not in {2,3,5}
    assert not is_regular_number(11)
    assert not is_regular_number(0)


def test_nearest_base60_fraction():
    assert nearest_base60_fraction(0.5) == (1, 2)
    assert nearest_base60_fraction(0.25) == (1, 4)
    assert nearest_base60_fraction(0.75) == (3, 4)
    assert nearest_base60_fraction(0.6667, tol=0.01) == (2, 3)
    assert nearest_base60_fraction(0.5772, tol=0.001) is None


if __name__ == "__main__":
    test_is_regular_number()
    test_nearest_base60_fraction()
    print("ok")

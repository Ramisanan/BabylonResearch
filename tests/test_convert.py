import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from babylon_prices import convert_x_to_1_over_x


def test_reciprocal_computed_correctly():
    df = pd.DataFrame({
        "Bar-interpretatie": [27, 30, None, " "],
        "Dat-interpretatie": [57, None, None, None],
        "Mus-interpretatie": [None, None, None, None],
        "Cre-interpretatie": [None, None, None, None],
        "Ses-interpretatie": [None, None, None, None],
        "Wo-interpretatie": [2, None, None, None],
    })

    out = convert_x_to_1_over_x(df)

    assert math.isclose(out["Bar-shekel_per_l"].iloc[0], 1 / 27)
    assert math.isclose(out["Bar-shekel_per_l"].iloc[1], 1 / 30)
    assert pd.isna(out["Bar-shekel_per_l"].iloc[2])
    assert pd.isna(out["Bar-shekel_per_l"].iloc[3])  # whitespace -> NaN
    assert math.isclose(out["Dat-shekel_per_l"].iloc[0], 1 / 57)
    assert math.isclose(out["Wo-shekel_per_mina"].iloc[0], 0.5)


if __name__ == "__main__":
    test_reciprocal_computed_correctly()
    print("ok")

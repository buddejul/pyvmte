import numpy as np
import pandas as pd  # type: ignore
import pytest
from pyvmte.config import TEST_DIR

from pyvmte.estimation.estimation import _estimate_estimand

import statsmodels.api as sm  # type: ignore

from itertools import product

# Import IV2SLS from statsmodels sandbox
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore

RNG = np.random.default_rng(158192581)

Z_SUPPORT = [1, 2, 3]
Z_PDF = [0.2, 0.3, 0.5]
Z_PSCORES = [0.35, 0.6, 0.7]

Z_DICT = dict(zip(Z_SUPPORT, Z_PSCORES))

SAMPLE_SIZE = 100_000

SAMPLED = RNG.choice(Z_SUPPORT, size=SAMPLE_SIZE, p=Z_PDF)
CORRESPONDING = np.array([Z_DICT[i] for i in SAMPLED])

DATA = pd.DataFrame({"z": SAMPLED, "pscore": CORRESPONDING})
DATA

for z_vals in DATA["z"].unique():
    p = Z_DICT[z_vals]
    length = len(DATA[DATA["z"] == z_vals])
    draw = RNG.choice([1, 0], size=length, p=[p, 1 - p])
    DATA.loc[DATA["z"] == z_vals, "d"] = draw


DATA["d"] = DATA["d"].astype(pd.BooleanDtype())

DATA["y"] = 1 + DATA["d"] * 2 + RNG.normal(size=SAMPLE_SIZE)

Y_DATA = np.array(DATA["y"].astype(float))
D_DATA = np.array(DATA["d"].astype(int))
Z_DATA = np.array(DATA["z"].astype(int))


def test_estimate_estimand_ols():
    X = sm.add_constant(D_DATA)
    model = sm.OLS(Y_DATA, X)
    results = model.fit()
    expected = results.params[1]

    estimand = {"type": "ols_slope"}

    actual = _estimate_estimand(estimand, Y_DATA, Z_DATA, D_DATA)

    assert actual == pytest.approx(expected)


def test_estimate_estimand_iv_slope():
    X = sm.add_constant(D_DATA)
    instruments = sm.add_constant(Z_DATA)
    model = IV2SLS(Y_DATA, X, instruments)
    results = model.fit()
    expected = results.params[1]

    estimand = {"type": "iv_slope"}

    actual = _estimate_estimand(estimand, Y_DATA, Z_DATA, D_DATA)

    assert actual == pytest.approx(expected, rel=0.001)


def test_estimate_estimand_cross_moment():
    expected = []
    actual = []
    for d, z in product(np.unique(D_DATA), np.unique(Z_DATA)):
        p = np.mean((D_DATA == d) & (Z_DATA == z))
        expected.append(np.mean(Y_DATA[(D_DATA == d) & (Z_DATA == z)]) * p)
        actual.append(
            _estimate_estimand(
                {"type": "cross", "dz_cross": (d, z)}, Y_DATA, Z_DATA, D_DATA
            )
        )

    assert actual == pytest.approx(expected)

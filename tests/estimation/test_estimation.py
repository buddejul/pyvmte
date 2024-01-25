from itertools import product

import numpy as np
import pandas as pd  # type: ignore
import pytest
import statsmodels.api as sm  # type: ignore
from pyvmte.config import Estimand
from pyvmte.estimation.estimation import _estimate_estimand
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore

RNG = np.random.default_rng(158192581)

Z_SUPPORT = [1, 2, 3]
Z_PDF = [0.2, 0.3, 0.5]
Z_PSCORES = [0.35, 0.6, 0.7]

Z_DICT = dict(zip(Z_SUPPORT, Z_PSCORES, strict=True))

SAMPLE_SIZE = 1_000

SAMPLED = RNG.choice(Z_SUPPORT, size=SAMPLE_SIZE, p=Z_PDF)
CORRESPONDING = np.array([Z_DICT[i] for i in SAMPLED])

DATA = pd.DataFrame({"z": SAMPLED, "pscore": CORRESPONDING})

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
    x = sm.add_constant(D_DATA)
    model = sm.OLS(Y_DATA, x)
    results = model.fit()
    expected = results.params[1]

    estimand = Estimand(esttype="ols_slope")

    actual = _estimate_estimand(estimand, Y_DATA, Z_DATA, D_DATA)

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


def test_estimate_estimand_iv_slope():
    x = sm.add_constant(D_DATA)
    instruments = sm.add_constant(Z_DATA)
    model = IV2SLS(Y_DATA, x, instruments)
    results = model.fit()
    expected = results.params[1]

    estimand = Estimand(esttype="iv_slope")

    actual = _estimate_estimand(estimand, Y_DATA, Z_DATA, D_DATA)

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


def test_estimate_estimand_cross_moment():
    expected = []
    actual = []
    for d, z in product(np.unique(D_DATA), np.unique(Z_DATA)):
        p = np.mean((d == D_DATA) & (z == Z_DATA))
        expected.append(np.mean(Y_DATA[(d == D_DATA) & (z == Z_DATA)]) * p)
        actual.append(
            _estimate_estimand(
                Estimand(esttype="cross", dz_cross=(d, z)),
                Y_DATA,
                Z_DATA,
                D_DATA,
            ),
        )

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))

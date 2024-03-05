"""Test consistent estimation of identified estimands."""
from itertools import product

import numpy as np
import pytest
import statsmodels.api as sm  # type: ignore
from pyvmte.config import RNG, Estimand
from pyvmte.estimation.estimation import _estimate_estimand
from pyvmte.utilities import simulate_data_from_paper_dgp
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore

SAMPLE_SIZE = 1_000


@pytest.fixture()
def data():
    return simulate_data_from_paper_dgp(
        SAMPLE_SIZE,
        rng=RNG,
    )


def test_estimate_estimand_ols(data):
    x = sm.add_constant(data["d"].astype(float))
    model = sm.OLS(data["y"], x)
    results = model.fit()
    expected = results.params[1]

    estimand = Estimand(esttype="ols_slope")

    actual = _estimate_estimand(estimand, data["y"], data["z"], data["d"])

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


def test_estimate_estimand_iv_slope(data):
    x = sm.add_constant(data["d"].astype(float))
    instruments = sm.add_constant(data["z"])
    model = IV2SLS(data["y"], x, instruments)
    results = model.fit()
    expected = results.params[1]

    estimand = Estimand(esttype="iv_slope")

    actual = _estimate_estimand(estimand, data["y"], data["z"], data["d"])

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


def test_estimate_estimand_cross_moment(data):
    expected = []
    actual = []
    for d, z in product(np.unique(data["d"]), np.unique(data["z"])):
        p = np.mean((d == data["d"]) & (z == data["z"]))
        expected.append(np.mean(data["y"][(d == data["d"]) & (z == data["z"])]) * p)
        actual.append(
            _estimate_estimand(
                Estimand(esttype="cross", dz_cross=(d, z)),
                data["y"],
                data["z"],
                data["d"],
            ),
        )

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))

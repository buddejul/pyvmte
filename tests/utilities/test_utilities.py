import numpy as np
import pandas as pd  # type: ignore
import pytest
import statsmodels.api as sm  # type: ignore
from pyvmte.config import DGP_MST, PARAMS_MST, RNG
from pyvmte.utilities import (
    simulate_data_from_paper_dgp,
)
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore

SAMPLE_SIZE = 100_000


@pytest.fixture()
def data():
    return simulate_data_from_paper_dgp(
        SAMPLE_SIZE,
        rng=RNG,
    )


def test_simulate_data_from_paper_dgp_ols(data):
    expected = PARAMS_MST["ols_slope"]

    x = sm.add_constant(data["d"].astype(float))
    y = data["y"].astype(float)

    model = sm.OLS(y, x)
    results = model.fit()
    actual = results.params[1]

    standard_error = results.bse[1]

    # Paper only reports rounded numbers so more lenient here
    assert actual == pytest.approx(expected, abs=0.001 + 3 * standard_error)


def test_simulate_data_from_paper_dgp_iv(data):
    expected = PARAMS_MST["iv_slope"]

    x = sm.add_constant(data["d"].astype(float))
    instruments = sm.add_constant(data["z"].astype(float))
    model = IV2SLS(data["y"].astype(float), x, instruments)
    results = model.fit()
    actual = results.params[1]

    standard_error = results.bse[1]
    # Paper only reports rounded numbers so more lenient here
    assert actual == pytest.approx(expected, abs=0.01 + 5 * standard_error)


def test_simulate_data_from_paper_dgp_pscores(data):
    expected = DGP_MST.pscores

    df_data = pd.DataFrame(data)

    actual = df_data.groupby("z")["d"].mean().values

    assert actual == pytest.approx(expected, abs=5 / np.sqrt(len(data["z"])))

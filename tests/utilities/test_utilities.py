import numpy as np
import pandas as pd
import pytest
from pyvmte.config import TEST_DIR
from pyvmte.utilities import load_paper_dgp, simulate_data_from_paper_dgp

import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

DGP = load_paper_dgp()
RNG = np.random.default_rng(9156781)


def test_simulate_data_from_paper_dgp_ols():
    expected = DGP["ols_slope"]

    sample_size = 100_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    X = sm.add_constant(data["d"].astype(float))
    y = data["y"].astype(float)

    model = sm.OLS(y, X)
    results = model.fit()
    actual = results.params["d"]

    standard_error = results.bse["d"]

    # Paper only reports rounded numbers so more lenient here
    assert actual == pytest.approx(expected, abs=0.001 + 3 * standard_error)


def test_simulate_data_from_paper_dgp_iv():
    expected = DGP["iv_slope"]

    sample_size = 250_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    X = sm.add_constant(data["d"].astype(float))
    instruments = sm.add_constant(data["z"].astype(float))
    model = IV2SLS(data["y"].astype(float), X, instruments)
    results = model.fit()
    actual = results.params[1]

    standard_error = results.bse[1]
    # Paper only reports rounded numbers so more lenient here
    assert actual == pytest.approx(expected, abs=0.001 + 3 * standard_error)


def test_simulate_data_from_paper_dgp_pscores():
    expected = DGP["pscore_z"]

    sample_size = 250_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    actual = data.groupby("z")["d"].mean().values

    assert actual == pytest.approx(expected, abs=2 / np.sqrt(sample_size))

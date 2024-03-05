import numpy as np
import pandas as pd  # type: ignore
import pytest
import statsmodels.api as sm  # type: ignore
from pyvmte.config import BFUNCS_MST, DGP_MST, PARAMS_MST, RNG, U_PART_MST
from pyvmte.utilities import (
    _generate_partition_midpoints,
    _generate_u_partition_from_basis_funcs,
    simulate_data_from_paper_dgp,
)
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore


def test_simulate_data_from_paper_dgp_ols():
    expected = PARAMS_MST["ols_slope"]

    sample_size = 250_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    x = sm.add_constant(data["d"].astype(float))
    y = data["y"].astype(float)

    model = sm.OLS(y, x)
    results = model.fit()
    actual = results.params[1]

    standard_error = results.bse[1]

    # Paper only reports rounded numbers so more lenient here
    assert actual == pytest.approx(expected, abs=0.001 + 3 * standard_error)


def test_simulate_data_from_paper_dgp_iv():
    expected = PARAMS_MST["iv_slope"]

    sample_size = 250_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    x = sm.add_constant(data["d"].astype(float))
    instruments = sm.add_constant(data["z"].astype(float))
    model = IV2SLS(data["y"].astype(float), x, instruments)
    results = model.fit()
    actual = results.params[1]

    standard_error = results.bse[1]
    # Paper only reports rounded numbers so more lenient here
    assert actual == pytest.approx(expected, abs=0.01 + 5 * standard_error)


def test_simulate_data_from_paper_dgp_pscores():
    expected = DGP_MST.pscores

    sample_size = 250_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    df_data = pd.DataFrame(data)

    actual = df_data.groupby("z")["d"].mean().values

    assert actual == pytest.approx(expected, abs=5 / np.sqrt(sample_size))


def test_generate_u_partition_from_basis_funcs():
    expected = U_PART_MST

    actual = _generate_u_partition_from_basis_funcs(BFUNCS_MST)

    assert actual == pytest.approx(expected)


def test_generate_partition_midpoints():
    expected = [0.175, 0.475, 0.65, 0.8, 0.95]
    actual = _generate_partition_midpoints(U_PART_MST)

    assert actual == pytest.approx(expected)

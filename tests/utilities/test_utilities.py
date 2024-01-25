import numpy as np
import pytest
import statsmodels.api as sm  # type: ignore
from pyvmte.utilities import (
    _generate_partition_midpoints,
    _generate_u_partition_from_basis_funcs,
    load_paper_dgp,
    simulate_data_from_paper_dgp,
)
from statsmodels.sandbox.regression.gmm import IV2SLS  # type: ignore

DGP = load_paper_dgp()
RNG = np.random.default_rng(9156781)


@pytest.mark.skip(reason="Not implemented yet")
def test_simulate_data_from_paper_dgp_ols():
    expected = DGP["ols_slope"]

    sample_size = 100_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    x = sm.add_constant(data["d"].astype(float))
    y = data["y"].astype(float)

    model = sm.OLS(y, x)
    results = model.fit()
    actual = results.params["d"]

    standard_error = results.bse["d"]

    # Paper only reports rounded numbers so more lenient here
    assert actual == pytest.approx(expected, abs=0.001 + 3 * standard_error)


def test_simulate_data_from_paper_dgp_iv():
    expected = DGP["iv_slope"]

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


@pytest.mark.skip(reason="Not implemented yet")
def test_simulate_data_from_paper_dgp_pscores():
    expected = DGP["pscores"]

    sample_size = 250_000

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    actual = data.groupby("z")["d"].mean().values

    assert actual == pytest.approx(expected, abs=5 / np.sqrt(sample_size))


def test_generate_u_partition_from_basis_funcs():
    expected = [0, 0.35, 0.6, 0.7, 0.9, 1]

    bfunc1 = {
        "type": "constant",
        "u_lo": 0,
        "u_hi": 0.35,
    }
    bfunc2 = {
        "type": "constant",
        "u_lo": 0.35,
        "u_hi": 0.6,
    }
    bfunc3 = {
        "type": "constant",
        "u_lo": 0.6,
        "u_hi": 0.7,
    }
    bfunc4 = {
        "type": "constant",
        "u_lo": 0.7,
        "u_hi": 0.9,
    }
    bfunc5 = {
        "type": "constant",
        "u_lo": 0.9,
        "u_hi": 1,
    }

    basis_funcs = [bfunc1, bfunc2, bfunc3, bfunc4, bfunc5]

    actual = _generate_u_partition_from_basis_funcs(basis_funcs)

    assert actual == pytest.approx(expected)


def test_generate_partition_midpoints():
    partition = [0, 0.2, 0.3, 0.5, 0.7, 1]

    expected = [0.1, 0.25, 0.4, 0.6, 0.85]

    actual = _generate_partition_midpoints(partition)

    assert actual == pytest.approx(expected)

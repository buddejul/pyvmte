import numpy as np
import pandas as pd  # type: ignore
import pytest
import scipy  # type: ignore[import-untyped]
import statsmodels.api as sm  # type: ignore
from pyvmte.config import DGP_MST, PARAMS_MST, RNG
from pyvmte.utilities import (
    generate_bernstein_basis_funcs,
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


def test_generate_bernstein_basis_funcs():
    k = 2

    generate_bernstein_basis_funcs(k=k)


def test_generate_bernstein_bassi_funcs_compute_late():
    expected = 0.046

    bfunc_dicts = generate_bernstein_basis_funcs(k=2)

    funcs = [bfunc["func"] for bfunc in bfunc_dicts]

    c0 = [0.6, 0.4, 0.3]
    c1 = [0.75, 0.5, 0.25]

    def m0(u):
        return c0[0] * funcs[0](u) + c0[1] * funcs[1](u) + c0[2] * funcs[2](u)

    def m1(u):
        return c1[0] * funcs[0](u) + c1[1] * funcs[1](u) + c1[2] * funcs[2](u)

    lo = 0.35
    hi = 0.9

    w = 1 / (hi - lo)

    actual = scipy.integrate.quad(lambda u: (m1(u) - m0(u)), lo, hi)[0] * w

    assert actual == pytest.approx(expected, abs=0.001)

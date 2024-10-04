"""Tests methods for custom classes."""

from functools import partial

import numpy as np
import pytest
from pyvmte.classes import DGP, Bern
from pyvmte.utilities import bern_bas
from scipy import integrate  # type: ignore[import-untyped]
from scipy.interpolate import BPoly  # type: ignore[import-untyped]


@pytest.fixture()
def example_dgp():
    def foo():
        pass

    return DGP(
        m0=foo,
        m1=foo,
        support_z=np.array([0, 1]),
        pmf_z=np.array([0.5, 0.5]),
        pscores=np.array([0.5, 0.5]),
        joint_pmf_dz={
            0: {0: 0.5, 1: 0.5},
            1: {0: 0.5, 1: 0.5},
        },
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("expectation_d", 0.5),
        ("expectation_z", 0.5),
        ("variance_d", 0.25),
        ("covariance_dz", 0.0),
    ],
)
def test_dgp_methods(example_dgp, method, expected):
    assert getattr(example_dgp, method) == pytest.approx(expected)


# Test Bern class integration method against scipy.integrate.quad
def test_bern_integration():
    a = 0.5
    b = 1

    # Expected
    bern_bas_02 = partial(bern_bas, n=2, v=2)
    expected = integrate.quad(lambda x: bern_bas_02(x=x), a, b)

    # Actual
    bern = Bern(n=2, coefs=np.array([0, 0, 1]))

    actual = bern.integrate(a, b)

    assert actual == pytest.approx(expected[0], abs=1e-5)


def test_bern_integration_mixed_coefs():
    a = 0.5
    b = 1

    # Expected
    x = [0, 1]
    c = [[0.5], [0.4], [0.3]]
    bp = BPoly(c, x)
    expected = integrate.quad(lambda x: bp(x=x), a, b)

    # Actual
    bern = Bern(n=2, coefs=[0.5, 0.4, 0.3])
    actual = bern.integrate(a, b)

    assert actual == pytest.approx(expected[0], abs=1e-5)


def test_bern_call():
    x_grid = np.linspace(0, 1, 1000)

    coef_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for i, coef in enumerate(coef_list):
        bern = Bern(n=2, coefs=coef)

        actual = bern(x_grid)

        expected = bern_bas(n=2, v=i, x=x_grid)

        assert actual == pytest.approx(expected)

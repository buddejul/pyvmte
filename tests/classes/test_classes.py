"""Tests methods for custom classes."""
import numpy as np
import pytest
from pyvmte.classes import DGP


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


def test_dgp_expectation_d(example_dgp):
    expected = 0.5
    assert example_dgp.expectation_d == expected


def test_dgp_expectation_z(example_dgp):
    expected = 0.5
    assert example_dgp.expectation_z == expected


def test_dgp_variance_d(example_dgp):
    expected = 0.5 * (1 - 0.5)
    assert example_dgp.variance_d == expected


def test_dgp_covariance_dz(example_dgp):
    expected = 0
    assert example_dgp.covariance_dz == expected

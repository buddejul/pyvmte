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

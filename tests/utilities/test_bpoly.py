"""Test workings of BPoly class for bernstein polynomial approximations."""
import numpy as np
import pytest
from scipy import integrate  # type: ignore[import-untyped]
from scipy.interpolate import BPoly  # type: ignore[import-untyped]


def test_bpoly_lates_numerical_example():
    coef_d0 = np.array([[0.6], [0.4], [0.3]])
    coef_d1 = np.array([[0.75], [0.5], [0.25]])

    expected = 0.046

    bpoly_d0 = BPoly(coef_d0, [0, 1])
    bpoly_d1 = BPoly(coef_d1, [0, 1])

    bpoly_diff = BPoly(coef_d1 - coef_d0, [0, 1])

    def bpoly_d10(x):
        return bpoly_d1(x) - bpoly_d0(x)

    lo = 0.35
    hi = 0.9

    actual2 = bpoly_diff.integrate(lo, hi) * 1.81
    actual3 = integrate.quad(bpoly_d10, lo, hi)[0] * 1.81

    assert actual2 == pytest.approx(expected, abs=0.001)
    assert actual3 == pytest.approx(expected, abs=0.001)

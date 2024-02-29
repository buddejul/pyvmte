import numpy as np
import pytest
from pyvmte.classes import Estimand
from pyvmte.config import BFUNC_LENS_MST, BFUNCS_MST, DGP_MST, IV_MST
from pyvmte.identification.identification import _compute_choice_weights

MOMENTS = {
    "expectation_d": DGP_MST.expectation_d,
    "variance_d": DGP_MST.variance_d,
    "expectation_z": DGP_MST.expectation_z,
    "covariance_dz": DGP_MST.covariance_dz,
}

late_weight = 1 / (0.9 - 0.35)
late_expected = np.array(
    [
        0,
        -late_weight,
        -late_weight,
        -late_weight,
        0,
        0,
        late_weight,
        late_weight,
        late_weight,
        0,
    ],
) * np.tile(BFUNC_LENS_MST, 2)

# Note: I calculated the following weights myself but they seem to coincide with what
# is plotted in the paper figures.
ols_weights_d0 = [
    0.0,
    -0.970873786407767,
    -1.7475728155339807,
    -1.94174757281553,
    -1.94174757281553,
]

ols_weights_d1 = [2.061855670103093, 1.0309278350515467, 0.20618556701030932, 0, 0]
ols_expected = np.array(ols_weights_d0 + ols_weights_d1) * np.tile(BFUNC_LENS_MST, 2)

weights_iv_d0 = [0.0, -3.3707865168539333, -1.5730337078651693, 0, 0]
weights_iv_d1 = [0, 3.370786516853933, 1.5730337078651686, 0, 0.0]

iv_expected = np.array(weights_iv_d0 + weights_iv_d1) * np.tile(BFUNC_LENS_MST, 2)


@pytest.mark.parametrize(
    ("esttype", "kwargs", "expected"),
    [
        ("ols_slope", {}, ols_expected),
        ("iv_slope", {}, iv_expected),
        ("late", {"u_lo": 0.35, "u_hi": 0.9}, late_expected),
    ],
)
def test_compute_choice_weights(esttype, kwargs, expected):
    target = Estimand(esttype=esttype, **kwargs)

    actual = _compute_choice_weights(
        target=target,
        basis_funcs=BFUNCS_MST,
        instrument=IV_MST,
    )

    assert actual == pytest.approx(expected)

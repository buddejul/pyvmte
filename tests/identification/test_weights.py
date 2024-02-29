"""Tests for the computation of choice weights (identification part)."""
import numpy as np
import pytest
from pyvmte.classes import Estimand
from pyvmte.config import BFUNC_LENS_MST, BFUNCS_MST, IV_MST
from pyvmte.identification.identification import _compute_choice_weights

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
ols_weights_d0 = [0.0, -0.97087378640, -1.747572815533, -1.9417475728, -1.9417475728]

ols_weights_d1 = [2.06185567010, 1.030927835051, 0.2061855670103, 0, 0]
ols_expected = np.array(ols_weights_d0 + ols_weights_d1) * np.tile(BFUNC_LENS_MST, 2)

weights_iv_d0 = [0.0, -3.370786516853, -1.573033707865, 0, 0]
weights_iv_d1 = [0, 3.37078651685, 1.573033707865, 0, 0.0]

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

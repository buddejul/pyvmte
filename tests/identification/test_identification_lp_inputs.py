import numpy as np
import pytest
from pyvmte.config import BFUNCS_MST, IV_MST, SETUP_FIG3, SETUP_FIG7
from pyvmte.identification.identification import (
    _compute_bernstein_weights,
    _compute_choice_weights,
    _compute_inequality_constraint_matrix,
)
from pyvmte.utilities import generate_bernstein_basis_funcs
from scipy import integrate  # type: ignore[import-untyped]


def test_lp_input_c_figure_3():
    target = SETUP_FIG3.target
    late_weight = 1 / (target.u_hi - target.u_lo)
    expected = [
        0,
        -late_weight * (BFUNCS_MST[1]["u_hi"] - BFUNCS_MST[1]["u_lo"]),
        -late_weight * (BFUNCS_MST[2]["u_hi"] - BFUNCS_MST[2]["u_lo"]),
        -late_weight * (BFUNCS_MST[3]["u_hi"] - BFUNCS_MST[3]["u_lo"]),
        0,
        0,
        late_weight * (BFUNCS_MST[1]["u_hi"] - BFUNCS_MST[1]["u_lo"]),
        late_weight * (BFUNCS_MST[2]["u_hi"] - BFUNCS_MST[2]["u_lo"]),
        late_weight * (BFUNCS_MST[3]["u_hi"] - BFUNCS_MST[3]["u_lo"]),
        0,
    ]

    actual = _compute_choice_weights(
        target=target,
        basis_funcs=BFUNCS_MST,
        instrument=IV_MST,
    )

    assert expected == pytest.approx(actual)


def test_compute_inequality_constraint_matrix():
    n_basis_funcs = 3

    expected_increasing = np.array(
        [
            [1, -1, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, -1],
        ],
    )

    expected_decreasing = -expected_increasing

    actual_increasing = _compute_inequality_constraint_matrix(
        shape_constraints=("increasing", "increasing"),
        n_basis_funcs=n_basis_funcs,
    )

    actual_decreasing = _compute_inequality_constraint_matrix(
        shape_constraints=("decreasing", "decreasing"),
        n_basis_funcs=n_basis_funcs,
    )

    assert expected_increasing == pytest.approx(actual_increasing)
    assert expected_decreasing == pytest.approx(actual_decreasing)


def test_compute_bernstein_weights():
    bfunc_dicts = generate_bernstein_basis_funcs(k=2)

    # Hard-code the base functions of the k=2 Bernstein polynomial
    def _b1(u):
        return (1 - u) ** 2

    def _b2(u):
        return 2 * u * (1 - u)

    def _b3(u):
        return u**2

    estimand = SETUP_FIG7.target
    instrument = IV_MST

    u_lo = estimand.u_lo
    u_hi = estimand.u_hi

    w = 1 / (u_hi - u_lo)

    expected = np.array(
        [
            -w * integrate.quad(_b1, u_lo, u_hi)[0],
            -w * integrate.quad(_b2, u_lo, u_hi)[0],
            -w * integrate.quad(_b3, u_lo, u_hi)[0],
            w * integrate.quad(_b1, u_lo, u_hi)[0],
            w * integrate.quad(_b2, u_lo, u_hi)[0],
            w * integrate.quad(_b3, u_lo, u_hi)[0],
        ],
    )

    actual = np.zeros(len(bfunc_dicts) * 2)

    i = 0
    for d in [0, 1]:
        for bfunc in bfunc_dicts:
            actual[i] = _compute_bernstein_weights(
                estimand=estimand,
                d_value=d,
                basis_function=bfunc,
                instrument=instrument,
            )

            i += 1

    assert expected == pytest.approx(actual)


# TODO(@buddejul): Add tests for other weights (IV, OLS, cross).

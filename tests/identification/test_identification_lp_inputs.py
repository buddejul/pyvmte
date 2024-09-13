import numpy as np
import pytest
from pyvmte.config import BFUNCS_MST, IV_MST, SETUP_FIG3
from pyvmte.identification.identification import (
    _compute_choice_weights,
    _compute_inequality_constraint_matrix,
)


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

import pytest
from pyvmte.config import BFUNCS_MST, IV_MST, SETUP_FIG3
from pyvmte.identification.identification import (
    _compute_choice_weights,
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


# TODO write these tests?
def test_lp_input_a_eq_figure_3():
    pass


# TODO write these tests?
def test_lp_input_b_eq_figure_3():
    pass

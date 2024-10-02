"""Test consistent estimation of linear map weights (inputs to estimation LP)."""


import numpy as np
import pytest
from pyvmte.classes import Estimand
from pyvmte.config import (
    DGP_MST,
    IV_MST,
    RNG,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    SETUP_FIG6,
    SETUP_FIG7,
    Setup,
)
from pyvmte.estimation.estimation import (
    estimation,
)
from pyvmte.identification import identification
from pyvmte.utilities import (
    generate_constant_splines_basis_funcs,
    simulate_data_from_paper_dgp,
)

SAMPLE_SIZE = 10_000
REPETITIONS = 250

MAX_SHARE_FAILED = 0.1

TOLERANCE = 3


# --------------------------------------------------------------------------------------
# Write a test that compares the estimated LP inputs in the second step estimation LP
# to the theoretical values when we plug the DGP into identification.
# Note: We need to ignore the additional choice variables and constraints in the
# second step LP that correspond to the absolute value.
@pytest.mark.parametrize(
    ("setup", "u_hi_extra", "shape_constraints"),
    [
        (SETUP_FIG2, 0.1, None),
        (SETUP_FIG3, 0.1, None),
        (SETUP_FIG5, 0.1, None),
        (SETUP_FIG2, 0.2, None),
        (SETUP_FIG3, 0.2, None),
        (SETUP_FIG5, 0.2, None),
        (SETUP_FIG6, 0.2, None),
        (SETUP_FIG7, 0.2, None),
        (SETUP_FIG6, 0.1, None),
        (SETUP_FIG7, 0.1, None),
        (SETUP_FIG6, 0.2, ("decreasing", "decreasing")),
        (SETUP_FIG7, 0.2, ("decreasing", "decreasing")),
        (SETUP_FIG6, 0.1, ("decreasing", "decreasing")),
        (SETUP_FIG7, 0.1, ("decreasing", "decreasing")),
    ],
    ids=[
        "fig2_0.2_none",
        "fig3_0.2_none",
        "fig5_0.2_none",
        "fig2_0.1_none",
        "fig3_0.1_none",
        "fig5_0.1_none",
        "fig6_0.2_none",
        "fig7_0.2_none",
        "fig6_0.1_none",
        "fig7_0.1_none",
        "fig6_0.2_decreasing",
        "fig7_0.2_decreasing",
        "fig6_0.1_decreasing",
        "fig7_0.1_decreasing",
    ],
)
def test_second_step_lp_a_ub_matrix_paper_figures_v2(
    setup: Setup,
    u_hi_extra: float,
    shape_constraints: tuple[str, str] | None,
):
    target_for_id = Estimand(
        esttype="late",
        u_lo=setup.target.u_lo,
        u_hi=0.7 + u_hi_extra,
    )

    identified_estimands = setup.identified_estimands

    # Question: Why is this + 1?
    number_identif_estimands = len(identified_estimands)

    # Note: Why the +1 above?

    # ----------------------------------------------------------------------------------
    # Compute expected matrix from identification problem
    # ----------------------------------------------------------------------------------
    u_partition_id = np.array([0, 0.35, 0.6, 0.7, 0.7 + u_hi_extra, 1])
    basis_funcs = generate_constant_splines_basis_funcs(u_partition_id)
    number_bfuncs = len(basis_funcs)

    res = identification(
        target=target_for_id,
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        m0_dgp=DGP_MST.m0,
        m1_dgp=DGP_MST.m1,
        instrument=IV_MST,
        u_partition=u_partition_id,
        shape_constraints=shape_constraints,
    )

    # This has shape len(identified_estimands) x (len(basis_funcs) * 2)
    expected = res.lp_inputs["a_eq"]

    # Now extract the "a_ub" lp_input matrix from the second step LP
    # Except for the slack variables to mimic the absolute value, the weights
    # should be the same. So we need to ignore the last len(identified_estimands) cols.

    # Columns: number_bfuncs * 2 for the basis functions for d == 1 and d == 2

    # and then the number_identif_estimands slack variables
    n_cols = number_bfuncs * 2 + number_identif_estimands

    if shape_constraints is None:
        # Rows: 1 for deviations, then 2 slack variables for each id estimand
        n_rows = 1 + number_identif_estimands * 2
    else:
        n_rows = 1 + number_identif_estimands * 2 + 2 * (number_bfuncs - 1)

    actual = np.zeros(
        (
            n_rows,
            n_cols,
        ),
    )

    for _ in range(REPETITIONS):
        data = simulate_data_from_paper_dgp(sample_size=SAMPLE_SIZE, rng=RNG)
        target_for_est = Estimand(esttype="late", u_hi_extra=u_hi_extra)

        _res = estimation(
            target=target_for_est,
            identified_estimands=identified_estimands,
            basis_func_type="constant",
            y_data=data["y"],
            d_data=data["d"],
            z_data=data["z"],
            shape_constraints=shape_constraints,
        )

        actual_a_ub = _res.lp_inputs["a_ub"]

        actual += actual_a_ub

    actual /= REPETITIONS

    # Pick out the first actual weight matrix block (first row is the deviations).
    actual_id_restr = actual[1 : number_identif_estimands + 1, : number_bfuncs * 2]

    assert actual_id_restr == pytest.approx(
        expected,
        abs=TOLERANCE / np.sqrt(SAMPLE_SIZE),
    )

    # Pick out the rows after 1 + 2 * number_identif_estimands
    if shape_constraints is not None:
        actual_shape_restr = actual[
            1 + number_identif_estimands * 2 :,
            : number_bfuncs * 2,
        ]

        expected_shape_restr = res.lp_inputs["a_ub"]

        assert actual_shape_restr.shape == expected_shape_restr.shape

        assert actual_shape_restr == pytest.approx(
            expected_shape_restr,
            abs=TOLERANCE / np.sqrt(SAMPLE_SIZE),
        )


# TODO(@buddejul): Also test inputs for other shape restrictions.
# TODO(@buddejul): Also test inputs other than the A_ub matrix.

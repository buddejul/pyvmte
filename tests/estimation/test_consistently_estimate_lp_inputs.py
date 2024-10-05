"""Test consistent estimation of linear map weights (inputs to estimation LP)."""

import numpy as np
import pytest
from pyvmte.classes import Estimand, Instrument
from pyvmte.config import (
    DGP_MST,
    IV_MST,
    RNG,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    SETUP_FIG6,
    SETUP_FIG7,
    SETUP_SM_IDLATE,
    SETUP_SM_SHARP,
    Setup,
)
from pyvmte.estimation.estimation import (
    estimation,
)
from pyvmte.identification import identification
from pyvmte.solutions import draw_valid_simple_model_params, no_solution_region
from pyvmte.utilities import (
    generate_constant_splines_basis_funcs,
    simulate_data_from_paper_dgp,
    simulate_data_from_simple_model_dgp,
)

SAMPLE_SIZE = 10_000
REPETITIONS = 250

MAX_SHARE_FAILED = 0.1

TOLERANCE = 3

identified_sharp = [
    Estimand(esttype="cross", dz_cross=(d, z)) for d in [0, 1] for z in [0, 1]
]

# Leave pscores unspecified, they are estimated in the simulation. This corresponds to
# an application where the true propensity scores are unknown and hence the true
# target parameter is unknown.
identified_late = [Estimand(esttype="late")]


# --------------------------------------------------------------------------------------
# Write a test that compares the estimated LP inputs in the second step estimation LP
# to the theoretical values when we plug the DGP into identification.
# Note: We need to ignore the additional choice variables and constraints in the
# second step LP that correspond to the absolute value.
@pytest.mark.parametrize(
    (
        "model",
        "setup",
        "u_hi_extra",
        "shape_constraints",
        "mte_monotone",
        "monotone_response",
    ),
    [
        ("paper", SETUP_FIG2, 0.1, None, None, None),
        ("paper", SETUP_FIG3, 0.1, None, None, None),
        ("paper", SETUP_FIG5, 0.1, None, None, None),
        ("paper", SETUP_FIG2, 0.2, None, None, None),
        ("paper", SETUP_FIG3, 0.2, None, None, None),
        ("paper", SETUP_FIG5, 0.2, None, None, None),
        ("paper", SETUP_FIG6, 0.2, None, None, None),
        ("paper", SETUP_FIG7, 0.2, None, None, None),
        ("paper", SETUP_FIG6, 0.1, None, None, None),
        ("paper", SETUP_FIG7, 0.1, None, None, None),
        ("paper", SETUP_FIG6, 0.2, ("decreasing", "decreasing"), None, None),
        ("paper", SETUP_FIG7, 0.2, ("decreasing", "decreasing"), None, None),
        ("paper", SETUP_FIG6, 0.1, ("decreasing", "decreasing"), None, None),
        ("paper", SETUP_FIG7, 0.1, ("decreasing", "decreasing"), None, None),
        (
            "simple_model",
            SETUP_SM_IDLATE,
            0.2,
            ("decreasing", "decreasing"),
            None,
            None,
        ),
        ("simple_model", SETUP_SM_IDLATE, 0.2, None, None, None),
        ("simple_model", SETUP_SM_IDLATE, 0.2, None, "decreasing", None),
        ("simple_model", SETUP_SM_IDLATE, 0.2, None, "increasing", None),
        ("simple_model", SETUP_SM_IDLATE, 0.2, None, None, "positive"),
        ("simple_model", SETUP_SM_IDLATE, 0.2, None, None, "negative"),
        (
            "simple_model",
            SETUP_SM_SHARP,
            0.2,
            ("decreasing", "decreasing"),
            None,
            None,
        ),
        ("simple_model", SETUP_SM_SHARP, 0.2, None, None, None),
        ("simple_model", SETUP_SM_SHARP, 0.2, None, "decreasing", None),
        ("simple_model", SETUP_SM_SHARP, 0.2, None, "increasing", None),
        ("simple_model", SETUP_SM_SHARP, 0.2, None, None, "positive"),
        ("simple_model", SETUP_SM_SHARP, 0.2, None, None, "negative"),
    ],
    ids=[
        "paper_fig2_0.2",
        "paper_fig3_0.2",
        "paper_fig5_0.2",
        "paper_fig2_0.1",
        "paper_fig3_0.1",
        "paper_fig5_0.1",
        "paper_fig6_0.2",
        "paper_fig7_0.2",
        "paper_fig6_0.1",
        "paper_fig7_0.1",
        "paper_fig6_0.2_decreasing",
        "paper_fig7_0.2_decreasing",
        "paper_fig6_0.1_decreasing",
        "paper_fig7_0.1_decreasing",
        "simple_model_idlate_0.2",
        "simple_model_idlate_0.2_decreasing",
        "simple_model_idlate_0.2_mte_monotone_increasing",
        "simple_model_idlate_0.2_mte_monotone_decreasing",
        "simple_model_idlate_0.2_monotone_response_positive",
        "simple_model_idlate_0.2_monotone_response_negative",
        "simple_model_sharp_0.2",
        "simple_model_sharp_0.2_decreasing",
        "simple_model_sharp_0.2_mte_monotone_increasing",
        "simple_model_sharp_0.2_mte_monotone_decreasing",
        "simple_model_sharp_0.2_monotone_response_positive",
        "simple_model_sharp_0.2_monotone_response_negative",
    ],
)
def test_second_step_lp_a_ub_matrix_paper_figures_v2(  # noqa: PLR0915
    model: str,
    setup: Setup,
    u_hi_extra: float,
    shape_constraints: tuple[str, str] | None,
    mte_monotone: str | None,
    monotone_response: str | None,
):
    _all_constr = [shape_constraints, mte_monotone, monotone_response]

    _min_none = 2

    if _all_constr.count(None) < _min_none:
        msg = "At least two of the shape restrictions must be None."
        raise ValueError(msg)

    if model == "paper":
        target_for_id = Estimand(
            esttype="late",
            u_lo=setup.target.u_lo,
            u_hi=0.7 + u_hi_extra,
        )

    elif model == "simple_model":
        pscore_lo = 0.4
        pscore_hi = 0.6

        instrument = Instrument(
            support=np.array([0, 1]),
            pmf=np.array([0.5, 0.5]),
            pscores=np.array([pscore_lo, pscore_hi]),
        )

        id_set = "idlate" if setup == SETUP_SM_IDLATE else "sharp"

        _no_sol = no_solution_region(
            id_set=id_set,
            shape_restrictions=shape_constraints,
            mts=mte_monotone,
            monotone_response=monotone_response,
        )

        # Leave pscores unspecified, they are estimated in the simulation.
        target_for_est = Estimand(
            "late",
            u_hi_extra=u_hi_extra,
        )

        # FIXME(@buddejul): This will not work for the ATE.
        target_for_id = Estimand(
            esttype="late",
            u_lo=pscore_lo,
            u_hi=pscore_hi + u_hi_extra,
        )

        (pscore_hi - pscore_lo) / (pscore_hi - pscore_lo + u_hi_extra)

        # FIXME(@buddejul): This will not work for the ATE.
        u_partition_id = np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi_extra, 1])

        y1_at, y1_c, y1_nt, y0_at, y0_c, y0_nt = draw_valid_simple_model_params(
            no_solution_region=_no_sol,
        )

        dgp_params = {
            "y1_at": y1_at,
            "y1_c": y1_c,
            "y1_nt": y1_nt,
            "y0_at": y0_at,
            "y0_c": y0_c,
            "y0_nt": y0_nt,
        }

    identified_estimands = setup.identified_estimands

    number_identif_estimands = len(identified_estimands)

    # ----------------------------------------------------------------------------------
    # Compute expected matrix from identification problem
    # ----------------------------------------------------------------------------------

    if model == "paper":
        u_partition_id = np.array([0, 0.35, 0.6, 0.7, 0.7 + u_hi_extra, 1])
        basis_funcs = generate_constant_splines_basis_funcs(u_partition_id)
        m0_dgp = DGP_MST.m0
        m1_dgp = DGP_MST.m1
        instrument = IV_MST

    if model == "simple_model":
        basis_funcs = generate_constant_splines_basis_funcs(u_partition_id)

        def _at(u: float) -> bool | np.ndarray:
            return np.where(u <= pscore_lo, 1, 0)

        def _c(u: float) -> bool | np.ndarray:
            return np.where((pscore_lo <= u) & (u < pscore_hi), 1, 0)

        def _nt(u: float) -> bool | np.ndarray:
            return np.where(u >= pscore_hi, 1, 0)

        def m0_dgp(u):
            return y0_at * _at(u) + y0_c * _c(u) + y0_nt * _nt(u)

        def m1_dgp(u):
            return y1_at * _at(u) + y1_c * _c(u) + y1_nt * _nt(u)

    number_bfuncs = len(basis_funcs)

    res = identification(
        target=target_for_id,
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        m0_dgp=m0_dgp,
        m1_dgp=m1_dgp,
        instrument=instrument,
        u_partition=u_partition_id,
        shape_constraints=shape_constraints,
        mte_monotone=mte_monotone,
        monotone_response=monotone_response,
    )

    # This has shape len(identified_estimands) x (len(basis_funcs) * 2)
    expected = res.lp_inputs["a_eq"]

    # Now extract the "a_ub" lp_input matrix from the second step LP
    # Except for the slack variables to mimic the absolute value, the weights
    # should be the same. So we need to ignore the last len(identified_estimands) cols.

    # Columns: number_bfuncs * 2 for the basis functions for d == 1 and d == 2

    # and then the number_identif_estimands slack variables
    n_cols = number_bfuncs * 2 + number_identif_estimands

    n_rows = 1 + number_identif_estimands * 2
    if shape_constraints is not None:
        n_rows += 2 * (number_bfuncs - 1)
    elif mte_monotone is not None:
        n_rows += number_bfuncs - 1
    elif monotone_response is not None:
        n_rows += number_bfuncs

    actual = np.zeros(
        (
            n_rows,
            n_cols,
        ),
    )

    for _ in range(REPETITIONS):
        if model == "paper":
            data = simulate_data_from_paper_dgp(sample_size=SAMPLE_SIZE, rng=RNG)
        elif model == "simple_model":
            data = simulate_data_from_simple_model_dgp(
                sample_size=SAMPLE_SIZE,
                rng=RNG,
                dgp_params=dgp_params,
            )

        target_for_est = Estimand(esttype="late", u_hi_extra=u_hi_extra)

        _res = estimation(
            target=target_for_est,
            identified_estimands=identified_estimands,
            basis_func_type="constant",
            y_data=data["y"],
            d_data=data["d"],
            z_data=data["z"],
            shape_constraints=shape_constraints,
            mte_monotone=mte_monotone,
            monotone_response=monotone_response,
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

    if mte_monotone is not None:
        actual_mte_restr = actual[
            1 + number_identif_estimands * 2 :,
            : number_bfuncs * 2,
        ]

        expected_mte_restr = res.lp_inputs["a_ub"]

        assert actual_mte_restr.shape == expected_mte_restr.shape

        assert actual_mte_restr == pytest.approx(
            expected_mte_restr,
            abs=TOLERANCE / np.sqrt(SAMPLE_SIZE),
        )

    if monotone_response is not None:
        actual_monotone_restr = actual[
            1 + number_identif_estimands * 2 :,
            : number_bfuncs * 2,
        ]

        expected_monotone_restr = res.lp_inputs["a_ub"]

        assert actual_monotone_restr.shape == expected_monotone_restr.shape

        assert actual_monotone_restr == pytest.approx(
            expected_monotone_restr,
            abs=TOLERANCE / np.sqrt(SAMPLE_SIZE),
        )


# TODO(@buddejul): Also test inputs other than the A_ub matrix.

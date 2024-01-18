"""Test consistent estimation of linear map weights."""
import numpy as np
import pandas as pd  # type: ignore
import pytest
from pyvmte.config import SETUP_FIG2, SETUP_FIG3, SETUP_FIG5

from pyvmte.estimation.estimation import (
    _build_first_step_ub_matrix,
    _generate_basis_funcs,
    _estimate_prop_z,
    _build_first_step_ub_matrix,
    _compute_u_partition,
    _compute_choice_weights_second_step,
    _build_second_step_ub_matrix,
)
from pyvmte.identification.identification import (
    _compute_equality_constraint_matrix,
    _compute_choice_weights,
)
from pyvmte.utilities import simulate_data_from_paper_dgp, load_paper_dgp

from pyvmte.utilities import simulate_data_from_paper_dgp

RNG = np.random.default_rng(9156781)

U_PARTITION = np.array([0.0, 0.35, 0.6, 0.7, 0.9, 1.0])

BFUNCS = _generate_basis_funcs("constant", U_PARTITION)

DGP = load_paper_dgp()

INSTRUMENT = {
    "support_z": DGP["support_z"],
    "pdf_z": DGP["pdf_z"],
    "pscore_z": DGP["pscore_z"],
}

OLS_SLOPE_WEIGHTS = _compute_equality_constraint_matrix(
    identified_estimands=[{"type": "ols_slope"}],
    basis_funcs=BFUNCS,
    instrument=INSTRUMENT,
)

IV_SLOPE_WEIGHTS = _compute_equality_constraint_matrix(
    identified_estimands=[{"type": "iv_slope"}],
    basis_funcs=BFUNCS,
    instrument=INSTRUMENT,
)

CROSS_WEIGHTS = _compute_equality_constraint_matrix(
    identified_estimands=[{"type": "cross", "dz_cross": (0, 1)}],
    basis_funcs=BFUNCS,
    instrument=INSTRUMENT,
)

# get vector of bfunc lengths, i.e. "u_hi" - "u_lo"
BFUNC_LENGTHS = np.diff(U_PARTITION)
BFUNC_LENGTHS = np.tile(BFUNC_LENGTHS, 2)

OLS_SLOPE_WEIGHTS = OLS_SLOPE_WEIGHTS
IV_SLOPE_WEIGHTS = IV_SLOPE_WEIGHTS
CROSS_WEIGHTS = CROSS_WEIGHTS

SAMPLE_SIZE = 1_000
REPETITIONS = 1_000


@pytest.mark.parametrize(
    "setup,u_hi_target",
    [
        (SETUP_FIG2, 0.9),
        (SETUP_FIG3, 0.9),
        (SETUP_FIG5, 0.9),
        (SETUP_FIG2, 0.8),
        (SETUP_FIG3, 0.8),
        (SETUP_FIG5, 0.8),
    ],
    ids=[
        "fig2_0.9",
        "fig3_0.9",
        "fig5_0.9",
        "fig2_0.8",
        "fig3_0.8",
        "fig5_0.8",
    ],
)
def test_first_step_lp_A_ub_matrix_paper_figures(setup, u_hi_target):
    target = setup["target"]
    target["u_hi"] = u_hi_target

    identified_estimands = setup["identified_estimands"]

    if type(identified_estimands) is not list:
        identified_estimands = [identified_estimands]

    actual = np.zeros(
        (
            2 * len(identified_estimands),
            2 * (len(BFUNCS) + 1) + len(identified_estimands),
        )
    )
    expected = np.zeros(
        (
            2 * len(identified_estimands),
            2 * (len(BFUNCS) + 1) + len(identified_estimands),
        )
    )

    number_iter_diff_shape = 0

    for _ in range(REPETITIONS):
        data = simulate_data_from_paper_dgp(sample_size=SAMPLE_SIZE, rng=RNG)

        pscore_z = _estimate_prop_z(z_data=data["z"], d_data=data["d"])
        u_partition = _compute_u_partition(
            target=target, pscore_z=pscore_z, identified_estimands=identified_estimands
        )
        basis_funcs = _generate_basis_funcs("constant", u_partition)
        A_ub = _build_first_step_ub_matrix(
            basis_funcs=basis_funcs,
            identified_estimands=identified_estimands,
            d_data=data["d"],
            z_data=data["z"],
        )
        # Sometimes we exactly estimate the pscore and then we have fewer bfuncs
        # Require shape of actual and A_ub coincides for update

        weights = _compute_equality_constraint_matrix(
            identified_estimands=identified_estimands,
            basis_funcs=basis_funcs,
            instrument=INSTRUMENT,
        )

        expected_upper = np.hstack((weights, -np.eye(len(identified_estimands))))
        expected_lower = np.hstack((-weights, -np.eye(len(identified_estimands))))
        expected_add = np.vstack((expected_upper, expected_lower))
        if actual.shape == A_ub.shape:
            actual += A_ub
            expected += expected_add
        else:
            number_iter_diff_shape += 1

    expected /= REPETITIONS - number_iter_diff_shape
    actual /= REPETITIONS - number_iter_diff_shape

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


@pytest.mark.parametrize(
    "setup,u_hi_target",
    [
        (SETUP_FIG2, 0.9),
        (SETUP_FIG3, 0.9),
        (SETUP_FIG5, 0.9),
        (SETUP_FIG2, 0.8),
        (SETUP_FIG3, 0.8),
        (SETUP_FIG5, 0.8),
    ],
    ids=[
        "fig2_0.9",
        "fig3_0.9",
        "fig5_0.9",
        "fig2_0.8",
        "fig3_0.8",
        "fig5_0.8",
    ],
)
def test_second_step_lp_c_vector_paper_figures(setup, u_hi_target):
    identified_estimands = setup["identified_estimands"]
    if type(identified_estimands) is not list:
        identified_estimands = [identified_estimands]
    target = setup["target"]
    target["u_hi"] = u_hi_target

    actual = np.zeros((len(BFUNCS) + 1) * 2 + len(identified_estimands))
    expected = np.zeros((len(BFUNCS) + 1) * 2 + len(identified_estimands))

    number_iter_diff_shape = 0

    for _ in range(REPETITIONS):
        data = simulate_data_from_paper_dgp(sample_size=SAMPLE_SIZE, rng=RNG)

        pscore_z = _estimate_prop_z(z_data=data["z"], d_data=data["d"])
        u_partition = _compute_u_partition(
            target=target, pscore_z=pscore_z, identified_estimands=identified_estimands
        )
        basis_funcs = _generate_basis_funcs("constant", u_partition)

        expected_weight = _compute_choice_weights(
            target=target,
            basis_funcs=basis_funcs,
            instrument=INSTRUMENT,
        )

        expected_c = np.hstack((expected_weight, np.zeros(len(identified_estimands))))

        actual_c = _compute_choice_weights_second_step(
            target=target,
            basis_funcs=basis_funcs,
            identified_estimands=identified_estimands,
        )

        if actual.shape == actual_c.shape:
            actual += actual_c
            expected += expected_c
        else:
            number_iter_diff_shape += 1

    expected /= REPETITIONS - number_iter_diff_shape
    actual /= REPETITIONS - number_iter_diff_shape

    if number_iter_diff_shape / REPETITIONS > 0.1:
        raise ValueError("More than 10% of iterations had different shapes.")

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


@pytest.mark.parametrize(
    "setup,u_hi_target",
    [
        (SETUP_FIG2, 0.9),
        (SETUP_FIG3, 0.9),
        (SETUP_FIG5, 0.9),
        (SETUP_FIG2, 0.8),
        (SETUP_FIG3, 0.8),
        (SETUP_FIG5, 0.8),
    ],
    ids=[
        "fig2_0.9",
        "fig3_0.9",
        "fig5_0.9",
        "fig2_0.8",
        "fig3_0.8",
        "fig5_0.8",
    ],
)
def test_second_step_lp_A_ub_matrix_paper_figures(setup, u_hi_target):
    identified_estimands = setup["identified_estimands"]
    if type(identified_estimands) is not list:
        identified_estimands = [identified_estimands]
    target = setup["target"]
    target["u_hi"] = u_hi_target

    number_bfuncs = (len(BFUNCS) + 1) * 2
    number_identif_estimands = len(identified_estimands)

    actual = np.zeros(
        (
            1 + number_identif_estimands * 2,
            number_bfuncs + number_identif_estimands,
        )
    )

    expected_without_first_row = np.zeros(
        (
            number_identif_estimands * 2,
            number_bfuncs + number_identif_estimands,
        )
    )

    number_iter_diff_shape = 0

    for _ in range(REPETITIONS):
        data = simulate_data_from_paper_dgp(sample_size=SAMPLE_SIZE, rng=RNG)

        pscore_z = _estimate_prop_z(z_data=data["z"], d_data=data["d"])
        u_partition = _compute_u_partition(
            target=target, pscore_z=pscore_z, identified_estimands=identified_estimands
        )
        basis_funcs = _generate_basis_funcs("constant", u_partition)

        expected_weight = _compute_equality_constraint_matrix(
            identified_estimands=identified_estimands,
            basis_funcs=basis_funcs,
            instrument=INSTRUMENT,
        )

        expected_second_row_block = np.hstack(
            (expected_weight, -np.eye(number_identif_estimands))
        )
        expected_third_row_block = np.hstack(
            (-expected_weight, -np.eye(number_identif_estimands))
        )

        expected_add = np.vstack((expected_second_row_block, expected_third_row_block))

        actual_A_ub = _build_second_step_ub_matrix(
            basis_funcs=basis_funcs,
            identified_estimands=identified_estimands,
            z_data=data["z"],
            d_data=data["d"],
        )

        if actual.shape == actual_A_ub.shape:
            actual += actual_A_ub
            expected_without_first_row += expected_add
        else:
            number_iter_diff_shape += 1

    expected_without_first_row /= REPETITIONS - number_iter_diff_shape
    actual /= REPETITIONS - number_iter_diff_shape

    if number_iter_diff_shape / REPETITIONS > 0.1:
        raise ValueError("More than 10% of iterations had different shapes.")

    first_row = np.hstack((np.zeros(number_bfuncs), np.ones(number_identif_estimands)))
    expected = np.vstack((first_row, expected_without_first_row))

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))

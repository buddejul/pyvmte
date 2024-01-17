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
)
from pyvmte.identification.identification import (
    _compute_equality_constraint_matrix,
    _compute_choice_weights,
)
from pyvmte.utilities import simulate_data_from_paper_dgp, load_paper_dgp

from pyvmte.utilities import simulate_data_from_paper_dgp

RNG = np.random.default_rng(9156781)

U_PARTITION = [0.0, 0.35, 0.6, 0.7, 0.9, 1.0]

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


def test_build_first_step_ub_matrix_consistency_ols_slope():
    repetitions = REPETITIONS
    sample_size = SAMPLE_SIZE

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    actual = np.zeros((2 * 1, 5 * 2 + 1))

    pscores = _estimate_prop_z(z_data=data["z"], d_data=data["d"])

    u_partition = [0, 0.9, 1]
    u_partition.extend(pscores)

    u_partition = np.unique(u_partition)

    bfuncs_estimation = _generate_basis_funcs("constant", u_partition)

    for i in range(repetitions):
        out = _build_first_step_ub_matrix(
            basis_funcs=bfuncs_estimation,
            identified_estimands=[{"type": "ols_slope"}],
            d_data=data["d"],
            z_data=data["z"],
        )

        # Weighted average of actual and out depending on i
        actual = (i * actual + out) / (i + 1)

    # Drop last row (negative duplicate) and last column (dummy variables)
    actual = actual[:, :-1]
    actual = actual[:-1, :]

    print(f"Partition estimation: {u_partition}")
    print(f"Partition identification: {U_PARTITION}")

    print(f"Actual: {actual}")
    print(f"Expected: {OLS_SLOPE_WEIGHTS}")

    assert actual == pytest.approx(OLS_SLOPE_WEIGHTS, abs=5 / np.sqrt(sample_size))


def test_build_first_step_ub_matrix_consistency_iv_slope():
    repetitions = REPETITIONS
    sample_size = SAMPLE_SIZE

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    actual = np.zeros((2 * 1, 5 * 2 + 1))

    pscores = _estimate_prop_z(z_data=data["z"], d_data=data["d"])

    u_partition = [0, 0.9, 1]
    u_partition.extend(pscores)

    u_partition = np.unique(u_partition)

    bfuncs_estimation = _generate_basis_funcs("constant", u_partition)

    for i in range(repetitions):
        out = _build_first_step_ub_matrix(
            basis_funcs=bfuncs_estimation,
            identified_estimands=[{"type": "iv_slope"}],
            d_data=data["d"],
            z_data=data["z"],
        )

        # Weighted average of actual and out depending on i
        actual = (i * actual + out) / (i + 1)

    # Drop last row (negative duplicate) and last column (dummy variables)
    actual = actual[:, :-1]
    actual = actual[:-1, :]

    print(f"Partition estimation: {u_partition}")
    print(f"Partition identification: {U_PARTITION}")

    print(f"Actual: {actual}")
    print(f"Expected: {IV_SLOPE_WEIGHTS}")

    assert actual == pytest.approx(IV_SLOPE_WEIGHTS, abs=7 / np.sqrt(sample_size))


def test_build_first_step_ub_matrix_consistency_cross():
    repetitions = REPETITIONS
    sample_size = SAMPLE_SIZE

    data = simulate_data_from_paper_dgp(sample_size, rng=RNG)

    actual = np.zeros((2 * 1, 5 * 2 + 1))

    pscores = _estimate_prop_z(z_data=data["z"], d_data=data["d"])

    u_partition = [0, 0.9, 1]
    u_partition.extend(pscores)

    u_partition = np.unique(u_partition)

    bfuncs_estimation = _generate_basis_funcs("constant", u_partition)

    for i in range(repetitions):
        out = _build_first_step_ub_matrix(
            basis_funcs=bfuncs_estimation,
            identified_estimands=[{"type": "cross", "dz_cross": (0, 1)}],
            d_data=data["d"],
            z_data=data["z"],
        )

        # Weighted average of actual and out depending on i
        actual = (i * actual + out) / (i + 1)

    # Drop last row (negative duplicate) and last column (dummy variables)
    actual = actual[:, :-1]
    actual = actual[:-1, :]

    print(f"Partition estimation: {u_partition}")
    print(f"Partition identification: {U_PARTITION}")

    print(f"Actual: {actual}")
    print(f"Expected: {CROSS_WEIGHTS}")

    assert actual == pytest.approx(CROSS_WEIGHTS, abs=5 / np.sqrt(sample_size))


@pytest.mark.parametrize(
    "setup",
    [(SETUP_FIG2), (SETUP_FIG3), (SETUP_FIG5)],
    ids=["fig2", "fig3", "fig5"],
)
def test_first_step_lp_A_ub_matrix_paper_figures(setup):
    target = setup["target"]
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
    "setup",
    [(SETUP_FIG2), (SETUP_FIG3), (SETUP_FIG5)],
    ids=["fig2", "fig3", "fig5"],
)
def test_second_step_lp_c_vector_paper_figures(setup):
    identified_estimands = setup["identified_estimands"]
    if type(identified_estimands) is not list:
        identified_estimands = [identified_estimands]
    target = setup["target"]

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

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))

"""Test consistent estimation of linear map weights (inputs to estimation LP)."""
from dataclasses import replace

import numpy as np
import pytest
from pyvmte.config import (
    BFUNCS_MST,
    RNG,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    Setup,
)
from pyvmte.estimation.estimation import (
    _build_first_step_ub_matrix,
    _build_second_step_ub_matrix,
    _compute_choice_weights_second_step,
    _compute_u_partition,
    _estimate_instrument_characteristics,
    _estimate_prop_z,
    _generate_array_of_pscores,
    _generate_basis_funcs,
)
from pyvmte.identification.identification import (
    _compute_choice_weights,
    _compute_equality_constraint_matrix,
)
from pyvmte.utilities import simulate_data_from_paper_dgp

SAMPLE_SIZE = 1_000
REPETITIONS = 250

MAX_SHARE_FAILED = 0.1


@pytest.mark.parametrize(
    ("setup", "u_hi_target"),
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
def test_first_step_lp_a_ub_matrix_paper_figures(setup: Setup, u_hi_target: float):
    target = replace(setup.target, u_hi=u_hi_target)
    identified_estimands = setup.identified_estimands

    actual = np.zeros(
        (
            2 * len(identified_estimands),
            2 * (len(BFUNCS_MST) + 1) + len(identified_estimands),
        ),
    )
    expected = np.zeros(
        (
            2 * len(identified_estimands),
            2 * (len(BFUNCS_MST) + 1) + len(identified_estimands),
        ),
    )

    number_iter_diff_shape = 0

    for _ in range(REPETITIONS):
        simulated = _simulate_and_compute(target=target)

        a_ub = _build_first_step_ub_matrix(
            basis_funcs=simulated["basis_funcs"],
            identified_estimands=identified_estimands,
            data=simulated["data"],
            instrument=simulated["instrument"],
        )
        # Sometimes we exactly estimate the pscore and then we have fewer bfuncs
        # Require shape of actual and a_ub coincides for update

        weights = _compute_equality_constraint_matrix(
            identified_estimands=identified_estimands,
            basis_funcs=simulated["basis_funcs"],
            instrument=simulated["instrument"],
        )

        expected_upper = np.hstack((weights, -np.eye(len(identified_estimands))))
        expected_lower = np.hstack((-weights, -np.eye(len(identified_estimands))))
        expected_add = np.vstack((expected_upper, expected_lower))
        if actual.shape == a_ub.shape:
            actual += a_ub
            expected += expected_add
        else:
            number_iter_diff_shape += 1

    expected /= REPETITIONS - number_iter_diff_shape
    actual /= REPETITIONS - number_iter_diff_shape
    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


@pytest.mark.parametrize(
    ("setup", "u_hi_target"),
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
def test_second_step_lp_c_vector_paper_figures(setup: Setup, u_hi_target: float):
    target = replace(setup.target, u_hi=u_hi_target)
    identified_estimands = setup.identified_estimands

    actual = np.zeros((len(BFUNCS_MST) + 1) * 2 + len(identified_estimands))
    expected = np.zeros((len(BFUNCS_MST) + 1) * 2 + len(identified_estimands))

    number_iter_diff_shape = 0

    for _ in range(REPETITIONS):
        simulated = _simulate_and_compute(target=target)

        expected_weight = _compute_choice_weights(
            target=target,
            basis_funcs=simulated["basis_funcs"],
            instrument=simulated["instrument"],
        )

        expected_c = np.hstack((expected_weight, np.zeros(len(identified_estimands))))

        actual_c = _compute_choice_weights_second_step(
            target=target,
            basis_funcs=simulated["basis_funcs"],
            identified_estimands=identified_estimands,
            instrument=simulated["instrument"],
        )

        if actual.shape == actual_c.shape:
            actual += actual_c
            expected += expected_c
        else:
            number_iter_diff_shape += 1

    expected /= REPETITIONS - number_iter_diff_shape
    actual /= REPETITIONS - number_iter_diff_shape

    if number_iter_diff_shape / REPETITIONS > MAX_SHARE_FAILED:
        msg = "More than 10% of iterations had different shapes."
        raise ValueError(msg)

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


@pytest.mark.parametrize(
    ("setup", "u_hi_target"),
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
def test_second_step_lp_a_ub_matrix_paper_figures(setup: Setup, u_hi_target: float):
    target = replace(setup.target, u_hi=u_hi_target)
    identified_estimands = setup.identified_estimands

    number_bfuncs = (len(BFUNCS_MST) + 1) * 2
    number_identif_estimands = len(identified_estimands)

    actual = np.zeros(
        (
            1 + number_identif_estimands * 2,
            number_bfuncs + number_identif_estimands,
        ),
    )

    expected_without_first_row = np.zeros(
        (
            number_identif_estimands * 2,
            number_bfuncs + number_identif_estimands,
        ),
    )

    number_iter_diff_shape = 0

    for _ in range(REPETITIONS):
        simulated = _simulate_and_compute(target=target)

        expected_weight = _compute_equality_constraint_matrix(
            identified_estimands=identified_estimands,
            basis_funcs=simulated["basis_funcs"],
            instrument=simulated["instrument"],
        )

        expected_second_row_block = np.hstack(
            (expected_weight, -np.eye(number_identif_estimands)),
        )
        expected_third_row_block = np.hstack(
            (-expected_weight, -np.eye(number_identif_estimands)),
        )

        expected_add = np.vstack((expected_second_row_block, expected_third_row_block))

        actual_a_ub = _build_second_step_ub_matrix(
            basis_funcs=simulated["basis_funcs"],
            identified_estimands=identified_estimands,
            data=simulated["data"],
            instrument=simulated["instrument"],
        )

        if actual.shape == actual_a_ub.shape:
            actual += actual_a_ub
            expected_without_first_row += expected_add
        else:
            number_iter_diff_shape += 1

    expected_without_first_row /= REPETITIONS - number_iter_diff_shape
    actual /= REPETITIONS - number_iter_diff_shape

    if number_iter_diff_shape / REPETITIONS > MAX_SHARE_FAILED:
        msg = "More than 10% of iterations had different shapes."
        raise ValueError(msg)

    first_row = np.hstack((np.zeros(number_bfuncs), np.ones(number_identif_estimands)))
    expected = np.vstack((first_row, expected_without_first_row))

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))


def _simulate_and_compute(target):
    data = simulate_data_from_paper_dgp(sample_size=SAMPLE_SIZE, rng=RNG)

    pscore_z = _estimate_prop_z(
        z_data=data["z"],
        d_data=data["d"],
        support=np.unique(data["z"]),
    )
    u_partition = _compute_u_partition(
        target=target,
        pscore_z=pscore_z,
    )
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    instrument = _estimate_instrument_characteristics(
        z_data=data["z"],
        d_data=data["d"],
    )

    data["pscores"] = _generate_array_of_pscores(
        z_data=data["z"],
        support=instrument.support,
        pscores=instrument.pscores,
    )

    return {
        "data": data,
        "instrument": instrument,
        "basis_funcs": basis_funcs,
    }

import numpy as np
import pandas as pd
import pytest
from pyvmte.config import TEST_DIR

from pyvmte.estimation.estimation import (
    _generate_basis_funcs,
    _estimate_prop_z,
    _generate_array_of_pscores,
    _build_first_step_ub_matrix,
    _compute_first_step_bounds,
    _first_step_linear_program,
    _compute_choice_weights_second_step,
    _create_funcs_from_dicts,
    _build_second_step_ub_matrix,
    _compute_second_step_bounds,
    _compute_first_step_upper_bounds,
    _second_step_linear_program,
    _compute_second_step_upper_bounds,
    _estimate_identified_estimands,
    _compute_u_partition,
    _estimate_weights_estimand,
)

from pyvmte.utilities import simulate_data_from_paper_dgp

RNG = np.random.default_rng(9156781)


def test_generate_basis_funcs():
    u_partition = [0, 0.35, 0.65, 0.7, 1]

    expected = [
        {"type": "constant", "u_lo": 0, "u_hi": 0.35},
        {"type": "constant", "u_lo": 0.35, "u_hi": 0.65},
        {"type": "constant", "u_lo": 0.65, "u_hi": 0.7},
        {"type": "constant", "u_lo": 0.7, "u_hi": 1},
    ]

    actual = _generate_basis_funcs(basis_func_type="constant", u_partition=u_partition)

    assert actual == expected


def test_estimate_prop_z():
    z_data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    d_data = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1])

    expected = np.array([0.5, 2 / 3, 0.75])

    actual = _estimate_prop_z(z_data, d_data)

    assert actual == pytest.approx(expected)


def test_generate_array_of_pscores():
    z_data = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])
    d_data = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1])

    expected = np.array([0.5, 0.5, 2 / 3, 2 / 3, 2 / 3, 0.75, 0.75, 0.75, 0.75])

    actual = _generate_array_of_pscores(z_data, d_data)

    assert actual == pytest.approx(expected)


def test_build_first_step_ub_matrix():
    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    iv_estimand = {
        "type": "iv_slope",
    }

    ols_estimand = {
        "type": "ols_slope",
    }

    identified_estimands = [iv_estimand, ols_estimand]

    d_data = RNG.choice([0, 1], size=100)
    z_data = RNG.choice([1, 2, 3], size=100)

    A_ub = _build_first_step_ub_matrix(
        basis_funcs=basis_funcs,
        identified_estimands=identified_estimands,
        d_data=d_data,
        z_data=z_data,
    )

    expected = (2 * 2, 4 * 2 + 2)
    actual = A_ub.shape

    assert actual == expected


def test_compute_first_step_bounds():
    identified_estimands = [
        {
            "type": "iv_slope",
        },
        {
            "type": "ols_slope",
        },
    ]

    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    expected = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (None, None),
        (None, None),
    ]

    actual = _compute_first_step_bounds(identified_estimands, basis_funcs)

    assert actual == expected


def test_first_step_linear_program_runs_and_non_zero():
    identified_estimands = [
        {
            "type": "iv_slope",
        },
        {
            "type": "ols_slope",
        },
    ]

    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    d_data = RNG.choice([0, 1], size=100)
    z_data = RNG.choice([1, 2, 3], size=100)
    y_data = RNG.normal(size=100)

    beta_hat = RNG.normal(size=len(identified_estimands))
    result = _first_step_linear_program(
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        y_data=y_data,
        d_data=d_data,
        z_data=z_data,
        beta_hat=beta_hat,
    )

    assert result["minimal_deviations"] != 0


def test_compute_choice_weights_second_step():
    late_estimand = {
        "type": "late",
        "u_lo": 0.5,
        "u_hi": 0.75,
    }

    iv_estimand = {"type": "iv_slope"}
    ols_estimand = {"type": "ols_slope"}

    u_partition = [0, 0.35, 0.65, 0.7, 1]

    basis_funcs = _generate_basis_funcs("constant", u_partition)

    identified_estimands = [iv_estimand, ols_estimand]
    result = _compute_choice_weights_second_step(
        target=late_estimand,
        basis_funcs=basis_funcs,
        identified_estimands=identified_estimands,
    )

    assert result.shape == (len(basis_funcs) * 2 + len(identified_estimands),)


def test_create_funcs_from_dicts():
    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    out = _create_funcs_from_dicts(basis_funcs)

    assert all([callable(func) for func in out]) and len(out) == len(u_partition) - 1


def test_build_second_step_ub_matrix():
    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    iv_estimand = {
        "type": "iv_slope",
    }

    ols_estimand = {
        "type": "ols_slope",
    }

    identified_estimands = [iv_estimand, ols_estimand]

    d_data = RNG.choice([0, 1], size=100)
    z_data = RNG.choice([1, 2, 3], size=100)

    result = _build_second_step_ub_matrix(
        basis_funcs, identified_estimands, z_data, d_data
    )

    assert result.shape == (
        1 + 2 * len(identified_estimands),
        len(basis_funcs) * 2 + len(identified_estimands),
    )


def test_compute_second_step_bounds():
    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)
    identified_estimands = [{"type": "iv_slope"}, {"type": "ols_slope"}]

    actual = _compute_second_step_bounds(basis_funcs, identified_estimands)

    expected = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (None, None),
        (None, None),
    ]

    assert actual == expected


def test_compute_first_step_upper_bounds():
    beta_hat = np.array([1, 2, 3])
    expected = np.array([1, 2, 3, -1, -2, -3])

    actual = _compute_first_step_upper_bounds(beta_hat)

    assert actual == pytest.approx(expected)


@pytest.mark.skip(reason="Not implemented yet")
def test_estimate_identified_estimands():
    pass


def test_second_step_linear_program_runs():
    target = {"type": "late", "u_lo": 0.35, "u_hi": 0.9}
    identified_estimands = [
        {
            "type": "iv_slope",
        },
        {
            "type": "ols_slope",
        },
    ]

    u_partition = [0, 0.35, 0.65, 0.7, 0.9, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    sample_size = 100_000

    data = simulate_data_from_paper_dgp(sample_size=sample_size, rng=RNG)

    y_data = np.array(data["y"].astype(float))
    d_data = np.array(data["d"].astype(float))
    z_data = np.array(data["z"].astype(float))

    beta_hat = _estimate_identified_estimands(
        identified_estimands=identified_estimands,
        y_data=y_data,
        d_data=d_data,
        z_data=z_data,
    )

    beta_hat = np.array(beta_hat)

    tolerance = 1 / sample_size

    minimal_deviations = 10e-5

    result = _second_step_linear_program(
        target,
        identified_estimands,
        basis_funcs,
        z_data,
        d_data,
        minimal_deviations,
        tolerance,
        beta_hat,
    )

    assert result is not None


def test_compute_second_step_upper_bounds():
    beta_hat = np.array([1, 2, 3])
    tolerance = 1 / 100
    minimal_deviations = 10e-5

    expected = np.array([tolerance + minimal_deviations, 1, 2, 3, -1, -2, -3])

    actual = _compute_second_step_upper_bounds(
        minimal_deviations=minimal_deviations, tolerance=tolerance, beta_hat=beta_hat
    )

    assert actual == pytest.approx(expected)


def test_compute_u_partition():
    target = {"type": "late", "u_lo": 0.35, "u_hi": 0.9}
    pscore_z = [0.1, 0.2, 0.64, 0.83]

    expected = [0, 0.1, 0.2, 0.35, 0.64, 0.83, 0.9, 1]

    actual = _compute_u_partition(target=target, pscore_z=pscore_z)

    assert actual == pytest.approx(expected)


def test_estimate_weights_estimand_length():
    u_partitition = [0, 0.35, 0.65, 0.7, 0.9, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partitition)

    actual = _estimate_weights_estimand(
        estimand={"type": "iv_slope"},
        basis_funcs=basis_funcs,
        z_data=RNG.normal(size=100),
        d_data=RNG.normal(size=100),
    )

    assert len(actual) == len(basis_funcs) * 2


@pytest.mark.skip(reason="Unsure whether this is true in finite sample")
def test_estimate_weights_estimand_symmetry():
    u_partitition = [0, 0.35, 0.65, 0.7, 0.9, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partitition)

    actual = _estimate_weights_estimand(
        estimand={"type": "iv_slope"},
        basis_funcs=basis_funcs,
        z_data=RNG.normal(size=100),
        d_data=RNG.normal(size=100),
    )

    first_half = np.array(actual[: len(actual) // 2])
    second_half = -1 * np.array(actual[len(actual) // 2 :])

    assert first_half == pytest.approx(second_half)


@pytest.mark.skip(reason="Unsure whether this is true in finite sample")
def test_build_first_step_ub_matrix_symmetry():
    u_partition = [0, 0.35, 0.65, 0.7, 1]
    basis_funcs = _generate_basis_funcs("constant", u_partition)

    iv_estimand = {
        "type": "iv_slope",
    }

    ols_estimand = {
        "type": "ols_slope",
    }

    identified_estimands = [iv_estimand, ols_estimand]

    d_data = RNG.choice([0, 1], size=100)
    z_data = RNG.choice([1, 2, 3], size=100)

    A_ub = _build_first_step_ub_matrix(
        basis_funcs=basis_funcs,
        identified_estimands=identified_estimands,
        d_data=d_data,
        z_data=z_data,
    )

    num_bfuncs = len(basis_funcs) * 2
    # Take first row and first len(basis_funcs)*2 columns
    weights = A_ub[0, :num_bfuncs]

    # Check first len(basis_funcs) weights are the same as -1 the ret
    assert weights[: num_bfuncs // 2] == pytest.approx(-1 * weights[num_bfuncs // 2 :])

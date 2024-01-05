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
)

RNG = np.random.default_rng(9156781)


def test_generate_basis_funcs():
    u_partition = [0, 0.35, 0.65, 0.7, 1]

    expected = [
        {"d_value": 0, "u_lo": 0, "u_hi": 0.35},
        {"d_value": 0, "u_lo": 0.35, "u_hi": 0.65},
        {"d_value": 0, "u_lo": 0.65, "u_hi": 0.7},
        {"d_value": 0, "u_lo": 0.7, "u_hi": 1},
        {"d_value": 1, "u_lo": 0, "u_hi": 0.35},
        {"d_value": 1, "u_lo": 0.35, "u_hi": 0.65},
        {"d_value": 1, "u_lo": 0.65, "u_hi": 0.7},
        {"d_value": 1, "u_lo": 0.7, "u_hi": 1},
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


def test_first_step_linear_program_runs():
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

    result = _first_step_linear_program(
        identified_estimands, basis_funcs, d_data, z_data
    )

    assert type(result) == float


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

    assert result.shape == (len(basis_funcs) + len(identified_estimands),)


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
        len(basis_funcs) + len(identified_estimands),
    )

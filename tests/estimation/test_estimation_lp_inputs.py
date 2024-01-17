"""Test consistent estimation of linear map weights."""
import numpy as np
import pandas as pd  # type: ignore
import pytest
from pyvmte.config import TEST_DIR

from pyvmte.estimation.estimation import (
    _build_first_step_ub_matrix,
    _generate_basis_funcs,
    _estimate_prop_z,
)
from pyvmte.identification.identification import _compute_equality_constraint_matrix
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


def test_build_first_step_ub_matrix_consistency_ols_slope():
    repetitions = 1_000
    sample_size = 10_000

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
    repetitions = 1_000
    sample_size = 10_000

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
    repetitions = 1_000
    sample_size = 10_000

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

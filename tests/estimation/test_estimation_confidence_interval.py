"""Test functions for (bootstrap) confidence intervals."""

import numpy as np
import pytest
from pyvmte.classes import Estimand
from pyvmte.config import RNG
from pyvmte.estimation import estimation
from pyvmte.utilities import simulate_data_from_simple_model_dgp


@pytest.fixture()
def dgp_params():
    return {
        "y1_at": 1,
        "y0_at": 0,
        "y1_c": 0.5,
        "y0_c": 0,
        "y1_nt": 1,  # ensures we are at the upper bound of the identified set
        "y0_nt": 0,
    }


@pytest.fixture()
def estimation_setup(dgp_params):
    data = simulate_data_from_simple_model_dgp(
        sample_size=10_000,
        rng=RNG,
        dgp_params=dgp_params,
    )
    return {
        "target": Estimand("late", u_hi_extra=0.2),
        "identified_estimands": [
            Estimand(
                "late",
            ),
        ],
        "basis_func_type": "constant",
        "y_data": data["y"],
        "z_data": data["z"],
        "d_data": data["d"],
        "confidence_interval": "bootstrap",
    }


def test_estimation_with_confidence_interval_runs(estimation_setup):
    estimation(**estimation_setup)


def test_estimation_with_confidence_interval_correct_ordering(estimation_setup):
    res = estimation(**estimation_setup)

    assert res.ci_lower <= res.lower_bound
    assert res.ci_upper >= res.upper_bound


def test_ci_right_coverage(estimation_setup, dgp_params):
    num_sims = 100

    alpha = 0.05

    # Delete the "data" key in estimation_setup
    for data in ["y_data", "z_data", "d_data"]:
        estimation_setup.pop(data)

    w = 0.5
    late_c = dgp_params["y1_c"] - dgp_params["y0_c"]
    late_nt = dgp_params["y1_nt"] - dgp_params["y0_nt"]
    true = w * late_c + (1 - w) * late_nt

    sims_ci_lower = np.zeros(num_sims)
    sims_ci_upper = np.zeros(num_sims)

    for i in range(num_sims):
        _sim_data = simulate_data_from_simple_model_dgp(
            sample_size=10_000,
            rng=RNG,
            dgp_params=dgp_params,
        )
        _res = estimation(
            **estimation_setup,
            y_data=_sim_data["y"],
            z_data=_sim_data["z"],
            d_data=_sim_data["d"],
        )

        sims_ci_lower[i] = _res.ci_lower
        sims_ci_upper[i] = _res.ci_upper

    actual_coverage = np.mean((sims_ci_lower <= true) & (sims_ci_upper >= true))

    actual_coverage_upper = np.mean(sims_ci_upper >= true)
    actual_coverage_lower = np.mean(sims_ci_lower <= true)

    expected_coverage = 1 - alpha
    expected_coverage_upper = 1 - alpha
    expected_coverage_lower = 1

    assert actual_coverage_lower == pytest.approx(expected_coverage_lower, abs=0.02)
    assert actual_coverage_upper == pytest.approx(expected_coverage_upper, abs=0.02)

    assert actual_coverage == pytest.approx(expected_coverage, abs=0.02)

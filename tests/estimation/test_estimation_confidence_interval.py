"""Test functions for (bootstrap) confidence intervals."""

import numpy as np
import pytest
from pyvmte.classes import Estimand
from pyvmte.config import RNG
from pyvmte.estimation import estimation
from pyvmte.utilities import simulate_data_from_simple_model_dgp

# --------------------------------------------------------------------------------------
# Parameters for tests
# --------------------------------------------------------------------------------------
alpha = 0.1
n_boot = 250
n_subsamples = n_boot
sample_size = 1_000
subsample_size = int(np.floor(np.sqrt(sample_size)))
num_sims = 100

inference_methods = ["subsampling", "recentered_bootstrap"]


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
        sample_size=sample_size,
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
        "confidence_interval_options": {
            "n_boot": n_boot,
            "alpha": alpha,
            "n_subsamples": n_subsamples,
            "subsample_size": subsample_size,
        },
    }


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("confidence_interval", inference_methods)
def test_estimation_with_confidence_interval_runs(
    estimation_setup,
    confidence_interval,
):
    estimation_setup["confidence_interval"] = confidence_interval
    estimation(**estimation_setup)


@pytest.mark.parametrize("confidence_interval", inference_methods)
def test_estimation_with_confidence_interval_correct_ordering(
    confidence_interval,
    estimation_setup,
):
    estimation_setup["confidence_interval"] = confidence_interval
    res = estimation(**estimation_setup)

    assert res.ci_lower <= res.lower_bound
    assert res.ci_upper >= res.upper_bound


@pytest.mark.parametrize("confidence_interval", inference_methods)
def test_ci_right_coverage(
    confidence_interval: str,
    estimation_setup: dict,
    dgp_params: dict,
):
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
            sample_size=sample_size,
            rng=RNG,
            dgp_params=dgp_params,
        )
        _res = estimation(
            confidence_interval=confidence_interval,
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

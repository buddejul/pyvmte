"""Test functions for (bootstrap) confidence intervals."""

import pytest
from pyvmte.classes import Estimand
from pyvmte.config import RNG
from pyvmte.estimation import estimation
from pyvmte.utilities import simulate_data_from_simple_model_dgp


@pytest.fixture()
def estimation_setup():
    dgp_params = {
        "y1_at": 1,
        "y0_at": 0,
        "y1_c": 0.5,
        "y0_c": 0,
        "y1_nt": 0,
        "y0_nt": 0,
    }

    data = simulate_data_from_simple_model_dgp(
        sample_size=1_000,
        rng=RNG,
        dgp_params=dgp_params,
    )
    return {
        "target": Estimand("late", u_lo=0.4, u_hi=0.6, u_hi_extra=0.2),
        "identified_estimands": [Estimand("late", u_lo=0.4, u_hi=0.6)],
        "basis_func_type": "constant",
        "y_data": data["y"],
        "z_data": data["z"],
        "d_data": data["d"],
        "confidence_interval": "bootstrap",
    }


def test_estimation_with_confidence_interval_runs(estimation_setup):
    estimation(**estimation_setup)

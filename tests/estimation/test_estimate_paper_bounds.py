"""Test we consistently estimate the bounds of the paper."""
import numpy as np
import pytest
from pyvmte.config import SETUP_FIG2, SETUP_FIG3, SETUP_FIG5, Setup
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte

RNG = np.random.default_rng(495618721)


SAMPLE_SIZE = 1_000
NUM_SIMULATIONS = 250


@pytest.mark.parametrize(
    ("setup", "method"),
    [
        (SETUP_FIG2, "highs"),
        (SETUP_FIG3, "highs"),
        (SETUP_FIG5, "highs"),
        (SETUP_FIG2, "copt"),
        (SETUP_FIG3, "copt"),
        (SETUP_FIG5, "copt"),
    ],
    ids=[
        "fig2_highs",
        "fig3_highs",
        "fig5_highs",
        "fig2_copt",
        "fig3_copt",
        "fig5_copt",
    ],
)
def test_consistently_estimate_figure_bounds(setup: Setup, method: str):
    expected = [setup.lower_bound, setup.upper_bound]

    results = monte_carlo_pyvmte(
        sample_size=SAMPLE_SIZE,
        repetitions=NUM_SIMULATIONS,
        target=setup.target,
        identified_estimands=setup.identified_estimands,
        basis_func_type="constant",
        tolerance=1 / SAMPLE_SIZE,
        rng=RNG,
        method=method,
    )

    mean_upper_bound = np.mean(results["upper_bounds"])
    mean_lower_bound = np.mean(results["lower_bounds"])

    actual = [mean_lower_bound, mean_upper_bound]

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))

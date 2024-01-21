"""Test we consistently estimate the bounds of the paper."""
import numpy as np
import pandas as pd  # type: ignore
from pyvmte.estimation import estimation
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte

import pytest

from pyvmte.config import (
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
)

RNG = np.random.default_rng(495618721)


SAMPLE_SIZE = 1_000
NUM_SIMULATIONS = 250


@pytest.mark.parametrize(
    "setup",
    [(SETUP_FIG2), (SETUP_FIG3), (SETUP_FIG5)],
    ids=["fig2", "fig3", "fig5"],
)
def test_consistently_estimate_figure_bounds(setup):
    expected = [setup["lower_bound"], setup["upper_bound"]]

    results = monte_carlo_pyvmte(
        sample_size=SAMPLE_SIZE,
        repetitions=NUM_SIMULATIONS,
        target=setup["target"],
        identified_estimands=setup["identified_estimands"],
        basis_func_type="constant",
        tolerance=1 / SAMPLE_SIZE,
        rng=RNG,
    )

    mean_upper_bound = np.mean(results["upper_bounds"])
    mean_lower_bound = np.mean(results["lower_bounds"])

    actual = [mean_lower_bound, mean_upper_bound]

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))

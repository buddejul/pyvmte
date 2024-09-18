"""Test we consistently estimate the bounds reported in MST."""

import numpy as np
import pytest
from pyvmte.config import (
    RNG,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    SETUP_FIG6,
    SETUP_FIG7,
    Setup,
)
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte

SAMPLE_SIZE = 10_000
NUM_SIMULATIONS = 250


@pytest.mark.parametrize(
    ("setup", "method"),
    [
        (SETUP_FIG2, "highs"),
        (SETUP_FIG3, "highs"),
        (SETUP_FIG5, "highs"),
        (SETUP_FIG6, "highs"),
        (SETUP_FIG7, "highs"),
        (SETUP_FIG2, "copt"),
        (SETUP_FIG3, "copt"),
        (SETUP_FIG5, "copt"),
        (SETUP_FIG6, "copt"),
        (SETUP_FIG7, "copt"),
    ],
    ids=[
        "fig2_highs",
        "fig3_highs",
        "fig5_highs",
        "fig6_highs",
        "fig7_highs",
        "fig2_copt",
        "fig3_copt",
        "fig5_copt",
        "fig6_copt",
        "fig7_copt",
    ],
)
def test_consistently_estimate_bounds_mst(setup: Setup, method: str):
    expected = [setup.lower_bound, setup.upper_bound]

    if setup == SETUP_FIG7 and setup.polynomial is not None:
        basis_func_type = setup.polynomial[0]
        basis_func_options = {"k_degree": setup.polynomial[1]}
    else:
        basis_func_type = "constant"
        basis_func_options = None

    results = monte_carlo_pyvmte(
        sample_size=SAMPLE_SIZE,
        repetitions=NUM_SIMULATIONS,
        target=setup.target,
        identified_estimands=setup.identified_estimands,
        basis_func_type=basis_func_type,
        shape_constraints=setup.shape_constraints,
        tolerance=1 / SAMPLE_SIZE,
        rng=RNG,
        method=method,
        basis_func_options=basis_func_options,
    )

    mean_upper_bound = np.mean(results["upper_bounds"])
    mean_lower_bound = np.mean(results["lower_bounds"])

    actual = [mean_lower_bound, mean_upper_bound]

    assert actual == pytest.approx(expected, abs=3 / np.sqrt(SAMPLE_SIZE))

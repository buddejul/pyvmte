"""Task for running simulation by sample size."""
from itertools import product
from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore
from pytask import Product, task

from pyvmte.classes import MonteCarloSetup, Setup
from pyvmte.config import (
    BLD,
    RNG,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    SRC,
)
from pyvmte.config_mc_by_size import MC_SAMPLE_SIZES, MONTE_CARLO_BY_SIZE
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte


class _Arguments(NamedTuple):
    setup: Setup
    path_to_data: Path
    monte_carlo_setup: MonteCarloSetup


figures = [("figure2", SETUP_FIG2), ("figure3", SETUP_FIG3), ("figure5", SETUP_FIG5)]

ID_TO_KWARGS = {
    f"{fig}_{sample_size}": _Arguments(
        setup=fig[1],
        path_to_data=BLD
        / "python"
        / "data"
        / "by_sample_size"
        / Path(f"sim_results_{fig[0]}_sample_size_{sample_size}.pkl"),
        monte_carlo_setup=MONTE_CARLO_BY_SIZE._replace(sample_size=sample_size),
    )
    for sample_size, fig in product(MC_SAMPLE_SIZES, figures)
}


for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore
    def task_run_monte_carlo_simulation(
        setup: Setup,
        path_to_data: Annotated[Path, Product],
        monte_carlo_setup: MonteCarloSetup,
        config: Path = SRC / "config.py",
        config_mc_by_size: Path = SRC / "config_mc_by_size.py",
    ) -> None:
        """Run simulation for different figures in the paper."""
        tolerance = 1 / monte_carlo_setup.sample_size

        result = monte_carlo_pyvmte(
            sample_size=monte_carlo_setup.sample_size,
            repetitions=monte_carlo_setup.repetitions,
            target=setup.target,
            identified_estimands=setup.identified_estimands,
            basis_func_type="constant",
            tolerance=tolerance,
            rng=RNG,
            lp_outputs=True,
        )

        # Put upper_bounds and lower_bounds into dataframe
        bounds = {
            "upper_bound": result["upper_bound"],
            "lower_bound": result["lower_bound"],
        }
        data = pd.DataFrame(bounds)

        data.to_pickle(path_to_data)

"""Task for running simulation."""
from itertools import product
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore
from pytask import Product, task

from pyvmte.classes import MonteCarloSetup, Setup
from pyvmte.config import (
    BLD,
    MONTE_CARLO_SIMPLE,
    RNG,
    SAMPLE_SIZES,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
)
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte


class _Arguments(NamedTuple):
    setup: Setup
    path_to_data: Path
    monte_carlo_setup: MonteCarloSetup


figures = [SETUP_FIG2, SETUP_FIG3, SETUP_FIG5]

ID_TO_KWARGS = {
    f"{figure}_{sample_size}": _Arguments(
        setup=figure,
        path_to_data=BLD
        / "python"
        / "data"
        / Path(f"sim_results_{figure}_{sample_size}.pkl"),
        monte_carlo_setup=MONTE_CARLO_SIMPLE._replace(sample_size=sample_size),
    )
    for sample_size, figure in product(SAMPLE_SIZES, figures)
}


for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore
    def task_run_monte_carlo_simulation(
        setup: Setup,
        path_to_data: Annotated[Path, Product],
        monte_carlo_setup: MonteCarloSetup,
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

        # Get length of u_partition
        lens = [len(x) for x in result["u_partition"]]
        u_partition_length = max(lens)

        # Get ith elements of result["u_partition"] entries and put into data column
        for i in range(u_partition_length):
            for x in result["u_partition"]:
                if len(x) == u_partition_length:
                    data[f"u_partition_{i}"] = x[i]
                else:
                    data[f"u_partition_{i}"] = np.nan

        data.to_pickle(path_to_data)

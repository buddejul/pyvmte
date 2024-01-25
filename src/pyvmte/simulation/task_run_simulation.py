"""Task for running simulation."""
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore
from pytask import Product, task

from pyvmte.config import (
    BLD,
    RNG,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    SETUP_MONTE_CARLO,
    Setup,
)
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte


class _Arguments(NamedTuple):
    setup: Annotated[dict, "Setup for simulation DGP."]
    path_to_data: Path


ID_TO_KWARGS = {
    "figure2": _Arguments(
        setup=SETUP_FIG2,
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure2.pkl"),
    ),
    "figure3": _Arguments(
        setup=SETUP_FIG3,
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure3.pkl"),
    ),
    "figure5": _Arguments(
        setup=SETUP_FIG5,
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure5.pkl"),
    ),
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore
    def task_run_monte_carlo_simulation(
        setup: Setup,
        path_to_data: Annotated[Path, Product],
        setup_mc: dict = SETUP_MONTE_CARLO,
    ) -> None:
        """Run simulation for different figures in the paper."""
        tolerance = 1 / setup_mc["sample_size"]

        result = monte_carlo_pyvmte(
            sample_size=setup_mc["sample_size"],
            repetitions=setup_mc["repetitions"],
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

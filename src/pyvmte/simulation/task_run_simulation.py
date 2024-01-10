from pathlib import Path
from typing import Annotated, NamedTuple

from pytask import Product
from pytask import task

from pyvmte.config import BLD, SETUP_FIG2, SETUP_FIG3, SETUP_FIG5
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte

from pyvmte.config import RNG

import pandas as pd


class _Arguments(NamedTuple):
    sample_size: int
    repetitions: int
    setup: Annotated[dict, "Setup for simulation."]
    path_to_data: Path


ID_TO_KWARGS = {
    "figure2": _Arguments(
        sample_size=10_000,
        repetitions=100,
        setup=SETUP_FIG2,
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure2.pkl"),
    ),
    "figure3": _Arguments(
        sample_size=10_000,
        repetitions=100,
        setup=SETUP_FIG3,
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure3.pkl"),
    ),
    "figure5": _Arguments(
        sample_size=10_000,
        repetitions=100,
        setup=SETUP_FIG5,
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure5.pkl"),
    ),
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)
    def task_run_monte_carlo_simulation(
        sample_size: int,
        repetitions: int,
        setup: dict,
        path_to_data: Annotated[Path, Product],
    ) -> None:
        tolerance = 1 / sample_size

        result = monte_carlo_pyvmte(
            sample_size=sample_size,
            repetitions=repetitions,
            target=setup["target"],
            identified_estimands=setup["identified_estimands"],
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
        df = pd.DataFrame(bounds)

        # Get length of u_partition
        u_partition_length = len(result["u_partition"][0])

        # Get ith elements of result["u_partition"] entries and put into df column
        for i in range(u_partition_length):
            df[f"u_partition_{i}"] = [x[i] for x in result["u_partition"]]

        df.to_pickle(path_to_data)

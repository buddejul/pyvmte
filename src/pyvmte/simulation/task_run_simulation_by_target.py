from pathlib import Path
from typing import Annotated, NamedTuple

from pytask import Product
from pytask import task

from pyvmte.config import BLD, SETUP_FIG5, SETUP_MONTE_CARLO_BY_TARGET, U_HI_RANGE
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte

from pyvmte.config import RNG, Setup, Estimand

import pandas as pd  # type: ignore
import numpy as np


for u_hi_target in U_HI_RANGE:

    @task  # type: ignore
    def task_run_monte_carlo_simulation(
        setup: Setup = SETUP_FIG5,
        path_to_data: Annotated[Path, Product] = BLD
        / "python"
        / "data"
        / "by_target"
        / Path(f"sim_results_figure5_u_hi_{u_hi_target}.pkl"),
        setup_mc: dict = SETUP_MONTE_CARLO_BY_TARGET,
        u_hi_target: float = u_hi_target,
    ) -> None:
        tolerance = 1 / setup_mc["sample_size"]
        target = Estimand(type="late", u_lo=0.35, u_hi=u_hi_target)
        result = monte_carlo_pyvmte(
            sample_size=setup_mc["sample_size"],
            repetitions=setup_mc["repetitions"],
            target=target,
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
        df = pd.DataFrame(bounds)

        # TODO @buddejul Store simulation setting alongside results or in separate object

        df.to_pickle(path_to_data)

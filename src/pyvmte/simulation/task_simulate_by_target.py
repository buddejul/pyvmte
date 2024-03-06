"""Tasks for running simulation by target."""
from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore
from pytask import Product, task

from pyvmte.classes import Estimand, MonteCarloSetup, Setup
from pyvmte.config import (
    BLD,
    MONTE_CARLO_BY_TARGET,
    RNG,
    SETUP_FIG5,
    SRC,
)
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte

for u_hi_target in MONTE_CARLO_BY_TARGET.u_hi_range:  # type: ignore

    @task  # type: ignore
    def task_run_monte_carlo_simulation(
        setup: Setup = SETUP_FIG5,
        path_to_data: Annotated[Path, Product] = BLD
        / "python"
        / "data"
        / "by_target"
        / Path(f"sim_results_figure5_u_hi_{u_hi_target}.pkl"),
        setup_mc: MonteCarloSetup = MONTE_CARLO_BY_TARGET,
        u_hi_target: float = u_hi_target,
        config: Path = SRC / "config.py",
    ) -> None:
        """Run simulation by target parameter."""
        tolerance = 1 / setup_mc.sample_size
        target = Estimand(esttype="late", u_lo=0.35, u_hi=u_hi_target)
        result = monte_carlo_pyvmte(
            sample_size=setup_mc.sample_size,
            repetitions=setup_mc.repetitions,
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
        data = pd.DataFrame(bounds)

        data.to_pickle(path_to_data)

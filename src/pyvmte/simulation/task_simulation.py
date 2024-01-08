from pyvmte.config import BLD
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte
from pyvmte.simulation.simulation_plotting import plot_upper_and_lower_bounds

from pyvmte.config import RNG

import pandas as pd


def task_run_monte_carlo_simulation(
    produces=BLD / "python" / "data" / "monte_carlo_simulation_results.pkl",
):
    late_target = {"type": "late", "u_lo": 0.35, "u_hi": 0.9}
    iv_slope_target = {"type": "iv_slope"}
    ols_slope_target = {"type": "ols_slope"}
    identified_estimands = [iv_slope_target, ols_slope_target]

    sample_size = 10_000

    tolerance = 1 / sample_size

    result = monte_carlo_pyvmte(
        sample_size=sample_size,
        repetitions=1000,
        target=late_target,
        identified_estimands=identified_estimands,
        basis_func_type="constant",
        tolerance=tolerance,
        rng=RNG,
    )

    df = pd.DataFrame(result)
    df.to_pickle(produces)

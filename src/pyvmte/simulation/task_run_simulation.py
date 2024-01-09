from pyvmte.config import BLD
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte

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
    repetitions = 1_000

    tolerance = 1 / sample_size

    # TODO change target to be propensity score; still, why doesn't it work?
    # Analytical reslult of constant basis functions no longer holds
    # Should work with very small linear basis instead?
    result = monte_carlo_pyvmte(
        sample_size=sample_size,
        repetitions=repetitions,
        target=late_target,
        identified_estimands=identified_estimands,
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

    df.to_pickle(produces)

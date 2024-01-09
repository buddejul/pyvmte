"""Functions for running monte carlo simulations using pyvmte."""

import numpy as np

from pyvmte.estimation.estimation import estimation
from pyvmte.utilities import simulate_data_from_paper_dgp
from pyvmte.estimation.estimation import _estimate_prop_z


def monte_carlo_pyvmte(
    sample_size,
    repetitions,
    target,
    identified_estimands,
    basis_func_type,
    tolerance,
    rng,
    lp_outputs=False,
):
    """Run monte carlo simulation using pyvmte module."""

    upper_bounds = np.zeros(repetitions)
    lower_bounds = np.zeros(repetitions)
    if lp_outputs is True:
        minimal_deviations = np.zeros(repetitions)
        first_step_lp_inputs = []
        second_step_lp_inputs = []
        u_partitions = []
        scipy_return_first_step = []
        pscores = []

    for rep in range(repetitions):
        data = simulate_data_from_paper_dgp(sample_size=sample_size, rng=rng)

        y_data = data["y"]
        z_data = data["z"]
        d_data = data["d"]

        results = estimation(
            target,
            identified_estimands,
            basis_func_type,
            y_data,
            z_data,
            d_data,
            tolerance,
            x_data=None,
            u_partition=None,
        )

        upper_bounds[rep] = results["upper_bound"]
        lower_bounds[rep] = results["lower_bound"]
        if lp_outputs is True:
            minimal_deviations[rep] = results["minimal_deviations"]
            first_step_lp_inputs.append(results["inputs_first_step"])
            second_step_lp_inputs.append(results["inputs_second_step"])
            u_partitions.append(results["u_partition"])
            scipy_return_first_step.append(results["scipy_return_first_step"])
            pscores.append(_estimate_prop_z(z_data, d_data))

    if lp_outputs is True:
        out = {
            "upper_bound": upper_bounds,
            "lower_bound": lower_bounds,
            "minimal_deviation": minimal_deviations,
            "first_step_lp_input": first_step_lp_inputs,
            "second_step_lp_input": second_step_lp_inputs,
            "u_partition": u_partitions,
            "scipy_return_first_step": scipy_return_first_step,
            "pscore": pscores,
        }
    else:
        out = {
            "upper_bounds": upper_bounds,
            "lower_bounds": lower_bounds,
        }

    return out

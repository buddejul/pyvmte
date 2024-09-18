"""Functions for running monte carlo simulations using pyvmte."""

import numpy as np

from pyvmte.classes import Estimand
from pyvmte.estimation.estimation import _estimate_prop_z, estimation
from pyvmte.utilities import simulate_data_from_paper_dgp


def monte_carlo_pyvmte(
    sample_size: int,
    repetitions: int,
    target: Estimand,
    identified_estimands: list[Estimand],
    basis_func_type: str,
    rng: np.random.Generator,
    shape_constraints: tuple[str, str] | None = None,
    tolerance: float | None = None,
    lp_outputs: bool = False,  # noqa: FBT001, FBT002
    method: str = "highs",
    basis_func_options: dict | None = None,
) -> dict:
    """Run monte carlo simulation using pyvmte module.

    Args:
        sample_size (int): The number of observations in each sample.
        repetitions (int): The number of repetitions of the simulation.
        target (Estimand): The estimand to be estimated.
        identified_estimands (list[Estimand]): The list of identified estimands.
        basis_func_type (str): The type of basis function to use.
        rng (np.random.Generator): The random number generator.
        tolerance (float, optional): The tolerance for the optimization algorithm.
        lp_outputs (bool, optional): Whether to return the outputs from the linear
            programming algorithm.
        method (str, optional): The method to use for the optimization algorithm.
        shape_constraints: Shape constraints on MTR functions.
        basis_func_options: Options for basis functions.

    Returns:
        dict: A dictionary containing the results of the monte carlo simulation.

    """
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

        y_data = np.array(data["y"])
        z_data = np.array(data["z"])
        d_data = np.array(data["d"])

        results = estimation(
            target,
            identified_estimands,
            basis_func_type,
            y_data,
            z_data,
            d_data,
            tolerance,
            shape_constraints=shape_constraints,
            u_partition=None,
            method=method,
            basis_func_options=basis_func_options,
        )

        upper_bounds[rep] = results["upper_bound"]
        lower_bounds[rep] = results["lower_bound"]
        if lp_outputs is True:
            minimal_deviations[rep] = results["minimal_deviations"]
            first_step_lp_inputs.append(results["inputs_first_step"])
            second_step_lp_inputs.append(results["inputs_second_step"])
            u_partitions.append(results["u_partition"])
            scipy_return_first_step.append(results["scipy_return_first_step"])
            pscores.append(_estimate_prop_z(z_data, d_data, np.unique(z_data)))

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

"""The main interface of the pyvmte package."""

from pyvmte.processing import process_pyvmte_inputs
from estimation.estimation import estimation
from identification.identification import identification


def pyvmte(
    mode,
    target,
    identified_estimands,
    basis_funcs,
    m0_dgp,
    m1_dgp,
    instrument,
    u_partition,
    analytical_integration=False,
):
    """Main interface of the pyvmte package.

    Args:
        mode (str): Which mode to use (allowed values: "identificaiton", "estimation").
        target (dict): Dictionary with target estimand and relevant parameters.
        identified_estimands (dict or list of dicts): Dictionary with identified
            estimand and relevant parameters. List of dicts if multiple estimands.

    Returns:
        dict: Dictionary with results.

    """
    # processed_arguments = process_pyvmte_inputs()

    # for argname in pyvmte.__code__.co_varnames:
    #     if argname in processed_arguments:
    #         argname = processed_arguments[argname]

    if mode == "identification":
        results = identification(
            target,
            identified_estimands,
            basis_funcs,
            m0_dgp,
            m1_dgp,
            instrument,
            u_partition=u_partition,
            analytical_integration=analytical_integration,
        )
    elif mode == "estimation":
        results = estimation()

    return results

"""The main interface of the pyvmte package."""

from pyvmte.estimation.estimation import estimation
from pyvmte.identification.identification import identification


def pyvmte(
    mode,
    target,
    identified_estimands,
    basis_funcs,
    m0_dgp,
    m1_dgp,
    instrument,
    u_partition,
):
    """Main interface of the pyvmte package.

    Args:
        mode (str): Which mode to use (allowed values: "identificaiton", "estimation").
        target (dict): Dictionary with target estimand and relevant parameters.
        identified_estimands (dict or list of dicts): Dictionary with identified
            estimand and relevant parameters. List of dicts if multiple estimands.
        basis_funcs (dict or list of dicts): Dictionary with basis functions and
            relevant parameters. List of dicts if multiple basis functions.
        m0_dgp (function): The MTR function for d=0 of the DGP.
        m1_dgp (function): The MTR function for d=1 of the DGP.
        instrument (Instrument): Dictionary with instrument and relevant parameters.
        u_partition (list or np.array, optional): Partition of u for basis_funcs.
            Defaults to None.


    Returns:
        dict: Dictionary with results.

    """
    # for argname in pyvmte.__code__.co_varnames:
    #     if argname in processed_arguments:

    if mode == "identification":
        results = identification(
            target,
            identified_estimands,
            basis_funcs,
            m0_dgp,
            m1_dgp,
            instrument,
            u_partition=u_partition,
        )
    elif mode == "estimation":
        results = estimation()

    return results

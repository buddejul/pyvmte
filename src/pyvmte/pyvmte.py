"""The main interface of the pyvmte package."""

from pyvmte.processing import process_pyvmte_inputs
from estimation import estimation
from identification import identification


def pyvmte(mode):
    """Main interface of the pyvmte package.

    Args:
        mode (str): Which mode to use (allowed values: "identificaiton", "estimation").

    Returns:
        dict: Dictionary with results.

    """
    processed_arguments = process_pyvmte_inputs()

    for argname in pyvmte.__code__.co_varnames:
        if argname in processed_arguments:
            argname = processed_arguments[argname]

    if mode == "identification":
        results = identification()
    elif mode == "estimation":
        results = estimation()

    return results

"""The main interface of the pyvmte package."""

from pyvmte.processing import process_pyvmte_inputs
from pyvmte.pyvmte_estimation import pyvmte_estimation
from pyvmte.pyvmte_identification import pyvmte_identification


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
        results = pyvmte_identification()
    elif mode == "estimation":
        results = pyvmte_estimation()

    return results

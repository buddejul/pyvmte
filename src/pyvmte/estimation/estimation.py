"""Function for estimation."""
import numpy as np
from pyvmte.identification.identification import _compute_choice_weights


def estimation(
    target,
    identified_estimands,
    basis_func_type,
    y_data,
    z_data,
    d_data,
    x_data=None,
    u_partition=None,
    analytical_integration=False,
):
    """Estimate bounds on target estimand given identified estimands estimated using
    data (estimation).

    Args:
        target (dict): Dictionary containing all information about the target estimand.
        identified_estimands (dict or list of dicts): Dictionary containing all information about the identified estimand(s). List of dicts if multiple identified estimands.
        basis_func_type (str): Type of basis function.
        y_data (np.array): Array containing the outcome data.
        z_data (np.array): Array containing the instrument data.
        d_data (np.array): Array containing the treatment data.
        x_data (np.array, optional): Array containing the covariate data.
        u_partition (list or np.array, optional): Partition of u for basis_funcs. Defaults to None.
        analytical_integration (bool, optional): Whether to use analytical integration. Defaults to False.

    Returns:
        dict: A dictionary containing the estimated upper and lower bound of the target estimand.

    """
    if isinstance(identified_estimands, dict):
        identified_estimands = [identified_estimands]

    instrument = _get_instrument_supp_pdf_pscore(z_data, d_data)
    u_partition = _compute_u_partition(basis_func_type, instrument["pscore"])
    basis_funcs = _generate_basis_funcs(basis_func_type, u_partition)

    lp_inputs = {}

    lp_inputs["c"] = _compute_choice_weights(target, basis_funcs)
    lp_inputs["b_eq"] = _estimate_identified_estimands(
        identified_estimands, y_data, z_data, x_data
    )
    lp_inputs["A_eq"] = _estimate_equality_constraint_matrix(
        identified_estimands, basis_funcs, instrument=instrument
    )

    upper_bound = (-1) * _solve_lp_estimation(lp_inputs, "max").fun
    lower_bound = _solve_lp_estimation(lp_inputs, "min").fun

    return {"upper_bound": upper_bound, "lower_bound": lower_bound}


def _solve_lp_estimation(lp_inputs, min_or_max):
    """Solve linear program for estimation."""
    pass


def _get_instrument_supp_pdf_pscore(z_data, d_data):
    """Estimate the support, marginal density and propensity score of instrument z and
    treatment d."""
    out = {}
    out["pdf"] = _estimate_instrument_pdf(z_data)
    out["pscore"] = _estimate_instrument_pscore(z_data, d_data)
    out["support"] = np.unique(z_data)

    return out


def _estimate_instrument_pdf(z_data):
    """Estimate the marginal density of instrument z."""
    pass


def _estimate_instrument_pscore(z_data, d_data):
    """Estimate the propensity score of instrument z and treatment d."""
    pass


def _compute_u_partition(basis_func_type, z_pscore):
    """Compute the partition of u based on type of basis function and pscore of z."""
    u_partition = np.array([0, 1])
    return u_partition


def _generate_basis_funcs(basis_func_type, u_partition):
    """Generate the basis functions."""
    list_of_basis_funcs = []

    return list_of_basis_funcs


def _estimate_identified_estimands(identified_estimands, y_data, z_data, x_data):
    """Estimate the identified estimands."""
    pass


def _estimate_equality_constraint_matrix():
    """Estimate the equality constraint matrix."""
    pass

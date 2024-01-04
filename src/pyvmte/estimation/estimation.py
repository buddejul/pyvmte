"""Function for estimation."""
import numpy as np
from pyvmte.identification.identification import _compute_choice_weights
from pyvmte.utilities import s_cross, s_iv_slope, s_late, s_ols_slope


def estimation(
    target,
    identified_estimands,
    basis_func_type,
    y_data,
    z_data,
    d_data,
    tolerance,
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
        tolerance (float): Tolerance for the second-step linear program.
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

    lp_first_inputs = {}
    # The program is of the form
    # min sum_s |tau_s(m) - beta_s|
    # subject to basis funcs in [0, 1]
    #
    lp_first_inputs["c"] = _estimate_first_lp_weights(targets, basis_funcs)
    lp_first_inputs["b_eq"] = _estimate_identified_estimands(
        identified_estimands, y_data, z_data, d_data
    )
    lp_first_inputs["A_eq"] = []

    # First step linear program to find minimal deviations in constraint
    minimal_deviations = _solve_first_step_lp_estimation()

    # Second step linear program
    lp_second_inputs = {}
    lp_second_inputs["b_eq"] = minimal_deviations + tolerance
    lp_second_inputs["c"] = _compute_choice_weights(target, basis_funcs)
    lp_second_inputs["A_eq"] = _estimate_equality_constraint_matrix(
        identified_estimands, basis_funcs, instrument=instrument
    )

    upper_bound = (-1) * _solve_second_step_lp_estimation(lp_second_inputs, "max").fun
    lower_bound = _solve_second_step_lp_estimation(lp_second_inputs, "min").fun

    return {"upper_bound": upper_bound, "lower_bound": lower_bound}


def _solve_first_step_lp_estimation(lp_first_inputs):
    """Solve first step linear program to get minimal deviations in constraint."""
    pass


def _solve_second_step_lp_estimation(lp_second_inputs, min_or_max):
    """Solve for upper/lower bound given minimal deviations from first step."""
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


def _estimate_identified_estimands(identified_estimands, y_data, z_data, d_data):
    """Estimate the identified estimands."""
    list_of_estimands = []
    for estimand in identified_estimands:
        result = _estimate_estimand(estimand, y_data, z_data, d_data)
        list_of_estimands.append(result)

    return list_of_estimands


def _estimate_estimand(estimand, y_data, z_data, d_data):
    """Estimate single identified estimand based on data."""

    if estimand["type"] == "late":
        pass
        # sfunc = lambda u: s_late(u, estimand["u_lo"], estimand["u_hi"])

    elif estimand["type"] == "cross":
        ind_elements = s_cross(d_data, z_data, estimand["dz_cross"]) * y_data

    elif estimand["type"] == "iv_slope":
        ez = np.mean(z_data)
        cov_dz = np.cov(d_data, z_data)[0, 1]
        ind_elements = s_iv_slope(z_data, ez=ez, cov_dz=cov_dz) * y_data

    elif estimand["type"] == "ols_slope":
        ed = np.mean(d_data)
        var_d = np.var(d_data)
        ind_elements = s_ols_slope(d_data, ed=ed, var_d=var_d) * y_data

    return np.mean(ind_elements)


def _estimate_equality_constraint_matrix():
    """Estimate the equality constraint matrix."""
    pass


def _estimate_first_lp_weights(
    identified_estimands, basis_funcs, y_data, z_data, d_data
):
    """Estimate the choice weights for a set of identified estimands by computing the
    weight matrix associated with each estimand and using linearity."""

    weights = np.zeros(len(basis_funcs))

    for estimand in identified_estimands:
        weight_matrix += _estimate_weights_estimand(
            estimand, basis_funcs, z_data, d_data
        )

    return weights


def _estimate_weights_estimand(estimand, basis_funcs, z_data, d_data):
    """Estimate the weights on each basis function for a single estimand."""

    estimand_type = estimand["type"]

    z_p = _generate_array_of_p_scores(z_data, d_data)

    if estimand_type == "ols_slope":
        var_d = np.var(d_data)
        ed = np.mean(d_data)

        def s(d, z):
            return s_ols_slope(d, ed=ed, var_d=var_d)

    if estimand_type == "iv_slope":
        ez = np.mean(z_data)
        ed = np.mean(d_data)
        edz = np.mean(d_data * z_data)
        cov_dz = edz - ed * ez

        def s(d, z):
            return s_iv_slope(z, ez=ez, cov_dz=cov_dz)

    if estimand_type == "cross":

        def s(d, z):
            return s_cross(d, z, dz_cross=estimand["dz_cross"])

    weights = []

    for basis_func in basis_funcs:
        d_value = basis_func["d_value"]
        u_lo = basis_func["u_lo"]
        u_hi = basis_func["u_hi"]

        # TODO potential bug: u_hi - u_lo for both cases?
        if d_value == 0:
            val = np.mean(
                # FIXME check whether d_val or d_data here
                (u_lo > z_p)
                * s(d_data, z_data)
                * (u_hi - u_lo)
            )

        elif d_value == 1:
            val = np.mean(
                # FIXME check whether d_val or d_data here
                (u_hi <= z_p)
                * s(d_data, z_data)
                * (u_hi - u_lo)
            )

        weights.append(val)

    return weights


def _generate_array_of_p_scores(z_data, d_data):
    """For input data on instrment and treatment generates array of same length with
    estimated propensity scores for each corresponding entry of z."""
    pass

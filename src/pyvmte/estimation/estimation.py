"""Function for estimation."""
import numpy as np
from pyvmte.identification.identification import _compute_choice_weights
from pyvmte.utilities import s_cross, s_iv_slope, s_late, s_ols_slope

from scipy.optimize import linprog


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

    minimal_deviations = _first_step_linear_program()

    # Second step linear program

    results = _second_step_linear_program()

    return {"upper_bound": upper_bound, "lower_bound": lower_bound}


def _first_step_linear_program(identified_estimands, basis_funcs, d_data, z_data):
    """First step linear program to get minimal deviations in constraint."""
    lp_first_inputs = {}
    lp_first_inputs["c"] = np.hstack(
        (np.zeros(len(basis_funcs)), np.ones(len(identified_estimands)))
    )
    lp_first_inputs["b_ub"] = np.zeros(len(identified_estimands) * 2)
    lp_first_inputs["A_ub"] = _build_first_step_ub_matrix(
        basis_funcs, identified_estimands, d_data, z_data
    )
    lp_first_inputs["bounds"] = _compute_first_step_bounds(
        identified_estimands, basis_funcs
    )

    # First step linear program to find minimal deviations in constraint
    return _solve_first_step_lp_estimation(lp_first_inputs)


def _solve_first_step_lp_estimation(lp_first_inputs):
    """Solve first-step linear program."""
    result = linprog(
        c=lp_first_inputs["c"],
        A_ub=lp_first_inputs["A_ub"],
        b_ub=lp_first_inputs["b_ub"],
        bounds=lp_first_inputs["bounds"],
    )

    return result.fun


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
    """Generate list of dictionaries describing basis functions."""
    bfuncs_list = []

    if basis_func_type is "constant":
        for d_val in [0, 1]:
            for u_lo, u_hi in zip(u_partition[:-1], u_partition[1:]):
                bfuncs_list.append({"d_value": d_val, "u_lo": u_lo, "u_hi": u_hi})

    return bfuncs_list


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


def _estimate_first_lp_weights(identified_estimands, basis_funcs, z_data, d_data):
    """Estimate the choice weights for a set of identified estimands by computing the
    weight matrix associated with each estimand and using linearity."""

    weights = np.zeros(len(basis_funcs))

    for estimand in identified_estimands:
        weights += _estimate_weights_estimand(estimand, basis_funcs, z_data, d_data)

    return weights


def _estimate_weights_estimand(estimand, basis_funcs, z_data, d_data):
    """Estimate the weights on each basis function for a single estimand."""

    estimand_type = estimand["type"]

    z_p = _generate_array_of_pscores(z_data, d_data)

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


def _generate_array_of_pscores(z_data, d_data):
    """For input data on instrument and treatment generates array of same length with
    estimated propensity scores for each corresponding entry of z."""

    # Estimate propensity scores
    p = _estimate_prop_z(z_data, d_data)

    # Get vector of p corresponding to z
    supp_z = np.unique(z_data)
    return p[np.searchsorted(supp_z, z_data)]


def _estimate_prop_z(z_data, d_data):
    """Estimate propensity score of z given d."""

    supp_z = np.unique(z_data)

    pscore = []

    for z_val in supp_z:
        pscore.append(np.mean(d_data[z_data == z_val]))

    return np.array(pscore)


def _build_first_step_ub_matrix(basis_funcs, identified_estimands, d_data, z_data):
    """Build matrix for first step lp involving dummy variables."""
    num_bfuncs = len(basis_funcs)
    num_idestimands = len(identified_estimands)

    weight_matrix = np.empty(shape=(num_idestimands, num_bfuncs))

    for i, estimand in enumerate(identified_estimands):
        weights = _estimate_weights_estimand(estimand, basis_funcs, z_data, d_data)

        weight_matrix[i, :] = weights

    top = np.hstack((weight_matrix, -np.eye(num_idestimands)))
    bottom = np.hstack((-weight_matrix, -np.eye(num_idestimands)))

    out = np.vstack((top, bottom))

    return out


def _compute_first_step_bounds(identified_estimands, basis_funcs):
    """Generate list of tuples containing bounds for first step linear program."""
    num_idestimands = len(identified_estimands)
    num_bfuncs = len(basis_funcs)

    return [(0, 1) for _ in range(num_bfuncs)] + [
        (None, None) for _ in range(num_idestimands)
    ]


def _second_step_linear_program(
    target, identified_estimands, basis_funcs, minimal_deviations, tolerance
):
    """Second step linear program to estimate upper and lower bounds."""

    lp_second_inputs = {}
    lp_second_inputs["c"] = _compute_choice_weights_second_step(
        target, basis_funcs, identified_estimands
    )
    lp_second_inputs["b_ub"] = np.append(
        minimal_deviations + tolerance, np.zeros(2 * len(identified_estimands))
    )
    lp_second_inputs["A_ub"] = _build_second_step_ub_matrix(
        identified_estimands, basis_funcs, instrument=instrument
    )

    upper_bound = (-1) * _solve_second_step_lp_estimation(lp_second_inputs, "max").fun
    lower_bound = _solve_second_step_lp_estimation(lp_second_inputs, "min").fun

    return {"upper_bound": upper_bound, "lower_bound": lower_bound}


def _compute_choice_weights_second_step(target, basis_funcs, identified_estimands):
    """Compute choice weight vector c for second step linear program."""

    list_of_funcs = _create_funcs_from_dicts(basis_funcs)
    upper_part = _compute_choice_weights(target, basis_funcs=list_of_funcs)
    upper_part = np.array(upper_part)

    lower_part = np.zeros(len(identified_estimands))
    return np.append(upper_part, lower_part)


def _create_funcs_from_dicts(basis_funcs):
    """Create list of functions from list of dictionaries for constant splines."""

    def create_function(u_lo, u_hi):
        def f(x):
            return 1 if u_lo <= x < u_hi else 0

        return f

    relevant_bfuncs = basis_funcs[: int(len(basis_funcs) / 2)]

    list_of_funcs = []
    for bfunc in relevant_bfuncs:
        func = create_function(bfunc["u_lo"], bfunc["u_hi"])
        list_of_funcs.append(func)

    return list_of_funcs


def _build_second_step_ub_matrix(basis_funcs, identified_estimands, z_data, d_data):
    """Build A_ub matrix for second step linear program."""

    num_idestimands = len(identified_estimands)
    num_bfuncs = len(basis_funcs)

    first_row = np.append(np.zeros(num_bfuncs), np.ones(num_idestimands))

    weight_matrix = np.empty(shape=(num_idestimands, num_bfuncs))

    for i, estimand in enumerate(identified_estimands):
        weights = _estimate_weights_estimand(estimand, basis_funcs, z_data, d_data)

        weight_matrix[i, :] = weights

    top = np.hstack((weight_matrix, -np.eye(num_idestimands)))
    bottom = np.hstack((-weight_matrix, -np.eye(num_idestimands)))

    out = np.vstack((top, bottom))
    out = np.vstack((first_row, out))

    return out

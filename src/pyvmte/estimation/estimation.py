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

    instrument = _estimate_instrument_characteristics(z_data, d_data)
    u_partition = _compute_u_partition(
        target=target,
        identified_estimands=identified_estimands,
        pscore_z=instrument["pscore_z"],
    )
    basis_funcs = _generate_basis_funcs(basis_func_type, u_partition)

    beta_hat = _estimate_identified_estimands(
        identified_estimands=identified_estimands,
        y_data=y_data,
        z_data=z_data,
        d_data=d_data,
    )

    beta_hat = np.array(beta_hat)

    results_first_step = _first_step_linear_program(
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        y_data=y_data,
        d_data=d_data,
        z_data=z_data,
        beta_hat=beta_hat,
    )

    minimal_deviations = results_first_step["minimal_deviations"]

    results_second_step = _second_step_linear_program(
        target=target,
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        z_data=z_data,
        d_data=d_data,
        minimal_deviations=minimal_deviations,
        tolerance=tolerance,
        beta_hat=beta_hat,
    )

    out = {
        "upper_bound": results_second_step["upper_bound"],
        "lower_bound": results_second_step["lower_bound"],
        "minimal_deviations": minimal_deviations,
        "u_partition": u_partition,
        "beta_hat": beta_hat,
        "inputs_first_step": results_first_step["inputs"],
        "inputs_second_step": results_second_step["inputs"],
        "scipy_return_first_step": results_first_step["scipy_return"],
        "scipy_return_second_step_upper": results_second_step["scipy_return_upper"],
        "scipy_return_second_step_lower": results_second_step["scipy_return_lower"],
    }

    return out


def _first_step_linear_program(
    identified_estimands, basis_funcs, y_data, d_data, z_data, beta_hat
):
    """First step linear program to get minimal deviations in constraint."""
    num_bfuncs = len(basis_funcs) * 2
    lp_first_inputs = {}
    lp_first_inputs["c"] = np.hstack(
        (np.zeros(num_bfuncs), np.ones(len(identified_estimands)))
    )
    lp_first_inputs["b_ub"] = _compute_first_step_upper_bounds(beta_hat)
    lp_first_inputs["A_ub"] = _build_first_step_ub_matrix(
        basis_funcs, identified_estimands, d_data, z_data
    )
    lp_first_inputs["bounds"] = _compute_first_step_bounds(
        identified_estimands, basis_funcs
    )

    first_step_solution = _solve_first_step_lp_estimation(lp_first_inputs)

    minimal_deviations = first_step_solution.fun

    out = {
        "minimal_deviations": minimal_deviations,
        "scipy_return": first_step_solution,
        "inputs": lp_first_inputs,
    }

    return out


def _solve_first_step_lp_estimation(lp_first_inputs):
    """Solve first-step linear program."""
    result = linprog(
        c=lp_first_inputs["c"],
        A_ub=lp_first_inputs["A_ub"],
        b_ub=lp_first_inputs["b_ub"],
        bounds=lp_first_inputs["bounds"],
    )

    return result


def _solve_second_step_lp_estimation(lp_second_inputs, min_or_max):
    """Solve for upper/lower bound given minimal deviations from first step."""
    if min_or_max == "min":
        c = lp_second_inputs["c"]
    else:
        c = -lp_second_inputs["c"]

    result = linprog(
        c=c,
        A_ub=lp_second_inputs["A_ub"],
        b_ub=lp_second_inputs["b_ub"],
        bounds=lp_second_inputs["bounds"],
    )

    return result


def _estimate_instrument_pdf(z_data):
    """Estimate the marginal density of instrument z."""

    supp_z = np.unique(z_data)

    pdf_z = []

    for z_val in supp_z:
        pdf_z.append(np.mean(z_data == z_val))

    return np.array(pdf_z)


def _estimate_instrument_pscore(z_data, d_data):
    """Estimate the propensity score of instrument z and treatment d."""
    pass


def _compute_u_partition(target, pscore_z, identified_estimands=None):
    """Compute the partition of u based on identified, target estimands, and pscore of
    z."""
    knots = [0, 1]

    if target["type"] == "late":
        knots.append(target["u_lo"])
        knots.append(target["u_hi"])

    # Add p_score to list
    knots.extend(pscore_z)

    return np.unique(knots)


def _generate_basis_funcs(basis_func_type, u_partition):
    """Generate list of dictionaries describing basis functions."""
    bfuncs_list = []

    if basis_func_type == "constant":
        for u_lo, u_hi in zip(u_partition[:-1], u_partition[1:]):
            bfuncs_list.append({"type": "constant", "u_lo": u_lo, "u_hi": u_hi})

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


def _estimate_weights_estimand(estimand, basis_funcs, z_data, d_data):
    """Estimate the weights on each basis function for a single estimand."""

    moments = _estimate_moments_for_weights(estimand, z_data, d_data)
    instrument = _estimate_instrument_characteristics(z_data, d_data)

    weights = _compute_choice_weights(
        target=estimand,
        basis_funcs=basis_funcs,
        instrument=instrument,
        moments=moments,
    )

    # Get length of basis_funcs intervals using u_lo and u_hi key
    lengths = np.array([bfunc["u_hi"] - bfunc["u_lo"] for bfunc in basis_funcs])

    # Multiply weights by lengths to get weights on choice variables
    # repeating lengths 2x
    weights = weights * np.tile(lengths, 2)

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
    num_bfuncs = len(basis_funcs) * 2
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
    num_bfuncs = len(basis_funcs) * 2

    return [(0, 1) for _ in range(num_bfuncs)] + [
        (None, None) for _ in range(num_idestimands)
    ]


def _second_step_linear_program(
    target,
    identified_estimands,
    basis_funcs,
    z_data,
    d_data,
    minimal_deviations,
    tolerance,
    beta_hat,
):
    """Second step linear program to estimate upper and lower bounds."""

    lp_second_inputs = {}
    lp_second_inputs["c"] = _compute_choice_weights_second_step(
        target, basis_funcs, identified_estimands
    )
    lp_second_inputs["b_ub"] = _compute_second_step_upper_bounds(
        minimal_deviations=minimal_deviations, tolerance=tolerance, beta_hat=beta_hat
    )
    lp_second_inputs["A_ub"] = _build_second_step_ub_matrix(
        basis_funcs=basis_funcs,
        identified_estimands=identified_estimands,
        z_data=z_data,
        d_data=d_data,
    )
    lp_second_inputs["bounds"] = _compute_second_step_bounds(
        basis_funcs, identified_estimands
    )

    result_upper = _solve_second_step_lp_estimation(lp_second_inputs, "max")
    result_lower = _solve_second_step_lp_estimation(lp_second_inputs, "min")

    if result_upper.success == False or result_lower.success == False:
        raise ValueError(
            f"Failed to solve linear program: upper {result_upper.success}, lower {result_lower.success}."
        )

    out = {
        "upper_bound": -1 * result_upper.fun,
        "lower_bound": result_lower.fun,
        "inputs": lp_second_inputs,
        "scipy_return_upper": result_upper,
        "scipy_return_lower": result_lower,
    }

    return out


def _compute_choice_weights_second_step(target, basis_funcs, identified_estimands):
    """Compute choice weight vector c for second step linear program."""

    upper_part = _compute_choice_weights(target, basis_funcs=basis_funcs)
    upper_part = np.array(upper_part)

    lower_part = np.zeros(len(identified_estimands))
    return np.append(upper_part, lower_part)


def _create_funcs_from_dicts(basis_funcs):
    """Create list of functions from list of dictionaries for constant splines."""

    def create_function(u_lo, u_hi):
        def f(x):
            return 1 if u_lo <= x < u_hi else 0

        return f

    list_of_funcs = []
    for bfunc in basis_funcs:
        func = create_function(bfunc["u_lo"], bfunc["u_hi"])
        list_of_funcs.append(func)

    return list_of_funcs


def _build_second_step_ub_matrix(basis_funcs, identified_estimands, z_data, d_data):
    """Build A_ub matrix for second step linear program."""

    num_idestimands = len(identified_estimands)
    num_bfuncs = len(basis_funcs) * 2

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


def _compute_second_step_bounds(basis_funcs, identified_estimands):
    """Compute bounds for second step linear program."""

    num_bfuncs = len(basis_funcs) * 2
    num_idestimands = len(identified_estimands)

    return [(0, 1) for _ in range(num_bfuncs)] + [
        (None, None) for _ in range(num_idestimands)
    ]


def _compute_first_step_upper_bounds(beta_hat):
    """Compute b_ub vector with upper bounds of ineq constraint in first step LP."""

    b_ub = np.append(beta_hat, -beta_hat)
    return b_ub


def _compute_second_step_upper_bounds(minimal_deviations, tolerance, beta_hat):
    """Compute b_ub vector with upper bounds of ineq constraint in second step LP."""
    return np.append(
        np.array(minimal_deviations + tolerance), np.append(beta_hat, -beta_hat)
    )


def _estimate_moments_for_weights(estimand, z_data, d_data):
    """Estimate relevant moments for computing weights on LP choice variables."""

    moments = {}

    if estimand["type"] == "ols_slope":
        moments["expectation_d"] = np.mean(d_data)
        moments["variance_d"] = np.var(d_data)

    elif estimand["type"] == "iv_slope":
        moments["expectation_z"] = np.mean(z_data)
        moments["covariance_dz"] = np.cov(d_data, z_data)[0, 1]

    return moments


def _estimate_instrument_characteristics(z_data, d_data):
    """Estimate relevant characteristics of instrument z."""

    instrument = {}

    instrument["pscore_z"] = _estimate_prop_z(z_data, d_data)
    instrument["support_z"] = np.unique(z_data)
    instrument["pdf_z"] = _estimate_instrument_pdf(z_data)

    return instrument

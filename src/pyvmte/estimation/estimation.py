"""Function for estimation."""

from itertools import pairwise

import coptpy as cp  # type: ignore
import numpy as np
from coptpy import COPT
from scipy.optimize import (  # type: ignore
    OptimizeResult,
    linprog,  # type: ignore
)

from pyvmte.classes import Estimand, Instrument
from pyvmte.identification.identification import _compute_choice_weights
from pyvmte.utilities import (
    _error_report_estimand,
    _error_report_estimation_data,
    _error_report_invalid_basis_func_type,
    _error_report_method,
    _error_report_shape_constraints,
    _error_report_tolerance,
    _error_report_u_partition,
    s_cross,
    s_iv_slope,
    s_ols_slope,
    suppress_print,
)


def estimation(
    target: Estimand,
    identified_estimands: list[Estimand],
    basis_func_type: str,
    y_data: np.ndarray,
    z_data: np.ndarray,
    d_data: np.ndarray,
    tolerance: float | None = None,
    u_partition: np.ndarray | None = None,
    shape_constraints: tuple[str, str] | None = None,
    method: str = "highs",
):
    """Estimate bounds given target, identified estimands, and data (estimation).

    Args:
        target (dict): Dictionary containing all information about the target estimand.
        identified_estimands (dict or list of dicts): Dictionary containing all
        information about the identified estimand(s). List of dicts if multiple
        identified estimands.
        basis_func_type (str): Type of basis function.
        y_data (np.array): Array containing the outcome data.
        z_data (np.array): Array containing the instrument data.
        d_data (np.array): Array containing the treatment data.
        tolerance (float, optional): Tolerance for the second-step linear program.
        The default is 1 / sample_size.
        u_partition (list or np.array, optional): Partition of u for basis_funcs.
        Defaults to None.
        shape_constraints: Shape constraints for the MTR functions.
        method (str, optional): Method for scipy linprog solver. Default highs.

    Returns:
        dict: A dictionary containing the estimated upper and lower bound of the target
        estimand.

    """
    if isinstance(identified_estimands, Estimand):
        identified_estimands = [identified_estimands]

    _check_estimation_arguments(
        target,
        identified_estimands,
        basis_func_type,
        y_data,
        z_data,
        d_data,
        tolerance,
        u_partition,
        shape_constraints,
        method,
    )

    # ==================================================================================
    # Preliminary Computations (Partitions, Basis Functions, Instrument, Estimates)
    # ==================================================================================
    if tolerance is None:
        tolerance = 1 / len(y_data)

    instrument = _estimate_instrument_characteristics(z_data, d_data)

    u_partition = _compute_u_partition(
        target=target,
        pscore_z=instrument.pscores,
    )

    basis_funcs = _generate_basis_funcs(basis_func_type, u_partition)

    beta_hat = _estimate_identified_estimands(
        identified_estimands=identified_estimands,
        y_data=y_data,
        z_data=z_data,
        d_data=d_data,
    )

    data = {"y": y_data, "z": z_data, "d": d_data}

    data["pscores"] = _generate_array_of_pscores(
        z_data=z_data,
        support=instrument.support,
        pscores=instrument.pscores,
    )

    # ==================================================================================
    # First Step Linear Program (Compute Minimal Deviations)
    # ==================================================================================
    results_first_step = _first_step_linear_program(
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        data=data,
        beta_hat=beta_hat,
        instrument=instrument,
        shape_constraints=shape_constraints,
        method=method,
    )

    minimal_deviations = results_first_step["minimal_deviations"]

    # ==================================================================================
    # Second Step Linear Program (Compute Upper and Lower Bounds)
    # ==================================================================================
    results_second_step = _second_step_linear_program(
        target=target,
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        data=data,
        minimal_deviations=minimal_deviations,
        tolerance=tolerance,
        beta_hat=beta_hat,
        instrument=instrument,
        shape_constraints=shape_constraints,
        method=method,
    )

    # ==================================================================================
    # Return Results
    # ==================================================================================
    if method == "copt":
        return {
            "upper_bound": results_second_step["upper_bound"],
            "lower_bound": results_second_step["lower_bound"],
        }

    return {
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


def _first_step_linear_program(
    identified_estimands: list[Estimand],
    basis_funcs: list[dict],
    data: dict[str, np.ndarray],
    beta_hat: np.ndarray,
    instrument: Instrument,
    shape_constraints: tuple[str, str] | None,
    method: str,
) -> dict:
    """First step linear program to get minimal deviations in constraint."""
    num_bfuncs = len(basis_funcs) * 2
    lp_first_inputs = {}
    lp_first_inputs["c"] = np.hstack(
        (np.zeros(num_bfuncs), np.ones(len(identified_estimands))),
    )
    lp_first_inputs["b_ub"] = _compute_first_step_upper_bounds(beta_hat)
    lp_first_inputs["a_ub"] = _build_first_step_ub_matrix(
        basis_funcs,
        identified_estimands,
        data,
        instrument,
    )
    if shape_constraints is not None:
        lp_first_inputs["a_ub"] = _append_shape_constraints_a_ub(
            shape_constraints,
            lp_first_inputs["a_ub"],
            num_bfuncs,
            num_idestimands=len(identified_estimands),
        )
        lp_first_inputs["b_ub"] = _append_shape_constraints_b_ub(
            shape_constraints,
            lp_first_inputs["b_ub"],
            num_bfuncs,
        )

    lp_first_inputs["bounds"] = _compute_first_step_bounds(
        identified_estimands,
        basis_funcs,  # type: ignore
    )
    if method == "copt":
        first_step_solution = None
        minimal_deviations = _solve_lp_estimation_copt(lp_first_inputs, "min")
    else:
        first_step_solution = _solve_first_step_lp_estimation(lp_first_inputs, method)
        minimal_deviations = first_step_solution.fun

    return {
        "minimal_deviations": minimal_deviations,
        "scipy_return": first_step_solution,
        "inputs": lp_first_inputs,
    }


def _solve_first_step_lp_estimation(
    lp_first_inputs: dict,
    method: str,
) -> OptimizeResult:
    """Solve first-step linear program."""
    return linprog(
        c=lp_first_inputs["c"],
        A_ub=lp_first_inputs["a_ub"],
        b_ub=lp_first_inputs["b_ub"],
        bounds=lp_first_inputs["bounds"],
        method=method,
    )


def _solve_second_step_lp_estimation(
    lp_second_inputs: dict,
    min_or_max: str,
    method: str,
) -> OptimizeResult:
    """Solve for upper/lower bound given minimal deviations from first step."""
    c = lp_second_inputs["c"] if min_or_max == "min" else -lp_second_inputs["c"]

    return linprog(
        c=c,
        A_ub=lp_second_inputs["a_ub"],
        b_ub=lp_second_inputs["b_ub"],
        bounds=lp_second_inputs["bounds"],
        method=method,
    )


def _estimate_instrument_pdf(z_data: np.ndarray, support) -> np.ndarray:
    """Estimate the marginal density of instrument z."""
    pdf_z = []

    for z_val in support:
        pdf_z.append(np.mean(z_data == z_val))

    return np.array(pdf_z)


def _compute_u_partition(
    target: Estimand,
    pscore_z: np.ndarray,
) -> np.ndarray:
    """Compute the partition of u based on target estimand and pscore of z."""
    knots = np.array([0, 1])

    if target.esttype == "late":
        knots = np.append(knots, target.u_lo)  # type: ignore
        knots = np.append(knots, target.u_hi)  # type: ignore

    # Add p_score to list
    knots = np.append(knots, pscore_z)

    return np.unique(knots)


def _generate_basis_funcs(basis_func_type: str, u_partition: np.ndarray) -> list:
    """Generate list of dictionaries describing basis functions."""
    bfuncs_list = []

    if basis_func_type == "constant":
        for u_lo, u_hi in pairwise(u_partition):
            bfuncs_list.append({"type": "constant", "u_lo": u_lo, "u_hi": u_hi})

    return bfuncs_list


def _estimate_identified_estimands(
    identified_estimands: list[Estimand],
    y_data: np.ndarray,
    z_data: np.ndarray,
    d_data: np.ndarray,
) -> np.ndarray:
    """Estimate the identified estimands."""
    list_of_estimands = []
    for estimand in identified_estimands:
        result = _estimate_estimand(estimand, y_data, z_data, d_data)
        list_of_estimands.append(result)
    return np.array(list_of_estimands)


def _estimate_estimand(
    estimand: Estimand,
    y_data: np.ndarray,
    z_data: np.ndarray,
    d_data: np.ndarray,
) -> float:
    """Estimate single identified estimand based on data."""
    if estimand.esttype == "late":
        pass

    elif estimand.esttype == "cross":
        ind_elements = s_cross(d_data, z_data, estimand.dz_cross) * y_data

    elif estimand.esttype == "iv_slope":
        ez = np.mean(z_data)
        cov_dz = np.cov(d_data, z_data)[0, 1]
        ind_elements = s_iv_slope(z_data, ez=ez, cov_dz=cov_dz) * y_data

    elif estimand.esttype == "ols_slope":
        ed = np.mean(d_data)
        var_d = np.var(d_data)
        ind_elements = s_ols_slope(d_data, ed=ed, var_d=var_d) * y_data

    return np.mean(ind_elements)


def _estimate_weights_estimand(
    estimand: Estimand,
    basis_funcs: list,
    data: dict[str, np.ndarray],
    moments: dict,
    instrument: Instrument,
) -> np.ndarray:
    """Estimate the weights on each basis function for a single estimand."""
    weights = np.zeros(len(basis_funcs) * 2)

    for d_value in [0, 1]:
        for i, basis_func in enumerate(basis_funcs):
            idx = i + d_value * len(basis_funcs)
            weights[idx] = _estimate_gamma_for_basis_funcs(
                d_value=d_value,
                estimand=estimand,
                basis_func=basis_func,
                data=data,
                moments=moments,
                instrument=instrument,
            )

    return weights


def _generate_array_of_pscores(
    z_data: np.ndarray,
    support: np.ndarray,
    pscores: np.ndarray,
) -> np.ndarray:
    """Generate array of propensity scores corresponding to instrument array."""
    idx = np.searchsorted(support, z_data)

    return pscores[idx]


def _estimate_prop_z(
    z_data: np.ndarray,
    d_data: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Estimate propensity score of z given d."""
    pscore = []

    for z_val in support:
        pscore.append(np.mean(d_data[z_data == z_val]))

    return np.array(pscore)


def _build_first_step_ub_matrix(
    basis_funcs: list[dict],
    identified_estimands: list[Estimand],
    data: dict[str, np.ndarray],
    instrument: Instrument,
) -> np.ndarray:
    """Build matrix for first step lp involving dummy variables."""
    num_bfuncs = len(basis_funcs) * 2
    num_idestimands = len(identified_estimands)

    weight_matrix = np.empty(shape=(num_idestimands, num_bfuncs))

    moments = _estimate_moments_for_weights(data["z"], data["d"])

    for i, estimand in enumerate(identified_estimands):
        weights = _estimate_weights_estimand(
            estimand=estimand,
            basis_funcs=basis_funcs,
            data=data,
            moments=moments,
            instrument=instrument,
        )

        weight_matrix[i, :] = weights

    top = np.hstack((weight_matrix, -np.eye(num_idestimands)))
    bottom = np.hstack((-weight_matrix, -np.eye(num_idestimands)))

    return np.vstack((top, bottom))


def _compute_first_step_bounds(
    identified_estimands: list[Estimand],
    basis_funcs: list[dict],
) -> list:
    """Generate list of tuples containing bounds for first step linear program."""
    num_idestimands = len(identified_estimands)
    num_bfuncs = len(basis_funcs) * 2

    return [(0, 1) for _ in range(num_bfuncs)] + [
        (None, None) for _ in range(num_idestimands)
    ]


def _append_shape_constraints_a_ub(
    shape_constraints: tuple[str, str],
    a_ub,
    num_bfuncs: int,
    num_idestimands: int | None = None,
) -> np.ndarray:
    """Append shape constraints to a_ub matrix.

    num_idestimands needs to be supplied for the first step linear program.

    """
    a = np.eye(num_bfuncs - 1, num_bfuncs)
    b = np.eye(num_bfuncs - 1, num_bfuncs, 1)

    to_append = a - b

    # Now we need to delete the (num_bfuncs)th row, so we don't put cross-
    # restrictions on the MTR d = 0 and d = 1 functions.
    to_delete = int(num_bfuncs / 2 - 1)
    to_append = np.delete(to_append, to_delete, axis=0)

    # Add a matrix of zeros to the right with the same rows and num_idestimands columns
    # to match the first step linear program structure.
    if num_idestimands is not None:
        to_append = np.hstack(
            (to_append, np.zeros((to_append.shape[0], num_idestimands))),
        )

    if shape_constraints == ("increasing", "increasing"):
        return np.vstack((a_ub, to_append))

    if shape_constraints == ("decreasing", "decreasing"):
        return np.vstack((a_ub, -to_append))

    msg = "Invalid shape constraints."
    raise ValueError(msg)


def _append_shape_constraints_b_ub(
    shape_constraints: tuple[str, str],
    b_ub: np.ndarray,
    num_bfuncs: int,
) -> np.ndarray:
    """Append shape constraints to b_ub vector."""
    if shape_constraints in [
        ("increasing", "increasing"),
        ("decreasing", "decreasing"),
    ]:
        return np.append(b_ub, np.zeros(num_bfuncs - 2))

    msg = "Invalid shape constraints."
    raise ValueError(msg)


def _second_step_linear_program(
    target: Estimand,
    identified_estimands: list[Estimand],
    basis_funcs: list[dict],
    data: dict[str, np.ndarray],
    minimal_deviations: float,
    tolerance: float,
    beta_hat: np.ndarray,
    instrument: Instrument,
    shape_constraints: tuple[str, str] | None,
    method: str,
) -> dict:
    """Second step linear program to estimate upper and lower bounds."""
    lp_second_inputs = {}
    lp_second_inputs["c"] = _compute_choice_weights_second_step(
        target,
        basis_funcs,
        identified_estimands,
        instrument=instrument,
    )
    lp_second_inputs["b_ub"] = _compute_second_step_upper_bounds(
        minimal_deviations=minimal_deviations,
        tolerance=tolerance,
        beta_hat=beta_hat,
    )
    lp_second_inputs["a_ub"] = _build_second_step_ub_matrix(
        basis_funcs=basis_funcs,
        identified_estimands=identified_estimands,
        data=data,
        instrument=instrument,
    )
    if shape_constraints is not None:
        lp_second_inputs["a_ub"] = _append_shape_constraints_a_ub(
            shape_constraints,
            lp_second_inputs["a_ub"],
            len(basis_funcs) * 2,
            num_idestimands=len(identified_estimands),
        )
        lp_second_inputs["b_ub"] = _append_shape_constraints_b_ub(
            shape_constraints,
            lp_second_inputs["b_ub"],
            len(basis_funcs) * 2,
        )

    lp_second_inputs["bounds"] = _compute_second_step_bounds(
        basis_funcs,
        identified_estimands,  # type: ignore
    )

    if method == "copt":
        result_upper = _solve_lp_estimation_copt(lp_second_inputs, "max")
        result_lower = _solve_lp_estimation_copt(lp_second_inputs, "min")

        return {
            "upper_bound": -1 * result_upper,
            "lower_bound": result_lower,
        }

    result_upper = _solve_second_step_lp_estimation(lp_second_inputs, "max", method)
    result_lower = _solve_second_step_lp_estimation(lp_second_inputs, "min", method)

    return {
        "upper_bound": -1 * result_upper.fun,  # type: ignore
        "lower_bound": result_lower.fun,  # type: ignore
        "inputs": lp_second_inputs,
        "scipy_return_upper": result_upper,
        "scipy_return_lower": result_lower,
    }


def _compute_choice_weights_second_step(
    target: Estimand,
    basis_funcs: list[dict],
    identified_estimands: list,
    instrument: Instrument,
) -> np.ndarray:
    """Compute choice weight vector c for second step linear program."""
    upper_part = _compute_choice_weights(
        target,
        basis_funcs=basis_funcs,
        instrument=instrument,
    )

    lower_part = np.zeros(len(identified_estimands))
    return np.append(upper_part, lower_part)


def _build_second_step_ub_matrix(
    basis_funcs: list[dict],
    identified_estimands: list[Estimand],
    data: dict[str, np.ndarray],
    instrument: Instrument,
) -> np.ndarray:
    """Build a_ub matrix for second step linear program."""
    num_idestimands = len(identified_estimands)
    num_bfuncs = len(basis_funcs) * 2

    first_row = np.append(np.zeros(num_bfuncs), np.ones(num_idestimands))

    weight_matrix = np.empty(shape=(num_idestimands, num_bfuncs))

    moments = _estimate_moments_for_weights(data["z"], data["d"])

    for i, estimand in enumerate(identified_estimands):
        weights = _estimate_weights_estimand(
            estimand=estimand,
            basis_funcs=basis_funcs,
            data=data,
            moments=moments,
            instrument=instrument,
        )

        weight_matrix[i, :] = weights

    top = np.hstack((weight_matrix, -np.eye(num_idestimands)))
    bottom = np.hstack((-weight_matrix, -np.eye(num_idestimands)))

    out = np.vstack((top, bottom))
    return np.vstack((first_row, out))


def _compute_second_step_bounds(
    basis_funcs: list[dict],
    identified_estimands: list[Estimand],
) -> list:
    """Compute bounds for second step linear program."""
    num_bfuncs = len(basis_funcs) * 2
    num_idestimands = len(identified_estimands)

    return [(0, 1) for _ in range(num_bfuncs)] + [
        (None, None) for _ in range(num_idestimands)
    ]


def _compute_first_step_upper_bounds(beta_hat: np.ndarray) -> np.ndarray:
    """Compute b_ub vector with upper bounds of ineq constraint in first step LP."""
    return np.append(beta_hat, -beta_hat)


def _compute_second_step_upper_bounds(
    minimal_deviations: float,
    tolerance: float,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """Compute b_ub vector with upper bounds of ineq constraint in second step LP."""
    return np.append(
        np.array(minimal_deviations + tolerance),
        np.append(beta_hat, -beta_hat),
    )


def _estimate_moments_for_weights(z_data: np.ndarray, d_data: np.ndarray) -> dict:
    """Estimate relevant moments for computing weights on LP choice variables."""
    moments = {}

    moments["expectation_d"] = np.mean(d_data)
    moments["variance_d"] = np.var(d_data)
    moments["expectation_z"] = np.mean(z_data)
    moments["covariance_dz"] = np.cov(d_data, z_data)[0, 1]

    return moments


def _estimate_instrument_characteristics(
    z_data: np.ndarray,
    d_data: np.ndarray,
) -> Instrument:
    """Estimate instrument characteristics and return in instrument class."""
    support = np.unique(z_data)

    return Instrument(
        support=support,
        pscores=_estimate_prop_z(z_data, d_data, support),
        pmf=_estimate_instrument_pdf(z_data, support),
    )


def _estimate_gamma_for_basis_funcs(
    d_value: int,
    estimand: Estimand,
    basis_func: dict,
    data: dict,
    moments: dict,
    instrument: Instrument,
) -> float:
    """Estimate gamma linear map for basis function (cf.

    S33 in Appendix).

    """
    length = basis_func["u_hi"] - basis_func["u_lo"]

    if estimand.esttype == "ols_slope":
        coef = (d_value - moments["expectation_d"]) / moments["variance_d"]
    if estimand.esttype == "iv_slope":
        coef = (data["z"] - moments["expectation_z"]) / moments["covariance_dz"]
    if estimand.esttype == "cross":
        d_cross = estimand.dz_cross[0]  # type: ignore
        z_cross = estimand.dz_cross[1]  # type: ignore

        if d_value != d_cross:
            return 0

        if d_value == 0:
            # Could use instrment values here!
            if basis_func["u_lo"] < instrument.pscores[0]:
                return 0
            mask1 = basis_func["u_lo"] >= data["pscores"]
            mask2 = data["z"] == z_cross
            cross_indicators = np.logical_and(mask1, mask2)

        if d_value == 1:
            if basis_func["u_hi"] > instrument.pscores[-1]:
                return 0
            mask1 = basis_func["u_hi"] <= data["pscores"]
            mask2 = data["z"] == z_cross
            cross_indicators = np.logical_and(mask1, mask2)

        return length * np.count_nonzero(cross_indicators) / len(cross_indicators)

    if d_value == 0:
        # Create array of 1 if basis_funcs["u_lo"] > data["pscores"] else 0
        indicators = basis_func["u_lo"] >= data["pscores"]
    else:
        indicators = basis_func["u_hi"] <= data["pscores"]

    if estimand.esttype == "ols_slope":
        share = np.count_nonzero(indicators) / len(indicators)
        return length * coef * share

    return length * np.mean(coef * indicators)


def _solve_lp_estimation_copt(lp_second_inputs: dict, min_or_max: str) -> float:
    """Wrapper for solving LP using copt algorithm."""
    c = lp_second_inputs["c"] if min_or_max == "min" else -lp_second_inputs["c"]
    a_ub = lp_second_inputs["a_ub"]
    b_ub = lp_second_inputs["b_ub"]
    bounds = lp_second_inputs["bounds"]

    lb = np.array([x[0] if x[0] is not None else (-1) * COPT.INFINITY for x in bounds])
    ub = np.array([x[1] if x[1] is not None else COPT.INFINITY for x in bounds])

    env = cp.Envr()
    model = env.createModel("estimation")
    x = model.addMVar(len(c), nameprefix="x", lb=lb, ub=ub)
    model.setObjective(c @ x, COPT.MINIMIZE)
    model.addMConstr(a_ub, x, "L", b_ub, nameprefix="c")

    with suppress_print():
        model.solveLP()

    if model.status != COPT.OPTIMAL:
        msg = "LP not solved to optimality by copt."
        raise ValueError(msg)
    return model.objval


def _check_estimation_arguments(
    target: Estimand,
    identified_estimands: list[Estimand],
    basis_func_type: str,
    y_data: np.ndarray,
    z_data: np.ndarray,
    d_data: np.ndarray,
    tolerance: float | None,
    u_partition: np.ndarray | None,
    shape_constraints: tuple[str, str] | None,
    method: str,
):
    """Check args to estimation func, returns report if there are errors."""
    error_report = ""

    error_report += _error_report_estimand(target)
    for ident in identified_estimands:
        error_report += _error_report_estimand(ident)
    error_report += _error_report_invalid_basis_func_type(basis_func_type)

    error_report += _error_report_estimation_data(y_data, z_data, d_data)

    error_report += _error_report_tolerance(tolerance)
    error_report += _error_report_u_partition(u_partition)
    error_report += _error_report_method(method)
    error_report += _error_report_shape_constraints(shape_constraints)

    if error_report != "":
        raise ValueError(error_report)

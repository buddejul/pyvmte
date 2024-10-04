"""Function for estimation."""

from functools import partial
from itertools import pairwise

import coptpy as cp  # type: ignore
import numpy as np
from coptpy import COPT
from scipy.optimize import (  # type: ignore
    OptimizeResult,
    linprog,  # type: ignore
)

from pyvmte.classes import Estimand, Instrument, PyvmteResult
from pyvmte.identification.identification import _compute_choice_weights
from pyvmte.utilities import (
    _error_report_estimand,
    _error_report_estimation_data,
    _error_report_invalid_basis_func_type,
    _error_report_method,
    _error_report_missing_basis_func_options,
    _error_report_monotone_response,
    _error_report_mte_monotone,
    _error_report_shape_constraints,
    _error_report_tolerance,
    _error_report_u_partition,
    estimate_late,
    generate_bernstein_basis_funcs,
    s_cross,
    s_iv_slope,
    s_late,
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
    mte_monotone: str | None = None,
    monotone_response: str | None = None,
    method: str = "highs",
    basis_func_options: dict | None = None,
) -> PyvmteResult:
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
        mte_monotone: Shape constraint for the MTE, either "increasing" or "decreasing".
            Defaults to None. Corresponds to monotone treatment selection.
        monotone_response: Whether the treatment response is monotone.
            Defaults to None, allowed are "positive" and "negative".
        method (str, optional): Method for scipy linprog solver. Default highs.
        basis_func_options: Options for basis functions. Default None.

    Returns:
        PyvmteResult: Object containing the results of the estimation procedure.

    """
    if isinstance(identified_estimands, Estimand):
        identified_estimands = [identified_estimands]

    _check_estimation_arguments(
        target=target,
        identified_estimands=identified_estimands,
        basis_func_type=basis_func_type,
        y_data=y_data,
        z_data=z_data,
        d_data=d_data,
        tolerance=tolerance,
        u_partition=u_partition,
        shape_constraints=shape_constraints,
        method=method,
        basis_func_options=basis_func_options,
        mte_monotone=mte_monotone,
        monotone_response=monotone_response,
    )

    # ==================================================================================
    # Preliminary Computations (Partitions, Basis Functions, Instrument, Estimates)
    # ==================================================================================
    if tolerance is None:
        tolerance = 1 / len(y_data)

    instrument = _estimate_instrument_characteristics(z_data, d_data)

    # The target is typically taken at the estimated propensity scores. If target
    # is a LATE with missing u_lo and u_hi attributed, replace them by the propensity
    # scores.
    # TODO(@buddejul): Should be more flexible here with specifying the propensity
    # scores values/corresponding instrument values. Standard is to assume binary.
    # But could have instruments with larger support and this would fail.
    if target.esttype == "late":
        if target.u_lo is None:
            target.u_lo = (
                np.min(instrument.pscores) + target.u_lo_extra
                if target.u_lo_extra is not None
                else np.min(instrument.pscores)
            )
        if target.u_hi is None:
            target.u_hi = (
                np.max(instrument.pscores) + target.u_hi_extra
                if target.u_hi_extra is not None
                else np.max(instrument.pscores)
            )

    u_partition = _compute_u_partition(
        target=target,
        pscore_z=instrument.pscores,
    )

    if basis_func_type == "bernstein" and basis_func_options is not None:
        basis_funcs = _generate_basis_funcs(
            basis_func_type,
            u_partition,
            k_degree=basis_func_options["k_degree"],
        )
    else:
        basis_funcs = _generate_basis_funcs(basis_func_type, u_partition)

    beta_hat = _estimate_identified_estimands(
        identified_estimands=identified_estimands,
        y_data=y_data,
        z_data=z_data,
        d_data=d_data,
    )

    # Now do the same for identified estimands.
    for id_estimand in identified_estimands:
        if id_estimand.esttype == "late":
            if id_estimand.u_lo is None:
                id_estimand.u_lo = (
                    np.min(instrument.pscores) + id_estimand.u_lo_extra
                    if id_estimand.u_lo_extra is not None
                    else np.min(instrument.pscores)
                )
            if id_estimand.u_hi is None:
                id_estimand.u_hi = (
                    np.max(instrument.pscores) + id_estimand.u_hi_extra
                    if id_estimand.u_hi_extra is not None
                    else np.max(instrument.pscores)
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
        mte_monotone=mte_monotone,
        monotone_response=monotone_response,
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
        mte_monotone=mte_monotone,
        monotone_response=monotone_response,
        method=method,
    )

    # ==================================================================================
    # Return Results
    # ==================================================================================
    _success_lower = results_second_step["scipy_return_lower"].success
    _success_upper = results_second_step["scipy_return_upper"].success

    return PyvmteResult(
        procedure="estimation",
        success=(_success_lower, _success_upper),
        lower_bound=results_second_step["lower_bound"] if _success_lower else None,
        upper_bound=results_second_step["upper_bound"] if _success_upper else None,
        target=target,
        identified_estimands=identified_estimands,
        basis_funcs=basis_funcs,
        method=method,
        lp_api="coptpy" if method == "copt" else "scipy",
        lower_optres=(
            results_second_step["scipy_return_lower"] if method != "copt" else None
        ),
        upper_optres=(
            results_second_step["scipy_return_upper"] if method != "copt" else None
        ),
        lp_inputs=results_second_step["inputs"],
        est_u_partition=u_partition,
        est_beta_hat=beta_hat,
        first_minimal_deviations=results_first_step["minimal_deviations"],
        first_lp_inputs=results_first_step["inputs"],
        first_optres=results_first_step["scipy_return"] if method != "copt" else None,
        restrictions={
            "shape_constraints": shape_constraints,
            "mte_monotone": mte_monotone,
            "monotone_response": monotone_response,
        },
    )


def _first_step_linear_program(
    identified_estimands: list[Estimand],
    basis_funcs: list[dict],
    data: dict[str, np.ndarray],
    beta_hat: np.ndarray,
    instrument: Instrument,
    shape_constraints: tuple[str, str] | None,
    mte_monotone: str | None,
    monotone_response: str | None,
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

    _add_constraints = [shape_constraints, mte_monotone, monotone_response]

    if any(_add_constraints):
        _add_a_ub = _additional_constraints_a_ub(
            shape_constraints=shape_constraints,
            mte_monotone=mte_monotone,
            monotone_response=monotone_response,
            num_bfuncs=len(basis_funcs),
            num_idestimands=len(identified_estimands),
        )

        _add_b_ub = _additional_constraints_b_ub(
            shape_constraints=shape_constraints,
            mte_monotone=mte_monotone,
            monotone_response=monotone_response,
            num_bfuncs=len(basis_funcs),
        )

        lp_first_inputs["a_ub"] = np.vstack(
            (lp_first_inputs["a_ub"], _add_a_ub),
        )

        lp_first_inputs["b_ub"] = np.append(
            lp_first_inputs["b_ub"],
            _add_b_ub,
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


def _generate_basis_funcs(
    basis_func_type: str,
    u_partition: np.ndarray,
    k_degree: int | None = None,
) -> list:
    """Generate list of dictionaries describing basis functions."""
    if basis_func_type == "constant":
        bfuncs_list = []
        for u_lo, u_hi in pairwise(u_partition):
            bfuncs_list.append({"type": "constant", "u_lo": u_lo, "u_hi": u_hi})

        return bfuncs_list

    if basis_func_type == "bernstein" and k_degree is not None:
        return generate_bernstein_basis_funcs(k=k_degree)

    if basis_func_type == "bernstein" and not isinstance(k_degree, int):
        msg = "Type 'bernstein' provided but k_degree is not an integer."
    else:
        msg = "Could not generate basis functions."

    raise ValueError(msg)


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
        return estimate_late(y=y_data, d=d_data, z=z_data)

    if estimand.esttype == "cross":
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


def _additional_constraints_a_ub(
    shape_constraints: tuple[str, str] | None,
    mte_monotone: str | None,
    monotone_response: str | None,
    num_bfuncs: int,
    num_idestimands: int | None = None,
):
    to_concat = []

    if shape_constraints is not None:
        to_concat.append(
            _shape_constraints_a_ub(
                shape_constraints,
                num_bfuncs,
                num_idestimands,
            ),
        )

    if mte_monotone is not None:
        to_concat.append(
            _mte_monotone_a_ub(
                mte_monotone,
                num_bfuncs,
                num_idestimands,
            ),
        )
    if monotone_response is not None:
        to_concat.append(
            _monotone_response_a_ub(
                monotone_response,
                num_bfuncs,
                num_idestimands,
            ),
        )

    return np.vstack(to_concat)


def _additional_constraints_b_ub(
    shape_constraints: tuple[str, str] | None,
    mte_monotone: str | None,
    monotone_response: str | None,
    num_bfuncs: int,
):
    _num_bounds = 0

    if shape_constraints is not None:
        _num_bounds += 2 * (num_bfuncs - 1)

    if mte_monotone is not None:
        _num_bounds += num_bfuncs - 1

    if monotone_response is not None:
        _num_bounds += num_bfuncs

    return np.zeros(_num_bounds)


def _shape_constraints_a_ub(
    shape_constraints: tuple[str, str],
    num_bfuncs: int,
    num_idestimands: int | None = None,
) -> np.ndarray:
    """Append shape constraints to a_ub matrix.

    num_idestimands needs to be supplied for the first step linear program.

    """
    a = np.eye(num_bfuncs * 2 - 1, num_bfuncs * 2)
    b = np.eye(num_bfuncs * 2 - 1, num_bfuncs * 2, 1)

    out = a - b

    # Now we need to delete the (num_bfuncs)th row, so we don't put cross-
    # restrictions on the MTR d = 0 and d = 1 functions.
    to_delete = int(num_bfuncs - 1)
    out = np.delete(out, to_delete, axis=0)

    # Add a matrix of zeros to the right with the same rows and num_idestimands columns
    # to match the first step linear program structure.
    if num_idestimands is not None:
        out = np.hstack(
            (out, np.zeros((out.shape[0], num_idestimands))),
        )

    if shape_constraints == ("increasing", "increasing"):
        return out

    if shape_constraints == ("decreasing", "decreasing"):
        return -out

    msg = "Invalid shape constraints."
    raise ValueError(msg)


def _mte_monotone_a_ub(
    mte_monotone: str,
    n_basis_funcs: int,
    num_idestimands: int | None = None,
) -> np.ndarray:
    a = np.eye(n_basis_funcs) - np.eye(n_basis_funcs, k=1)
    a = a[:-1, :]
    a = np.hstack((a, -a))

    # Add a matrix of zeros to the right with the same rows and num_idestimands columns
    # to match the first step linear program structure.
    if num_idestimands is not None:
        a = np.hstack(
            (a, np.zeros((a.shape[0], num_idestimands))),
        )

    if mte_monotone == "decreasing":
        return a
    if mte_monotone == "increasing":
        return -a

    msg = f"Invalid MTE monotonicity constraint: {mte_monotone}."
    raise ValueError(msg)


def _monotone_response_a_ub(
    monotone_response: str,
    n_basis_funcs: int,
    num_idestimands: int | None = None,
) -> np.ndarray:
    a = np.hstack((np.eye(n_basis_funcs), -np.eye(n_basis_funcs)))

    # Add a matrix of zeros to the right with the same rows and num_idestimands columns
    # to match the first step linear program structure.
    if num_idestimands is not None:
        a = np.hstack(
            (a, np.zeros((a.shape[0], num_idestimands))),
        )

    if monotone_response == "positive":
        return a
    if monotone_response == "negative":
        return -a

    msg = f"Invalid monotone response constraint: {monotone_response}."
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
    mte_monotone: str | None,
    monotone_response: str | None,
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

    _add_constr = [shape_constraints, mte_monotone, monotone_response]

    # Add additional constraints to inputs if any are specified.
    if any(_add_constr):
        _add_a_ub = _additional_constraints_a_ub(
            shape_constraints=shape_constraints,
            mte_monotone=mte_monotone,
            monotone_response=monotone_response,
            num_bfuncs=len(basis_funcs),
            num_idestimands=len(identified_estimands),
        )

        _add_b_ub = _additional_constraints_b_ub(
            shape_constraints=shape_constraints,
            mte_monotone=mte_monotone,
            monotone_response=monotone_response,
            num_bfuncs=len(basis_funcs),
        )

        lp_second_inputs["a_ub"] = np.vstack(
            (lp_second_inputs["a_ub"], _add_a_ub),
        )

        lp_second_inputs["b_ub"] = np.append(
            lp_second_inputs["b_ub"],
            _add_b_ub,
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
            "inputs": lp_second_inputs,
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

    # This will allow to use the identified estimand in the constraints.
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
    bfunc_type = basis_func["type"]

    if bfunc_type == "constant":
        return _estimate_gamma_constant_spline(
            d_value,
            estimand,
            basis_func,
            data,
            moments,
            instrument,
        )

    if bfunc_type == "bernstein":
        return _estimate_gamma_bernstein(
            d_value,
            estimand,
            basis_func,
            data,
            moments,
            instrument,
        )

    msg = "Invalid basis function type."
    raise ValueError(msg)


def _estimate_gamma_bernstein(
    d_value: int,
    estimand: Estimand,
    basis_func: dict,
    data: dict,
    moments: dict,
    instrument: Instrument,
) -> float:
    _bfunc = basis_func["func"]

    # All the following weights functions only depend on z, but not u.
    # Hence we can pull them out of the integral.

    # Step 1: Get weight function s(D, Z) depending on the estimand type
    # For late target, we can estimate the weight directly
    if estimand.esttype == "late":
        _s_late = partial(s_late, d=d_value, u_lo=estimand.u_lo, u_hi=estimand.u_hi)

        _mid = (
            estimand.u_lo + (estimand.u_hi - estimand.u_lo) / 2  # type: ignore[operator]
        )

        return _s_late(u=_mid) * _bfunc.integrate(estimand.u_lo, estimand.u_hi)

    if estimand.esttype == "ols_slope":
        _s_ols_slope = partial(
            s_ols_slope,
            d=d_value,
            ed=moments["expectation_d"],
            var_d=moments["variance_d"],
        )

        def _sdz(z):
            return _s_ols_slope(z=z)

    if estimand.esttype == "iv_slope":
        _s_iv_slope = partial(
            s_iv_slope,
            ez=moments["expectation_z"],
            cov_dz=moments["covariance_dz"],
        )

        def _sdz(z):
            return _s_iv_slope(z=z)

    if estimand.esttype == "cross":

        def _sdz(z):
            return s_cross(d=d_value, z=z, dz_cross=estimand.dz_cross)

    # Step 3: Compute the weight separately for each z in support of instrument
    weight = 0

    for z in instrument.support:
        _pscore = instrument.pscores[np.where(instrument.support == z)][0]

        # For the case d == 0, lower bound of integration becomes pscore
        # For the case d == 1, upper bound of integration becomes pscore
        if d_value == 0:
            _integral = _bfunc.integrate(_pscore, 1)
        else:
            _integral = _bfunc.integrate(0, _pscore)

        _pos = np.where(instrument.support == z)[0][0]
        weight += _sdz(z) * _integral * instrument.pmf[_pos]

    return weight


def _estimate_gamma_constant_spline(  # noqa: PLR0911
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
    # Instead of integrating, for constant splines we can simply scale by the length.
    length = basis_func["u_hi"] - basis_func["u_lo"]

    if estimand.esttype == "late":
        _s_late = partial(s_late, d=d_value, u_lo=estimand.u_lo, u_hi=estimand.u_hi)

        _mid = (
            estimand.u_lo + (estimand.u_hi - estimand.u_lo) / 2  # type: ignore[operator]
        )

        # Note: Make sure multiplying by length does the right thing. It should be a
        # consistent estimator either way, but we are not avergaing over individuals.
        # Since target estimands *and* bfuncs are defined by their endpoints this could
        # as well be calculated analytically.
        return (
            _s_late(u=_mid)
            * (basis_func["u_lo"] <= _mid)
            * (_mid <= basis_func["u_hi"])
        ) * length

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
    mte_monotone: str | None,
    monotone_response: str | None,
    method: str,
    basis_func_options: dict | None,
):
    """Check args to estimation func, returns report if there are errors."""
    error_report = ""

    error_report += _error_report_estimand(target, mode="estimation")
    for ident in identified_estimands:
        error_report += _error_report_estimand(ident, mode="estimation")
    error_report += _error_report_invalid_basis_func_type(basis_func_type)
    error_report += _error_report_missing_basis_func_options(
        basis_func_type,
        basis_func_options,
    )

    error_report += _error_report_estimation_data(y_data, z_data, d_data)

    error_report += _error_report_tolerance(tolerance)
    error_report += _error_report_u_partition(u_partition)
    error_report += _error_report_method(method)
    error_report += _error_report_shape_constraints(shape_constraints)
    error_report += _error_report_mte_monotone(mte_monotone)
    error_report += _error_report_monotone_response(monotone_response)

    if error_report != "":
        raise ValueError(error_report)

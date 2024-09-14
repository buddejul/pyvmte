"""Function for identification."""

from collections.abc import Callable
from functools import partial

import coptpy as cp  # type: ignore
import numpy as np
from coptpy import COPT
from scipy import integrate  # type: ignore
from scipy.optimize import (  # type: ignore
    linprog,  # type: ignore
)

from pyvmte.classes import Estimand, Instrument
from pyvmte.utilities import (
    _error_report_basis_funcs,
    _error_report_estimand,
    _error_report_instrument,
    _error_report_method,
    _error_report_mtr_function,
    _error_report_shape_constraints,
    _error_report_u_partition,
    compute_moments,
    s_cross,
    s_iv_slope,
    s_late,
    s_ols_slope,
)


def identification(
    target: Estimand,
    identified_estimands: list[Estimand],
    basis_funcs: list[dict],
    m0_dgp: Callable,
    m1_dgp: Callable,
    instrument: Instrument,
    u_partition: np.ndarray,
    shape_constraints: tuple[str, str] | None = None,
    method: str = "highs",
    debug: bool = False,  # noqa: FBT001, FBT002
):
    """Compute bounds on target estimand given identified estimands and DGP.

    Args:
        target (Estimand): All information about the target estimand.
        identified_estimands (Estimand or list of Estimands): All
            information about the identified estimand(s). List if multiple
            identified estimands.
        basis_funcs (dict or list of dicts): Dictionaries describing the basis
            functions.
        m0_dgp (function): The MTR function for d=0 of the DGP.
        m1_dgp (function): The MTR function for d=1 of the DGP.
        instrument (Instrument): All information about the instrument.
        u_partition (list or np.array): Partition of u for basis_funcs.
            Defaults to None.
        shape_constraints: Shape constraints for the MTR functions.
        method (str, optional): Method for solving the linear program.
            Implemented are: all methods supported by scipy.linprog as well as copt.
            Defaults to "highs" using scipy.linprog.
        debug: Whether to return the full output of the linear program solver.

    Returns:
        dict: A dictionary containing the upper and lower bound of the target estimand.

    """
    # ==================================================================================
    # Perform some additional checks on arguments
    # ==================================================================================
    if isinstance(identified_estimands, Estimand):
        identified_estimands = [identified_estimands]

    if isinstance(basis_funcs, dict):
        basis_funcs = [basis_funcs]

    _check_identification_arguments(
        target,
        identified_estimands,
        basis_funcs,
        m0_dgp,
        m1_dgp,
        instrument,
        u_partition,
        shape_constraints,
        method,
    )

    # ==================================================================================
    # Generate linear program inputs
    # ==================================================================================

    lp_inputs = {}

    lp_inputs["c"] = _compute_choice_weights(target, basis_funcs, instrument=instrument)
    lp_inputs["b_eq"] = _compute_identified_estimands(
        identified_estimands,
        m0_dgp,
        m1_dgp,
        u_partition,
        instrument,
    )
    lp_inputs["a_eq"] = _compute_equality_constraint_matrix(
        identified_estimands,
        basis_funcs,
        instrument=instrument,
    )

    if shape_constraints is not None:
        lp_inputs["a_ub"] = _compute_inequality_constraint_matrix(
            shape_constraints=shape_constraints,
            n_basis_funcs=len(basis_funcs),
        )
        lp_inputs["b_ub"] = _compute_inequality_upper_bounds(
            shape_constraints=shape_constraints,
            n_basis_funcs=len(basis_funcs),
        )

    # ==================================================================================
    # Solve linear program
    # ==================================================================================

    if debug is True:
        return {
            "upper": _solve_lp(lp_inputs, "max", method=method, debug=debug),
            "lower": _solve_lp(lp_inputs, "min", method=method, debug=debug),
        }

    upper_bound = (-1) * _solve_lp(lp_inputs, "max", method=method)
    lower_bound = _solve_lp(lp_inputs, "min", method=method)

    return {"upper_bound": upper_bound, "lower_bound": lower_bound}


def _compute_identified_estimands(
    identified_estimands: list[Estimand],
    m0_dgp: Callable,
    m1_dgp: Callable,
    u_part: np.ndarray,
    instrument: Instrument,
) -> np.ndarray:
    """Wrapper for computing identified estimands based on provided dgp."""
    out = []
    for estimand in identified_estimands:
        result = _compute_estimand(estimand, m0_dgp, m1_dgp, u_part, instrument)
        out.append(result)

    return np.array(out)


def _compute_estimand(
    estimand: Estimand,
    m0: Callable,
    m1: Callable,
    u_part: np.ndarray | None = None,
    instrument: Instrument | None = None,
) -> float:
    """Compute single identified estimand."""
    a = _gamma_star(
        md=m0,
        d_value=0,
        estimand=estimand,
        instrument=instrument,
        u_part=u_part,
    )
    b = _gamma_star(
        md=m1,
        d_value=1,
        estimand=estimand,
        instrument=instrument,
        u_part=u_part,
    )

    return a + b


def _compute_choice_weights(
    target: Estimand,
    basis_funcs: list[dict],
    instrument: Instrument,
    moments: dict | None = None,
) -> np.ndarray:
    """Compute weights on the choice variables."""
    # TODO: Refactor this; current approach only allows for one type of func anyways.
    bfunc_type = basis_funcs[0]["type"]

    if bfunc_type == "constant":
        if moments is None and instrument is not None:
            moments = _compute_moments_for_weights(target, instrument)
        c = []
        for d in [0, 1]:
            for bfunc in basis_funcs:
                weight = _compute_constant_spline_weights(
                    estimand=target,
                    basis_function=bfunc,
                    d=d,
                    instrument=instrument,
                    moments=moments,
                )
                c.append(weight)

    if bfunc_type == "bernstein":
        c = []
        for d in [0, 1]:
            for bfunc in basis_funcs:
                weight = _compute_bernstein_weights(
                    estimand=target,
                    basis_function=bfunc,
                    d=d,
                    instrument=instrument,
                    moments=moments,
                )
                c.append(weight)

    return np.array(c)


def _compute_equality_constraint_matrix(
    identified_estimands: list[Estimand],
    basis_funcs: list,
    instrument: Instrument,
) -> np.ndarray:
    """Compute weight matrix for equality constraints."""
    c_matrix = []
    for target in identified_estimands:
        c_row = _compute_choice_weights(
            target=target,
            basis_funcs=basis_funcs,
            instrument=instrument,
        )

        c_matrix.append(c_row)

    return np.array(c_matrix)


def _compute_inequality_constraint_matrix(
    shape_constraints: tuple[str, str],
    n_basis_funcs: int,
) -> np.ndarray:
    """Returns the inequality constraint matrix incorporating shape constraints."""
    a = np.eye(2 * n_basis_funcs - 1, 2 * n_basis_funcs)
    b = np.eye(2 * n_basis_funcs - 1, 2 * n_basis_funcs, 1)

    out = a - b

    # Now we need to delete the (n_basis_funcs)th row, so we don't put cross-
    # restrictions on the MTR d = 0 and d = 1 functions.
    out = np.delete(out, n_basis_funcs - 1, axis=0)

    if shape_constraints == ("increasing", "increasing"):
        return out

    if shape_constraints == ("decreasing", "decreasing"):
        return -out

    msg = "Invalid shape constraints."
    raise ValueError(msg)


def _compute_inequality_upper_bounds(
    shape_constraints: tuple[str, str],
    n_basis_funcs: int,
) -> np.ndarray:
    return np.zeros(2 * n_basis_funcs - 2)


def _solve_lp(
    lp_inputs: dict,
    max_or_min: str,
    method: str,
    debug: bool = False,  # noqa: FBT001, FBT002
) -> float:
    """Wrapper for solving the linear program."""
    c = np.array(lp_inputs["c"]) if max_or_min == "min" else -np.array(lp_inputs["c"])

    b_eq = lp_inputs["b_eq"]
    a_eq = lp_inputs["a_eq"]

    a_ub = lp_inputs.get("a_ub", None)
    b_ub = lp_inputs.get("b_ub", None)

    if method == "copt":
        return _solve_lp_copt(c, a_eq, b_eq, a_ub, b_ub)

    if debug is True:
        return linprog(c=c, A_eq=a_eq, b_eq=b_eq, A_ub=a_ub, b_ub=b_ub, bounds=(0, 1))

    return linprog(c=c, A_eq=a_eq, b_eq=b_eq, A_ub=a_ub, b_ub=b_ub, bounds=(0, 1)).fun


def _compute_moments_for_weights(target: Estimand, instrument: Instrument) -> dict:
    """Compute moments of d and z for LP weights."""
    moments = {}

    if target.esttype == "ols_slope":
        moments["expectation_d"] = _compute_binary_expectation_using_lie(
            d_pdf_given_z=instrument.pscores,
            z_pdf=instrument.pmf,
        )
        moments["variance_d"] = moments["expectation_d"] * (
            1 - moments["expectation_d"]
        )

    if target.esttype == "iv_slope":
        moments["expectation_z"] = _compute_expectation(
            support=instrument.support,
            pdf=instrument.pmf,
        )
        moments["covariance_dz"] = _compute_covariance_dz(
            support_z=instrument.support,
            pscore_z=instrument.pscores,
            pdf_z=instrument.pmf,
        )

    return moments


def _compute_binary_expectation_using_lie(
    d_pdf_given_z: np.ndarray,
    z_pdf: np.ndarray,
) -> float:
    """Compute expectation of d using the law of iterated expectations."""
    return d_pdf_given_z @ z_pdf


def _compute_expectation(support: np.ndarray, pdf: np.ndarray) -> float:
    """Compute expectation of a discrete random variable."""
    return support @ pdf


def _compute_covariance_dz(
    support_z: np.ndarray,
    pscore_z: np.ndarray,
    pdf_z: np.ndarray,
) -> float:
    """Compute covariance between binary treatment and discrete instrument."""
    ez = support_z @ pdf_z
    ed = pscore_z @ pdf_z
    edz = np.sum(support_z * pscore_z * pdf_z)
    return edz - ed * ez


def _solve_lp_copt(
    c: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
) -> float:
    """Wrapper for solving LP using copt algorithm."""
    env = cp.Envr()
    model = env.createModel("identification")
    x = model.addMVar(len(c), nameprefix="x", lb=0, ub=1)
    model.setObjective(c @ x, COPT.MINIMIZE)
    model.addMConstr(a_eq, x, "E", b_eq, nameprefix="c")
    if a_ub is not None and b_ub is not None:
        model.addMConstr(a_ub, x, "L", b_ub)

    model.solveLP()

    if model.status != COPT.OPTIMAL:
        msg = "LP not solved to optimality by copt."
        raise ValueError(msg)
    return model.objval


def _gamma_star(
    md: Callable,
    d_value: int,
    estimand: Estimand,
    instrument: Instrument | None = None,
    dz_cross: tuple | None = None,
    u_part: np.ndarray | None = None,
):
    """Compute gamma* for a given MTR function and estimand.

    Args:
        md (function): MTR function
        d_value (np.int): value of the treatment
        estimand (str): the estimand to compute
        u_lo (float): lower bound of late target
        u_hi (float): upper bound of late target
        support_z (np.array): support of the instrument
        pscore_z (np.array): propensity given the instrument
        pdf_z (np.array): probability mass function of the instrument
        dz_cross (list): list of tuples of the form (d_spec, z_spec) for cross-moment
        analyt_int (Boolean): Whether to integrate manually or use analytic results.
        instrument (Instrument): Instrument object containing all information about the
            instrument.
        u_part (np.ndarray): partition of u

    """
    u_lo = estimand.u_lo
    u_hi = estimand.u_hi
    dz_cross = estimand.dz_cross

    if instrument is not None:
        pdf_z = instrument.pmf
        pscore_z = instrument.pscores
        support_z = instrument.support

    if estimand.esttype == "late":
        return integrate.quad(lambda u: md(u) * s_late(d_value, u, u_lo, u_hi), 0, 1)[0]

    ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)
    var_d = ed * (1 - ed)

    if estimand.esttype == "iv_slope":

        def inner_func(u, z):
            return md(u) * s_iv_slope(z, ez, cov_dz)

    if estimand.esttype == "ols_slope":

        def inner_func(u, z):
            return md(u) * s_ols_slope(d_value, ed, var_d)

    if estimand.esttype == "cross":

        def inner_func(u, z):
            return md(u) * s_cross(d_value, z, dz_cross)

    if d_value == 0:

        def func(u, z):
            return inner_func(u, z) if pscore_z[np.where(support_z == z)] < u else 0

    if d_value == 1:

        def func(u, z):
            return inner_func(u, z) if pscore_z[np.where(support_z == z)] > u else 0

    return np.sum(
        [
            integrate.quad(func, 0, 1, args=(supp,))[0] * pdf  # type: ignore
            for pdf, supp in zip(pdf_z, support_z, strict=True)  # type: ignore
        ],
    )


def _compute_constant_spline_weights(
    estimand: Estimand,
    d: int,
    basis_function: dict,
    instrument: Instrument,
    moments: dict | None = None,
):
    """Compute weights for constant spline basis.

    We use that for a constant spline basis with the right partition the weights are
    constant on each interval of the partition.

    We condition on z and compute the weights for each interval of the partition using
    the law of iterated expectations.

    """
    u = (basis_function["u_lo"] + basis_function["u_hi"]) / 2

    if estimand.esttype == "ols_slope":
        out = _compute_ols_weight_for_identification(
            u=u,
            d=d,
            instrument=instrument,
            moments=moments,
        )

    if estimand.esttype == "iv_slope":
        out = _compute_iv_slope_weight_for_identification(
            u=u,
            d=d,
            instrument=instrument,
            moments=moments,
        )

    if estimand.esttype == "late":
        if d == 1:
            weights_by_z = _weight_late(u, u_lo=estimand.u_lo, u_hi=estimand.u_hi)
        else:
            weights_by_z = -_weight_late(u, u_lo=estimand.u_lo, u_hi=estimand.u_hi)

        out = weights_by_z

    if estimand.esttype == "cross":
        out = _compute_cross_weight_for_identification(
            u=u,
            d=d,
            instrument=instrument,
            dz_cross=estimand.dz_cross,
        )

    return out * (basis_function["u_hi"] - basis_function["u_lo"])


def _compute_bernstein_weights(
    estimand: Estimand,
    d: int,
    basis_function: dict,
    instrument: Instrument,
    moments: dict | None = None,
) -> float:
    _bfunc = basis_function["func"]

    # For late target, we can compute the weight directly
    if estimand.esttype == "late":
        _s_late = partial(s_late, d=d, u_lo=estimand.u_lo, u_hi=estimand.u_hi)

        return integrate.quad(lambda u: _bfunc(u) * _s_late(u=u), 0, 1)[0]

    # Step 1: Get weight function s(D, Z) depending on the estimand type
    if estimand.esttype == "ols_slope":
        if moments is None:
            moments = _compute_moments_for_weights(estimand, instrument)

        _s_ols_slope = partial(
            s_ols_slope,
            d=d,
            ed=moments["expectation_d"],
            var_d=moments["variance_d"],
        )

        def _sdz(z, u):
            return _s_ols_slope(z=z, u=u)

    if estimand.esttype == "iv_slope":
        if moments is None:
            moments = _compute_moments_for_weights(estimand, instrument)

        _s_iv_slope = partial(
            s_iv_slope,
            ez=moments["expectation_z"],
            cov_dz=moments["covariance_dz"],
        )

        def _sdz(z, u):
            return _s_iv_slope(z=z, u=u)

    if estimand.esttype == "cross":

        def _sdz(z, u):
            return s_cross(d=d, z=z, dz_cross=estimand.dz_cross, u=u)

    # Step 2: Indicator for propensity score
    if d == 0:

        def _ind(z, u):
            return instrument.pscores[np.where(instrument.support == z)] < u

    elif d == 1:

        def _ind(z, u):
            return instrument.pscores[np.where(instrument.support == z)] >= u

    # Step 3: Compute the weight separately for each element of z
    weight = 0

    for z in instrument.support:
        # Make sure to binds z to the function definition in every iteration.
        # See: https://docs.astral.sh/ruff/rules/function-uses-loop-variable/.
        _to_integrate = partial(lambda u, z: _sdz(z, u) * _ind(z, u) * _bfunc(u), z=z)
        _integral = integrate.quad(_to_integrate, 0, 1)[0]
        _pos = np.where(instrument.support == z)[0][0]
        weight += _integral * instrument.pmf[_pos]

    return weight


def _weight_late(u, u_lo, u_hi):
    """Weight function for late target."""
    if u_lo < u < u_hi:
        return 1 / (u_hi - u_lo)
    return 0


def _weight_ols(u, d, pz, ed, var_d):
    """Weight function for OLS target."""
    if d == 0:
        return s_ols_slope(d, ed, var_d) if u > pz else 0
    return s_ols_slope(d, ed, var_d) if u <= pz else 0


def _weight_iv_slope(u, d, z, pz, ez, cov_dz):
    """Weight function for IV slope target."""
    if d == 0:
        return s_iv_slope(z, ez, cov_dz) if u > pz else 0
    return s_iv_slope(z, ez, cov_dz) if u <= pz else 0


def _weight_cross(u, d, z, pz, dz_cross):
    """Weight function for unconditional cross-moments E[D=d, Z=z]."""
    if d == 0:
        return s_cross(d, z, dz_cross) if u > pz else 0
    return s_cross(d, z, dz_cross) if u <= pz else 0


def _compute_ols_weight_for_identification(u, d, instrument: Instrument, moments):
    expectation_d = moments["expectation_d"]
    variance_d = moments["variance_d"]

    def _weight(u, d, pz):
        return _weight_ols(u, d, pz, ed=expectation_d, var_d=variance_d)

    pdf_z = instrument.pmf
    pscore_z = instrument.pscores

    weights_by_z = [_weight(u, d, pz) * pdf_z[i] for i, pz in enumerate(pscore_z)]
    return np.sum(weights_by_z)


def _compute_iv_slope_weight_for_identification(u, d, instrument: Instrument, moments):
    expectation_z = moments["expectation_z"]
    covariance_dz = moments["covariance_dz"]

    def _weight(u, d, z, pz):
        return _weight_iv_slope(u, d, z, pz, ez=expectation_z, cov_dz=covariance_dz)

    pdf_z = instrument.pmf
    pscore_z = instrument.pscores
    support_z = instrument.support

    weights_by_z = [
        _weight(u, d, z, pz) * pdf_z[i]
        for i, (z, pz) in enumerate(zip(support_z, pscore_z, strict=True))
    ]
    return np.sum(weights_by_z)


def _compute_cross_weight_for_identification(u, d, instrument: Instrument, dz_cross):
    def _weight(u, d, z, pz):
        return _weight_cross(u, d, z, pz, dz_cross=dz_cross)

    pdf_z = instrument.pmf
    pscore_z = instrument.pscores
    support_z = instrument.support

    weights_by_z = [
        _weight(u, d, z, pz) * pdf_z[i]
        for i, (z, pz) in enumerate(zip(support_z, pscore_z, strict=True))
    ]

    return np.sum(weights_by_z)


def _check_identification_arguments(
    target,
    identified_estimands,
    basis_funcs,
    m0_dgp,
    m1_dgp,
    instrument,
    u_partition,
    shape_constraints,
    method,
):
    """Check identification arguments.

    Fail and comprehensive report if invalid.

    """
    error_report = ""
    error_report += _error_report_estimand(target)
    for ident in identified_estimands:
        error_report += _error_report_estimand(ident)
    error_report += _error_report_basis_funcs(basis_funcs)
    error_report += _error_report_mtr_function(m0_dgp)
    error_report += _error_report_mtr_function(m1_dgp)
    error_report += _error_report_instrument(instrument)
    error_report += _error_report_u_partition(u_partition)
    error_report += _error_report_method(method)
    error_report += _error_report_shape_constraints(shape_constraints)

    if error_report:
        raise ValueError(error_report)

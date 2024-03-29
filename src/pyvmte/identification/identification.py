"""Function for identification."""
from collections.abc import Callable

import coptpy as cp  # type: ignore
import numpy as np
from coptpy import COPT
from scipy import integrate  # type: ignore
from scipy.optimize import (  # type: ignore
    linprog,  # type: ignore
)

from pyvmte.config import Estimand, Instrument
from pyvmte.utilities import compute_moments, s_cross, s_iv_slope, s_late, s_ols_slope


def identification(
    target: Estimand,
    identified_estimands: list[Estimand],
    basis_funcs: list[dict],
    m0_dgp: Callable,
    m1_dgp: Callable,
    instrument: Instrument,
    u_partition: np.ndarray,
    method: str = "highs",
):
    """Compute bounds on target estimand given identified estimands and DGP.

    Args:
        target (dict): Dictionary containing all information about the target estimand.
        identified_estimands (dict or list of dicts): Dictionary containing all
            information about the identified estimand(s). List of dicts if multiple
            identified estimands.
        basis_funcs (dict or list of dicts): Dictionaries describing the basis
            functions.
        m0_dgp (function): The MTR function for d=0 of the DGP.
        m1_dgp (function): The MTR function for d=1 of the DGP.
        instrument (dict): Dictionary containing all information about the instrument.
        u_partition (list or np.array, optional): Partition of u for basis_funcs.
            Defaults to None.
        method (str, optional): Method for solving the linear program.
            Implemented are: all methods supported by scipy.linprog as well as copt.
            Defaults to "highs" using scipy.linprog.

    Returns:
        dict: A dictionary containing the upper and lower bound of the target estimand.

    """
    # ==================================================================================
    # Perform some additional checks on arguments
    # ==================================================================================
    if isinstance(identified_estimands, dict):
        identified_estimands = [identified_estimands]

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

    # ==================================================================================
    # Solve linear program
    # ==================================================================================

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
    bfunc_type = basis_funcs[0]["type"]

    if bfunc_type == "constant":
        # TODO (@buddejul):  think about separating identification and estimation
        if moments is None and instrument is not None:
            moments = _compute_moments_for_weights(target, instrument)
        # FIXME check why this all works with moments = None and instruments = None
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

    return np.array(c)


def _compute_equality_constraint_matrix(
    identified_estimands: list[Estimand],
    basis_funcs: list,
    instrument: Instrument,
) -> np.ndarray:
    """Compute weight matrix for equality constraints."""
    bfunc_type = basis_funcs[0]["type"]

    if bfunc_type == "constant":
        c_matrix = []
        for target in identified_estimands:
            c_row = _compute_choice_weights(
                target=target,
                basis_funcs=basis_funcs,
                instrument=instrument,
            )

            c_matrix.append(c_row)

    return np.array(c_matrix)


def _solve_lp(lp_inputs: dict, max_or_min: str, method: str) -> float:
    """Wrapper for solving the linear program."""
    c = np.array(lp_inputs["c"]) if max_or_min == "min" else -np.array(lp_inputs["c"])

    b_eq = lp_inputs["b_eq"]
    a_eq = lp_inputs["a_eq"]

    if method == "copt":
        return _solve_lp_copt(c, a_eq, b_eq)

    return linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, 1)).fun


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
) -> float:
    """Wrapper for solving LP using copt algorithm."""
    env = cp.Envr()
    model = env.createModel("identification")
    x = model.addMVar(len(c), nameprefix="x", lb=0, ub=1)
    model.setObjective(c @ x, COPT.MINIMIZE)
    model.addMConstr(a_eq, x, "E", b_eq, nameprefix="c")
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
    # TODO (@buddejul):  change this, do not actually need u here I think
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

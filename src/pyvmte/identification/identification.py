"""Function for identification."""
from collections.abc import Callable

import coptpy as cp  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from coptpy import COPT
from scipy.optimize import (  # type: ignore
    linprog,  # type: ignore
)

from pyvmte.config import Estimand, Instrument
from pyvmte.utilities import (
    compute_constant_spline_weights,
    gamma_star,
)


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
    a = gamma_star(
        md=m0,
        d_value=0,
        estimand=estimand,
        instrument=instrument,
        u_part=u_part,
    )
    b = gamma_star(
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
    instrument: Instrument | None = None,
    moments: dict | None = None,
    data: pd.DataFrame | None = None,
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
                weight = compute_constant_spline_weights(
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

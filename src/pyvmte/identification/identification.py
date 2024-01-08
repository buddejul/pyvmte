"""Function for identification."""

import numpy as np
from pyvmte.utilities import (
    gamma_star,
    _compute_constant_spline_weights,
    _generate_u_partition_from_basis_funcs,
    _generate_partition_midpoints,
)

from scipy.optimize import linprog


def identification(
    target,
    identified_estimands,
    basis_funcs,
    m0_dgp,
    m1_dgp,
    instrument,
    u_partition=None,
    analytical_integration=False,
):
    """Compute bounds on target estimand given identified estimands based on known DGP
    (identification).

    Args:
        target (dict): Dictionary containing all information about the target estimand.
        identified_estimands (dict or list of dicts): Dictionary containing all information about the identified estimand(s). List of dicts if multiple identified estimands.
        basis_funcs (dict or list of dicts): Dictionaries describing the basis functions.
        m0_dgp (function): The MTR function for d=0 of the DGP.
        m1_dgp (function): The MTR function for d=1 of the DGP.
        instrument (dict): Dictionary containing all information about the instrument.
        u_partition (list or np.array, optional): Partition of u for basis_funcs. Defaults to None.
        analytical_integration (bool, optional): Whether to use analytical integration. Defaults to False.

    Returns:
        dict: A dictionary containing the upper and lower bound of the target estimand.

    """
    if isinstance(identified_estimands, dict):
        identified_estimands = [identified_estimands]

    lp_inputs = {}

    lp_inputs["c"] = _compute_choice_weights(target, basis_funcs, instrument=instrument)
    lp_inputs["b_eq"] = _compute_identified_estimands(
        identified_estimands, m0_dgp, m1_dgp, u_partition, instrument
    )
    lp_inputs["A_eq"] = _compute_equality_constraint_matrix(
        identified_estimands, basis_funcs, instrument=instrument
    )

    upper_bound = (-1) * _solve_lp(lp_inputs, "max").fun
    lower_bound = _solve_lp(lp_inputs, "min").fun

    return {"upper_bound": upper_bound, "lower_bound": lower_bound}


def _get_helper_variables():
    pass


def _compute_identified_estimands(
    identified_estimands, m0_dgp, m1_dgp, u_part, instrument
):
    """Wrapper for computing identified estimands based on provided dgp."""
    out = []
    for estimand in identified_estimands:
        result = _compute_estimand(estimand, m0_dgp, m1_dgp, u_part, instrument)
        out.append(result)

    return out


def _compute_estimand(estimand, m0, m1, u_part=None, instrument=None):
    """Compute single identified estimand."""
    a = gamma_star(
        md=m0,
        d=0,
        estimand_dict=estimand,
        instrument=instrument,
        u_part=u_part,
    )
    b = gamma_star(
        md=m1,
        d=1,
        estimand_dict=estimand,
        instrument=instrument,
        u_part=u_part,
    )

    return a + b


def _compute_choice_weights(target, basis_funcs, instrument=None):
    """Compute weights on the choice variables."""

    bfunc_type = basis_funcs[0]["type"]

    if bfunc_type == "constant":
        u_partition = _generate_u_partition_from_basis_funcs(basis_funcs)
        moments = _compute_moments_for_weights(target, instrument)
        u_evaluation_points = _generate_partition_midpoints(u_partition)

        c = []
        for d in [0, 1]:
            for u in u_evaluation_points:
                weight = _compute_constant_spline_weights(
                    estimand=target,
                    u=u,
                    d=d,
                    instrument=instrument,
                    moments=moments,
                )
                c.append(weight)

    else:
        c = []

        for d in [0, 1]:
            for basis_func in basis_funcs:
                weight = gamma_star(
                    md=basis_func,
                    estimand_dict=target,
                    d=d,
                    instrument=instrument,
                )
                c.append(weight)

    return c


def _compute_equality_constraint_matrix(identified_estimands, basis_funcs, instrument):
    """Compute weight matrix for equality constraints."""
    bfunc_type = basis_funcs[0]["type"]

    if bfunc_type == "constant":
        c_matrix = []
        for target in identified_estimands:
            c_row = _compute_choice_weights(
                target=target, basis_funcs=basis_funcs, instrument=instrument
            )

            c_matrix.append(c_row)

    else:
        c_matrix = []

        for target in identified_estimands:
            c_row = []

            for d in [0, 1]:
                for basis_func in basis_funcs:
                    weight = gamma_star(
                        md=basis_func, estimand_dict=target, d=d, instrument=instrument
                    )

                    c_row.append(weight)

            c_matrix.append(c_row)

    return np.array(c_matrix)


def _solve_lp(lp_inputs, max_or_min):
    """Solve the linear program."""

    if max_or_min == "min":
        c = np.array(lp_inputs["c"])
    else:
        c = -np.array(lp_inputs["c"])

    b_eq = lp_inputs["b_eq"]
    A_eq = lp_inputs["A_eq"]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))

    return result


def _compute_moments_for_weights(target, instrument):
    """Compute relevant moments for computing weights on choice variables given target
    parameter and binary treatment."""
    moments = {}

    if target["type"] == "ols_slope":
        moments["expectation_d"] = _compute_binary_expectation_using_lie(
            d_pdf_given_z=instrument["pscore_z"],
            z_pdf=instrument["pdf_z"],
        )
        moments["variance_d"] = moments["expectation_d"] * (
            1 - moments["expectation_d"]
        )

    if target["type"] == "iv_slope":
        moments["expectation_z"] = _compute_expectation(
            support=instrument["support_z"], pdf=instrument["pdf_z"]
        )
        moments["covariance_dz"] = _compute_covariance_dz(
            support_z=instrument["support_z"],
            pscore_z=instrument["pscore_z"],
            pdf_z=instrument["pdf_z"],
        )

    return moments


def _compute_binary_expectation_using_lie(d_pdf_given_z, z_pdf):
    """Compute expectation of d using the law of iterated expectations."""

    return d_pdf_given_z @ z_pdf


def _compute_expectation(support, pdf):
    """Compute expectation of a discrete random variable."""
    return support @ pdf


def _compute_covariance_dz(support_z, pscore_z, pdf_z):
    """Compute covariance between binary treatment and discrete instrument."""
    ez = support_z @ pdf_z
    ed = pscore_z @ pdf_z
    edz = np.sum(support_z * pscore_z * pdf_z)
    return edz - ed * ez

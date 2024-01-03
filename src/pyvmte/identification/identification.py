"""Function for identification."""

import numpy as np
from pyvmte.utilities import gamma_star

from scipy.optimize import linprog


def identification(
    target,
    identified_estimands,
    basis_funcs,
    m0_dgp,
    m1_dgp,
    u_partition=None,
    support_z=None,
    pscore_z=None,
    pdf_z=None,
    analytical_integration=False,
):
    values_identified = _compute_identified_estimands(
        identified_estimands,
        m0_dgp,
        m1_dgp,
        u_partition,
        support_z,
        pscore_z,
        pdf_z,
    )

    lp_inputs = {}

    lp_inputs["c"] = _compute_choice_weights(
        target,
        basis_funcs,
        support_z,
        pscore_z,
        pdf_z,
    )
    lp_inputs["b_eq"] = values_identified
    lp_inputs["A_eq"] = _compute_equality_constraint_matrix(
        identified_estimands,
        basis_funcs,
        support_z,
        pscore_z,
        pdf_z,
    )

    upper_bound = (-1) * _solve_lp(lp_inputs, "max").fun
    lower_bound = _solve_lp(lp_inputs, "min").fun

    return {"upper_bound": upper_bound, "lower_bound": lower_bound}


def _get_helper_variables():
    pass


def _compute_identified_estimands(
    identified_estimands,
    m0_dgp,
    m1_dgp,
    u_part,
    support_z,
    pscore_z,
    pdf_z,
):
    """Wrapper for computing identified estimands based on provided dgp."""
    out = []
    for estimand in identified_estimands:
        result = _compute_estimand(
            estimand, m0_dgp, m1_dgp, u_part, support_z, pscore_z, pdf_z
        )
        out.append(result)

    return out


def _compute_estimand(
    estimand,
    m0,
    m1,
    u_part=None,
    support_z=None,
    pscore_z=None,
    pdf_z=None,
):
    """Compute single identified estimand."""
    if estimand["type"] == "late":
        out = _compute_estimand_late(m0, m1, estimand["u_lo"], estimand["u_hi"], u_part)
    elif estimand["type"] == "iv_slope":
        out = _compute_estimand_iv_slope(m0, m1, u_part, support_z, pscore_z, pdf_z)
    elif estimand["type"] == "ols_slope":
        out = _compute_estimand_ols_slope(m0, m1, u_part, support_z, pscore_z, pdf_z)
    elif estimand["type"] == "cross":
        out = _compute_estimand_crossmoment(m0, m1)

    return out


def _compute_estimand_late(m0, m1, u_lo, u_hi, u_part):
    """Compute identified estimand for late."""
    a = gamma_star(m0, 0, estimand="late", u_lo=u_lo, u_hi=u_hi, u_part=u_part)
    b = gamma_star(m1, 1, estimand="late", u_lo=u_lo, u_hi=u_hi, u_part=u_part)

    return a + b


def _compute_estimand_iv_slope(m0, m1, u_part, support_z, pscore_z, pdf_z):
    """Compute identified estimand for iv slope."""
    a = gamma_star(
        m0,
        0,
        "iv_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    b = gamma_star(
        m1,
        1,
        "iv_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    return a + b


def _compute_estimand_ols_slope(m0, m1, u_part, support_z, pscore_z, pdf_z):
    """Compute identified estimand for ols slope."""
    a = gamma_star(
        m0,
        0,
        "ols_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    b = gamma_star(
        m1,
        1,
        "ols_slope",
        support_z=support_z,
        pscore_z=pscore_z,
        pdf_z=pdf_z,
        u_part=u_part,
    )

    return a + b


def _compute_estimand_crossmoment():
    """Compute identified estimand for crossmoment."""
    pass


def _compute_choice_weights(target, basis_funcs, support_z, pscore_z, pdf_z):
    """Compute weights on the choice variables."""
    c = []

    for d in [0, 1]:
        for basis_func in basis_funcs:
            weight = gamma_star(
                md=basis_func,
                estimand=target["type"],
                d=d,
                u_lo=target["u_lo"],
                u_hi=target["u_hi"],
                support_z=support_z,
                pscore_z=pscore_z,
                pdf_z=pdf_z,
            )
            c.append(weight)

    return c


def _compute_equality_constraint_matrix(
    identified_estimands, basis_funcs, support_z, pscore_z, pdf_z
):
    """Compute weight matrix for equality constraints."""

    c_matrix = []

    for target in identified_estimands:
        c_row = []

        for d in [0, 1]:
            for basis_func in basis_funcs:
                weight = gamma_star(
                    md=basis_func,
                    estimand=target["type"],
                    d=d,
                    u_lo=target["u_lo"],
                    u_hi=target["u_hi"],
                    support_z=support_z,
                    pscore_z=pscore_z,
                    pdf_z=pdf_z,
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

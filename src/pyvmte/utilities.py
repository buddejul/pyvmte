"""Utilities used in various parts of the project."""

import math
import numpy as np
from scipy import integrate


def gamma_star(
    md,
    d,
    estimand,
    u_lo=0,
    u_hi=1,
    support_z=None,
    pscore_z=None,
    pdf_z=None,
    dz_cross=None,
    analyt_int=False,
    u_part=None,
    u_part_lo=None,
    u_part_hi=None,
):
    """Compute gamma* for a given MTR function and estimand
    Args:
        md (function): MTR function
        d (np.int): value of the treatment
        estimand (str): the estimand to compute
        u_lo (float): lower bound of late target
        u_hi (float): upper bound of late target
        support_z (np.array): support of the instrument
        pscore_z (np.array): propensity given the instrument
        pdf_z (np.array): probability mass function of the instrument
        dz_cross (list): list of tuples of the form (d_spec, z_spec) for cross-moment
        analyt_int (Boolean): Whether to integrate manually or use analytic results
    """

    if estimand == "late":
        return integrate.quad(lambda u: md(u) * s_late(d, u, u_lo, u_hi), u_lo, u_hi)[0]

    # Do integration manually via scipy integrate
    if analyt_int == False:
        if estimand == "iv_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)

            if d == 0:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] < u:
                        return md(u) * s_iv_slope(z, ez, cov_dz)
                    else:
                        return 0

            if d == 1:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] > u:
                        return md(u) * s_iv_slope(z, ez, cov_dz)
                    else:
                        return 0

            # Integrate func over u in [0,1] for every z in support_z
            return np.sum(
                [
                    integrate.quad(func, 0, 1, args=(z,))[0] * pdf_z[i]
                    for i, z in enumerate(support_z)
                ]
            )

        if estimand == "ols_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)
            var_d = ed * (1 - ed)

            if d == 0:
                # need to condition on z
                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] < u:
                        return md(u) * s_ols_slope(d, ed, var_d)
                    else:
                        return 0

            if d == 1:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] > u:
                        return md(u) * s_ols_slope(d, ed, var_d)
                    else:
                        return 0

            # Integrate func over u in [0,1] for every z in support_z
            return np.sum(
                [
                    integrate.quad(func, 0, 1, args=(z,))[0] * pdf_z[i]
                    for i, z in enumerate(support_z)
                ]
            )

        if estimand == "cross":
            if d == 0:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] < u:
                        return md(u) * s_cross(d, z, dz_cross)
                    else:
                        return 0

            if d == 1:

                def func(u, z):
                    if pscore_z[np.where(support_z == z)[0][0]] >= u:
                        return md(u) * s_cross(d, z, dz_cross)
                    else:
                        return 0

            # Integrate func over u in [0,1] for every z in support_z
            return np.sum(
                [
                    integrate.quad(func, 0, 1, args=(z,))[0] * pdf_z[i]
                    for i, z in enumerate(support_z)
                ]
            )

    # Use analytic results on constant spline basis
    if analyt_int == True:
        if estimand == "iv_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)

            if d == 0:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_iv_slope(z, ez, cov_dz)
                        * (pscore_z[j] <= u_part_lo)
                        for j, z in enumerate(support_z)
                    ]
                )

            if d == 1:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_iv_slope(z, ez, cov_dz)
                        * (pscore_z[j] >= u_part_hi)
                        for j, z in enumerate(support_z)
                    ]
                )

        if estimand == "ols_slope":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)
            var_d = ed * (1 - ed)

            if d == 0:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_ols_slope(d, ed, var_d)
                        * (pscore_z[j] <= u_part_lo)
                        for j, z in enumerate(support_z)
                    ]
                )

            if d == 1:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j]
                        * s_ols_slope(d, ed, var_d)
                        * (pscore_z[j] >= u_part_hi)
                        for j, z in enumerate(support_z)
                    ]
                )

        if estimand == "cross":
            ez, ed, edz, cov_dz = compute_moments(support_z, pdf_z, pscore_z)
            var_d = ed * (1 - ed)

            if d == 0:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j] * s_cross(d, z, dz_cross) * (pscore_z[j] <= u_part_lo)
                        for j, z in enumerate(support_z)
                    ]
                )

            if d == 1:
                return (u_part_hi - u_part_lo) * np.sum(
                    [
                        pdf_z[j] * s_cross(d, z, dz_cross) * (pscore_z[j] >= u_part_hi)
                        for j, z in enumerate(support_z)
                    ]
                )


def compute_moments(supp_z, f_z, prop_z):
    """Calculate E[z], E[d], E[dz], Cov[d,z] for a discrete instrument z
    and binary d
    Args:
        supp_z (np.array): support of the instrument
        f_z (np.array): probability mass function of the instrument
        prop_z (np.array): propensity score of the instrument
    """

    ez = np.sum(supp_z * f_z)
    ed = np.sum(prop_z * f_z)
    edz = np.sum(supp_z * prop_z * f_z)
    cov_dz = edz - ed * ez

    return ez, ed, edz, cov_dz


def s_iv_slope(z, ez, cov_dz):
    """IV-like specification s(d,z): IV slope
    Args:
        z (np.int): value of the instrument
        ez (np.float): expected value of the instrument
        cov_dz (np.float): covariance between treatment and instrument
    """
    return (z - ez) / cov_dz


def s_ols_slope(d, ed, var_d):
    """OLS-like specification s(d,z): OLS slope
    Args:
        d (np.int): value of the treatment
        ed (np.float): expected value of the treatment
        var_d (np.float): variance of the treatment
    """
    return (d - ed) / var_d


def s_late(d, u, u_lo, u_hi):
    """IV-like specification s(d,z): late."""
    # Return 1 divided by u_hi - u_lo if u_lo < u < u_hi, 0 otherwise
    if u_lo < u < u_hi:
        w = 1 / (u_hi - u_lo)
    else:
        w = 0

    if d == 1:
        return w
    else:
        return -w


def s_cross(d, z, dz_cross):
    """IV_like specification s(d,z): Cross-moment d_spec * z_spec."""
    if (isinstance(d, np.ndarray) or d in [0, 1]) and isinstance(z, np.ndarray):
        return np.logical_and(d == dz_cross[0], z == dz_cross[1]).astype(int)
    else:
        return 1 if d == dz_cross[0] and z == dz_cross[1] else 0


def bern_bas(n, v, x):
    """Bernstein polynomial basis of degree n and index v at point x."""
    return math.comb(n, v) * x**v * (1 - x) ** (n - v)


def load_paper_dgp():
    """Load the dgp from MST 2018 ECMA."""
    out = {}
    out["m0"] = (
        lambda u: 0.6 * bern_bas(2, 0, u)
        + 0.4 * bern_bas(2, 1, u)
        + 0.3 * bern_bas(2, 2, u)
    )
    out["m1"] = (
        lambda u: 0.75 * bern_bas(2, 0, u)
        + 0.5 * bern_bas(2, 1, u)
        + 0.25 * bern_bas(2, 2, u)
    )
    out["support_z"] = np.array([0, 1, 2])
    out["pscore_z"] = np.array([0.35, 0.6, 0.7])
    out["pdf_z"] = np.array([0.5, 0.4, 0.1])
    out["ols_slope"] = 0.253
    out["late_35_90"] = 0.046
    out["iv_slope"] = 0.074

    return out

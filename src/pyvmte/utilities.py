"""Utilities used in various parts of the project."""

import math
import numpy as np
import pandas as pd
from scipy import integrate


def gamma_star(
    md,
    d,
    estimand_dict,
    instrument=None,
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

    estimand = estimand_dict["type"]
    u_lo = estimand_dict.get("u_lo")
    u_hi = estimand_dict.get("u_hi")
    dz_cross = estimand_dict.get("dz_cross")

    if instrument is not None:
        pdf_z = instrument.get("pdf_z")
        pscore_z = instrument.get("pscore_z")
        support_z = instrument.get("support_z")

    if estimand == "late":
        # breakpoint()
        return integrate.quad(lambda u: md(u) * s_late(d, u, u_lo, u_hi), 0, 1)[0]

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
    out["u_partition"] = [0, 0.35, 0.6, 0.7, 0.9, 1]

    out["joint_pmf_dz"] = {
        1: {0: 0.175, 1: 0.24, 2: 0.07},
        0: {0: 0.325, 1: 0.16, 2: 0.03},
    }

    out["expectation_d"] = np.sum(out["pscore_z"] * out["pdf_z"])
    out["variance_d"] = out["expectation_d"] * (1 - out["expectation_d"])
    out["expectation_z"] = np.sum(out["support_z"] * out["pdf_z"])

    out["covariance_dz"] = np.sum(
        [
            out["joint_pmf_dz"][d][z]
            * (d - out["expectation_d"])
            * (z - out["expectation_z"])
            for d in [0, 1]
            for z in [0, 1, 2]
        ]
    )

    return out


def simulate_data_from_paper_dgp(sample_size, rng):
    """Simulate data using the dgp from MST 2018 ECMA."""
    data = pd.DataFrame()

    dgp = load_paper_dgp()

    z_dict = dict(zip(dgp["support_z"], dgp["pscore_z"]))

    sampled = np.random.choice(dgp["support_z"], size=sample_size, p=dgp["pdf_z"])

    pscores_corresponding = np.array([z_dict[i] for i in sampled])

    data["z"] = sampled
    data["pscore_z"] = pscores_corresponding

    data["u"] = rng.uniform(size=sample_size)

    data["d"] = data["u"] < data["pscore_z"]

    m0 = dgp["m0"]
    m1 = dgp["m1"]

    data["y"] = np.where(data["d"] == 0, m0(data["u"]), m1(data["u"]))

    data["pscore_z"] = data["pscore_z"].astype(pd.Float64Dtype())
    data["z"] = data["z"].astype(pd.Int64Dtype())
    data["u"] = data["u"].astype(pd.Float64Dtype())
    data["d"] = data["d"].astype(pd.BooleanDtype())
    data["y"] = data["y"].astype(pd.Float64Dtype())

    return data


def _return_weight_function(
    estimand, d, pz=None, ed=None, var_d=None, ez=None, cov_dz=None
):
    """Return weight function corresponding to equation (6) in the paper."""

    if estimand["type"] == "late":
        if d == 0:
            return lambda u: -_weight_late(u, estimand["u_lo"], estimand["u_hi"])
        else:
            return lambda u: _weight_late(u, estimand["u_lo"], estimand["u_hi"])

    if estimand["type"] == "ols_slope":
        if d == 0:
            return lambda u: -_weight_ols(u, d, pz, ed, var_d)
        else:
            return lambda u: _weight_ols(u, d, pz, ed, var_d)

    if estimand["type"] == "iv_slope":
        if d == 0:
            return lambda u, z: -_weight_iv_slope(u, d, z, pz, ez, cov_dz)
        else:
            return lambda u, z: _weight_iv_slope(u, d, z, pz, ez, cov_dz)


def _weight_late(u, u_lo, u_hi):
    """Weight function for late target."""
    if u_lo < u < u_hi:
        return 1 / (u_hi - u_lo)
    else:
        return 0


def _weight_ols(u, d, pz, ed, var_d):
    """Weight function for OLS target."""
    if d == 0:
        return s_ols_slope(d, ed, var_d) if u > pz else 0
    else:
        return s_ols_slope(d, ed, var_d) if u <= pz else 0


def _weight_iv_slope(u, d, z, pz, ez, cov_dz):
    """Weight function for IV slope target."""
    if d == 0:
        return s_iv_slope(z, ez, cov_dz) if u > pz else 0
    else:
        return s_iv_slope(z, ez, cov_dz) if u <= pz else 0


def _weight_cross(u, d, z, pz, dz_cross):
    """Weight function for unconditional cross-moments E[D=d, Z=z]."""
    if d == 0:
        return s_cross(d, z, dz_cross) if u > pz else 0
    else:
        return s_cross(d, z, dz_cross) if u <= pz else 0


def _compute_constant_spline_weights(
    estimand,
    u,
    d,
    instrument=None,
    moments=None,
):
    """Compute weights for constant spline basis. We use that for a constant spline
    basis with the right partition the weights are constant on each interval of the
    partition.

    We condition on z and compute the weights for each interval of the partition using
    the law of iterated expectations.

    """
    if estimand["type"] == "ols_slope":
        expectation_d = moments["expectation_d"]
        variance_d = moments["variance_d"]
        _weight = lambda u, d, pz: _weight_ols(u, d, pz, expectation_d, variance_d)
        pdf_z = instrument["pdf_z"]
        pscore_z = instrument["pscore_z"]

        weights_by_z = [_weight(u, d, pz) * pdf_z[i] for i, pz in enumerate(pscore_z)]

    if estimand["type"] == "iv_slope":
        expectation_z = moments["expectation_z"]
        covariance_dz = moments["covariance_dz"]
        _weight = lambda u, d, z, pz: _weight_iv_slope(
            u, d, z, pz, ez=expectation_z, cov_dz=covariance_dz
        )
        pdf_z = instrument["pdf_z"]
        pscore_z = instrument["pscore_z"]
        support_z = instrument["support_z"]

        weights_by_z = [
            _weight(u, d, z, pz) * pdf_z[i]
            for i, (z, pz) in enumerate(zip(support_z, pscore_z))
        ]

    if estimand["type"] == "late":
        if d == 1:
            weights_by_z = _weight_late(u, u_lo=estimand["u_lo"], u_hi=estimand["u_hi"])
        else:
            weights_by_z = -_weight_late(
                u, u_lo=estimand["u_lo"], u_hi=estimand["u_hi"]
            )

    if estimand["type"] == "cross":
        _weight = lambda u, d, z, pz: _weight_cross(
            u, d, z, pz, dz_cross=estimand["dz_cross"]
        )

        pdf_z = instrument["pdf_z"]
        pscore_z = instrument["pscore_z"]
        support_z = instrument["support_z"]

        weights_by_z = [
            _weight(u, d, z, pz) * pdf_z[i]
            for i, (z, pz) in enumerate(zip(support_z, pscore_z))
        ]

    return np.sum(weights_by_z)


def _generate_u_partition_from_basis_funcs(basis_funcs):
    """Generate u_partition from basis_funcs dictionaries."""

    u_partition = [0]

    for basis_func in basis_funcs:
        u_partition.append(basis_func["u_hi"])

    return u_partition


def _generate_partition_midpoints(partition):
    """Generate midpoints of partition."""
    return np.array(
        [(partition[i] + partition[i + 1]) / 2 for i in range(len(partition) - 1)]
    )

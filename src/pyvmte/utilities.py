"""Utilities used in various parts of the project."""
import contextlib
import math
import os
from collections.abc import Callable

import numpy as np
import pandas as pd  # type: ignore
from scipy import integrate  # type: ignore

from pyvmte.classes import Estimand, Instrument


# TODO(@buddejul):  remove most of this/make simpler; only used in identification
def gamma_star(
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


def compute_moments(supp_z, f_z, prop_z):
    """Calculate E[z], E[d], E[dz], Cov[d,z] for a discrete instrument z and binary d.

    Args:
        supp_z (np.array): support of the instrument
        f_z (np.array): probability mass function of the instrument
        prop_z (np.array): propensity score of the instrument.

    """
    ez = np.sum(supp_z * f_z)
    ed = np.sum(prop_z * f_z)
    edz = np.sum(supp_z * prop_z * f_z)
    cov_dz = edz - ed * ez

    return ez, ed, edz, cov_dz


def s_iv_slope(z, ez, cov_dz):
    """IV-like specification s(d,z): IV slope.

    Args:
        z (np.int): value of the instrument
        ez (np.float): expected value of the instrument
        cov_dz (np.float): covariance between treatment and instrument.

    """
    return (z - ez) / cov_dz


def s_ols_slope(d, ed, var_d):
    """OLS-like specification s(d,z): OLS slope.

    Args:
        d (np.int): value of the treatment
        ed (np.float): expected value of the treatment
        var_d (np.float): variance of the treatment.

    """
    return (d - ed) / var_d


def s_late(d, u, u_lo, u_hi):
    """IV-like specification s(d,z): late."""
    # Return 1 divided by u_hi - u_lo if u_lo < u < u_hi, 0 otherwise
    w = 1 / (u_hi - u_lo) if u_lo < u < u_hi else 0

    if d == 1:
        return w
    return -w


def s_cross(d, z, dz_cross):
    """IV_like specification s(d,z): Cross-moment d_spec * z_spec."""
    if (isinstance(d, np.ndarray) or d in [0, 1]) and isinstance(z, np.ndarray):
        return np.logical_and(d == dz_cross[0], z == dz_cross[1]).astype(int)
    return 1 if d == dz_cross[0] and z == dz_cross[1] else 0


def bern_bas(n, v, x):
    """Bernstein polynomial basis of degree n and index v at point x."""
    return math.comb(n, v) * x**v * (1 - x) ** (n - v)


def simulate_data_from_paper_dgp(sample_size, rng):
    """Simulate data using the dgp from MST 2018 ECMA."""
    data = pd.DataFrame()

    support = np.array([0, 1, 2])
    pmf = np.array([0.5, 0.4, 0.1])
    pscores = np.array([0.35, 0.6, 0.7])

    choices = np.hstack([support.reshape(-1, 1), pscores.reshape(-1, 1)])

    # Draw random ndices
    idx = rng.choice(support, size=sample_size, p=pmf)

    data = choices[idx]

    # Put data into df
    z = np.array(data[:, 0], dtype=int)
    pscores = data[:, 1]

    u = rng.uniform(size=sample_size)
    d = u < pscores

    y = np.empty(sample_size)
    idx = d == 0
    # TODO (@buddejul):  do this properly
    y[idx] = (
        +0.6 * (1 - u[idx]) ** 2 + 0.4 * 2 * u[idx] * (1 - u[idx]) + 0.3 * u[idx] ** 2
    )

    y[~idx] = (
        +0.75 * (1 - u[~idx]) ** 2
        + 0.5 * 2 * u[~idx] * (1 - u[~idx])
        + 0.25 * u[~idx] ** 2
    )

    return {"z": z, "d": d, "y": y, "u": u}


def _weight_late(u, u_lo, u_hi):
    """Weight function for late target."""
    if u_lo < u < u_hi:
        return 1 / (u_hi - u_lo)
    return 0


def _weight_ols(u, d, pz, ed, var_d, d_data=None):
    """Weight function for OLS target."""
    # FIXME difference between d to indicate type of weight function and data d
    if d_data is None:
        d_data = d
    if d == 0:
        return s_ols_slope(d_data, ed, var_d) if u > pz else 0
    return s_ols_slope(d_data, ed, var_d) if u <= pz else 0


def _weight_iv_slope(u, d, z, pz, ez, cov_dz):
    # FIXME difference between d to indicate type of weight function and data d
    """Weight function for IV slope target."""
    if d == 0:
        return s_iv_slope(z, ez, cov_dz) if u > pz else 0
    return s_iv_slope(z, ez, cov_dz) if u <= pz else 0


def _weight_cross(u, d, z, pz, dz_cross, d_data=None):
    """Weight function for unconditional cross-moments E[D=d, Z=z]."""
    # FIXME difference between d to indicate type of weight function and data d
    if d_data is None:
        d_data = d
    if d == 0:
        return s_cross(d_data, z, dz_cross) if u > pz else 0
    return s_cross(d_data, z, dz_cross) if u <= pz else 0


# TODO(@buddejul):  remove the data part have separate function for this
def compute_constant_spline_weights(
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


def _check_estimation_arguments(
    target,
    identified_estimands,
    basis_func_type,
    y_data,
    z_data,
    d_data,
    tolerance,
    x_data,
    u_partition,
):
    """Check arguments provided to estimation function.

    If there are errors, returns a comprehensive error report for all arguments.

    """
    error_report = ""

    data_dict = {
        "y": y_data,
        "z": z_data,
        "d": d_data,
    }

    # Check that all data arguments are numpy arrays
    for key, value in data_dict.items():
        if not isinstance(value, np.ndarray):
            error_report += f"Argument {key} is not a numpy array.\n"

    if error_report != "":
        raise EstimationArgumentError(error_report)


class EstimationArgumentError(Exception):
    """Raised when arguments to estimation function are not valid."""


@contextlib.contextmanager
def suppress_print():
    """Suppress print statements in context."""
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # noqa: PTH123
        yield

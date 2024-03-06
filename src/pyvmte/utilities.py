"""Utilities used in various parts of the project."""
import contextlib
import math
import os

import numpy as np
import pandas as pd  # type: ignore


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

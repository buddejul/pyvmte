"""Utilities used in various parts of the project."""

import contextlib
import math
import os

import numpy as np
import pandas as pd  # type: ignore
from scipy.interpolate import BPoly  # type: ignore[import-untyped]

from pyvmte.classes import Bern, Estimand, Instrument


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


def s_ols_slope(d, ed, var_d, z=None):
    """OLS-like specification s(d,z): OLS slope.

    Args:
        d (np.int): value of the treatment
        ed (np.float): expected value of the treatment
        var_d (np.float): variance of the treatment.
        z: Only used to allow for the same function signature.

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
    """Simulate data using the dgp from MST 2018 ECMA.

    Args:
        sample_size (int): The number of observations in the sample.
        rng (np.random.Generator): The random number generator.

    Returns:
        dict: A dictionary containing the simulated data.

    """
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
    y[idx] = (
        +0.6 * (1 - u[idx]) ** 2 + 0.4 * 2 * u[idx] * (1 - u[idx]) + 0.3 * u[idx] ** 2
    )

    y[~idx] = (
        +0.75 * (1 - u[~idx]) ** 2
        + 0.5 * 2 * u[~idx] * (1 - u[~idx])
        + 0.25 * u[~idx] ** 2
    )

    return {"z": z, "d": d, "y": y, "u": u}


@contextlib.contextmanager
def suppress_print():
    """Suppress print statements in context."""
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # noqa: PTH123
        yield


def _error_report_estimand(estimand: Estimand):
    """Return error message if estimand is not valid."""
    error_report = ""
    if not isinstance(estimand, Estimand):
        error_report += f"Identified estimand {estimand} is not of type Estimand."
    else:
        if estimand.esttype not in ["iv_slope", "ols_slope", "late", "cross"]:
            error_report += (
                f"Estimand type {estimand.esttype} is not valid. "
                "Only iv_slope, ols_slope, late, and cross are valid."
            )
        if estimand.esttype == "cross" and not isinstance(estimand.dz_cross, tuple):
            error_report += (
                f"Estimand type cross requires dz_cross to be a tuple. "
                f"Got {estimand.dz_cross}."
            )
        if estimand.esttype == "late" and not (
            isinstance(estimand.u_lo, float | int)
            and isinstance(estimand.u_hi, float | int)
        ):
            error_report += (
                f"Estimand type late requires u_lo and u_hi to be floats. "
                f"Got {estimand.u_lo} and {estimand.u_hi}."
            )
        if (
            isinstance(estimand.u_lo, float | int)
            and isinstance(
                estimand.u_hi,
                float | int,
            )
            and estimand.esttype == "late"
            and not (0 <= estimand.u_lo < estimand.u_hi <= 1)
        ):
            error_report += (
                f"Estimand type late requires 0 <= u_lo < u_hi <= 1. "
                f"Got {estimand.u_lo} and {estimand.u_hi}."
            )
    return error_report


def _error_report_invalid_basis_func_type(basis_func_type):
    """Return error message if basis_func_type is not valid."""
    error_report = ""
    if basis_func_type not in ["constant", "bernstein"]:
        error_report += (
            f"Basis function type {basis_func_type} is not valid. "
            "Only 'constant' is currently implemented."
        )
    return error_report


def _error_report_missing_basis_func_options(basis_func_type, basis_func_options):
    """Return error message if options are missing for a basis_func_type."""
    error_report = ""
    if basis_func_type == "bernstein" and "k_degree" not in basis_func_options:
        error_report += (
            "Option 'k_degree' is missing for basis function type 'bernstein'."
        )
    return error_report


def _error_report_estimation_data(y_data, z_data, d_data):
    """Return error message if estimation data is not valid."""
    error_report = ""
    for i, data in enumerate([y_data, z_data, d_data]):
        if not isinstance(data, np.ndarray):
            data_names = ["y_data", "z_data", "d_data"]
            error_report += f"Data {data_names[i]} is not of type np.ndarray."
            return error_report

    if len(y_data) != len(z_data) or len(y_data) != len(d_data):
        error_report += (
            f"Data lengths are not equal. "
            f"Lengths are {len(y_data)}, {len(z_data)}, {len(d_data)}."
        )

    if not np.all(np.logical_or(d_data == 0, d_data == 1)):
        error_report += f"Data d_data is not binary. Got values {np.unique(d_data)}."

    return error_report


def _error_report_tolerance(tolerance):
    """Return error message if tolerance is not valid."""
    error_report = ""
    if tolerance is None:
        return error_report

    if not isinstance(tolerance, int | float):
        return f"Tolerance {tolerance} is not a number."
    if tolerance <= 0:
        error_report += f"Tolerance {tolerance} is not positive."
    return error_report


def _error_report_method(method):
    """Return error message if method is not valid."""
    error_report = ""
    if method not in ["highs", "copt"]:
        error_report += (
            f"Method {method} is not valid. Only 'highs' and 'copt' are implemented."
        )
    return error_report


def _error_report_u_partition(u_partition):
    """Return error message if u_partition is not valid."""
    error_report = ""
    if u_partition is None:
        return error_report

    if not isinstance(u_partition, np.ndarray):
        return f"u_partition {u_partition} is not of type np.ndarray."

    if not np.all(np.diff(u_partition) > 0):
        error_report += f"u_partition {u_partition} is not strictly increasing."

    if u_partition[0] < 0 or u_partition[-1] > 1:
        error_report += f"u_partition {u_partition} not between 0 and 1."

    return error_report


def _error_report_basis_funcs(basis_funcs):
    """Return error message if basis_funcs is not valid."""
    error_report = ""

    supported_bfuncs = ["constant", "bernstein"]

    # Check if list of dict not empty
    if not basis_funcs:
        error_report += "Basis functions list is empty."
        return error_report

    for i, basis_func in enumerate(basis_funcs):
        if not isinstance(basis_func, dict):
            error_report += (
                f"Basis function {basis_func} at index {i} is not of type dict."
            )
        else:
            if basis_func["type"] not in supported_bfuncs:
                error_report += (
                    f"Basis func type {basis_func['type']} at index {i} is invalid. "
                    f"Only {supported_bfuncs} is currently implemented."
                )
            if basis_func["type"] == "constant":
                if not (
                    isinstance(basis_func["u_lo"], float | int)
                    and isinstance(basis_func["u_hi"], float | int)
                ):
                    error_report += (
                        f"Basis func {basis_func} at index {i} requires u_lo and u_hi "
                        f"to be float. Got {basis_func['u_lo']}, {basis_func['u_hi']}."
                    )
                if not (0 <= basis_func["u_lo"] < basis_func["u_hi"] <= 1):
                    error_report += (
                        f"Basis function {basis_func} at index {i} requires "
                        f"0 <= u_lo < u_hi <= 1. "
                        f"Got {basis_func['u_lo']} and {basis_func['u_hi']}."
                    )
            elif basis_func["type"] == "bernstein" and not (
                isinstance(basis_func["func"], BPoly | Bern)
            ):
                error_report += f"Basis function {basis_func} at index {i} is not"
                "of type BPoly or Bern."
    return error_report


def _error_report_mtr_function(mtr_function):
    """Return error message if mtr_function is not valid."""
    error_report = ""
    if not callable(mtr_function):
        return f"MTR function {mtr_function} is not callable."

    if mtr_function.__code__.co_argcount != 1:
        error_report += (
            f"MTR function {mtr_function} does not have exactly one argument."
        )
    return error_report


def _error_report_instrument(instrument: Instrument):
    """Return error message if instrument is not valid."""
    error_report = ""
    if not isinstance(instrument, Instrument):
        return f"Instrument {instrument} is not of type Instrument."

    for attr in ["support", "pmf", "pscores"]:
        if not isinstance(getattr(instrument, attr), np.ndarray):
            error_report += f"Instrument attribute {attr} is not of type np.ndarray."

    for attr in ["pmf", "pscores"]:
        if not np.all(getattr(instrument, attr) >= 0):
            error_report += f"Instrument attribute {attr} contains negative values."
    if not np.isclose(np.sum(instrument.pmf), 1):
        error_report += "Instrument attribute pmf does not sum to 1."

    return error_report


def _error_report_shape_constraints(shape_constraints: tuple[str, str] | None) -> str:
    """Return error message if shape_constraints is not valid."""
    error_report = ""
    if shape_constraints is None:
        return error_report

    if not isinstance(shape_constraints, tuple):
        return f"Shape constraints {shape_constraints} is not of type tuple."

    n_shape_constraints = 2

    if len(shape_constraints) != n_shape_constraints:
        error_report += (
            f"Shape constraints {shape_constraints} does not have exactly two elements."
        )

    valid_constraints = ["decreasing", "increasing"]

    for constraint in shape_constraints:
        if constraint not in valid_constraints:
            error_report += (
                f"Shape constraint {constraint} is not valid. "
                f"Only {valid_constraints} are valid."
            )
    return error_report


def generate_bernstein_basis_funcs(k: int) -> list[dict]:
    """Generate list containing basis functions of kth-oder Bernstein polynomial.

    Arguments:
        k: The order of the Bernstein polynomial.

    Returns:
        A list of dictionaries containing the basis functions.

    """
    basis_funcs = []

    for i in range(k + 1):
        _c = np.zeros(k + 1).reshape(-1, 1)
        _c[i] = 1

        basis_func = {
            "type": "bernstein",
            "func": Bern(n=k, coefs=_c),
        }
        basis_funcs.append(basis_func)

    return basis_funcs

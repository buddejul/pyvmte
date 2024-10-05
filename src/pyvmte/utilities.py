"""Utilities used in various parts of the project."""

import contextlib
import math
import os
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from scipy.interpolate import BPoly  # type: ignore[import-untyped]

from pyvmte.classes import Bern, Estimand, Instrument, PyvmteResult


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


def estimate_late(y: np.ndarray, d: np.ndarray, z: np.ndarray):
    """Estimate a LATE using the Wald estimator."""
    yz1 = y[z == 1].mean()
    yz0 = y[z == 0].mean()

    dz1 = d[z == 1].mean()
    dz0 = d[z == 0].mean()

    return (yz1 - yz0) / (dz1 - dz0)


def bern_bas(n, v, x):
    """Bernstein polynomial basis of degree n and index v at point x."""
    return math.comb(n, v) * x**v * (1 - x) ** (n - v)


def simulate_data_from_paper_dgp(sample_size, rng: np.random.Generator):
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


def simulate_data_from_simple_model_dgp(
    sample_size,
    rng: np.random.Generator,
    dgp_params: dict,
):
    """Simulate data for the simple model.

    Args:
        sample_size (int): The number of observations in the sample.
        rng (np.random.Generator): The random number generator.
        dgp_params (dict): The parameters of the data generating process.

    Returns:
        dict: A dictionary containing the simulated data.

    """
    data = pd.DataFrame()

    pscore_lo = 0.4
    pscore_hi = 0.6

    support = np.array([0, 1])
    pmf = np.array([0.5, 0.5])
    pscores = np.array([pscore_lo, pscore_hi])

    choices = np.hstack([support.reshape(-1, 1), pscores.reshape(-1, 1)])

    # Draw random indices
    idx = rng.choice(support, size=sample_size, p=pmf)

    data = choices[idx]

    z = np.array(data[:, 0], dtype=int)
    pscores = data[:, 1]

    u = rng.uniform(size=sample_size)
    d = u < pscores

    def _at(u: float) -> bool | np.ndarray:
        return np.where(u <= pscore_lo, 1, 0)

    def _c(u: float) -> bool | np.ndarray:
        return np.where((pscore_lo <= u) & (u < pscore_hi), 1, 0)

    def _nt(u: float) -> bool | np.ndarray:
        return np.where(u >= pscore_hi, 1, 0)

    y = np.empty(sample_size)
    idx = d == 0

    def _m0(u, y0_at, y0_c, y0_nt):
        return y0_at * _at(u) + y0_c * _c(u) + y0_nt * _nt(u)

    def _m1(u, y1_at, y1_c, y1_nt):
        return y1_at * _at(u) + y1_c * _c(u) + y1_nt * _nt(u)

    y0_at = dgp_params["y0_at"]
    y0_c = dgp_params["y0_c"]
    y0_nt = dgp_params["y0_nt"]

    y1_at = dgp_params["y1_at"]
    y1_c = dgp_params["y1_c"]
    y1_nt = dgp_params["y1_nt"]

    y[idx] = _m0(u[idx], y0_at, y0_c, y0_nt) + rng.normal(size=np.sum(idx))

    y[~idx] = _m1(u[~idx], y1_at, y1_c, y1_nt) + rng.normal(size=np.sum(~idx))

    return {"z": z, "d": d, "y": y, "u": u}


@contextlib.contextmanager
def suppress_print():
    """Suppress print statements in context."""
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # noqa: PTH123
        yield


def _error_report_estimand(estimand: Estimand, mode: str):
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
        if (
            estimand.esttype == "late"
            and not (
                isinstance(estimand.u_lo, float | int)
                and isinstance(estimand.u_hi, float | int)
            )
            and mode == "identification"
        ):
            error_report += (
                f"Estimand type late requires u_lo and u_hi to be float or ints. "
                f"Got {estimand.u_lo} and {estimand.u_hi}."
            )
        if (
            estimand.esttype == "late"
            and not (
                isinstance(estimand.u_lo, float | int | None)
                and isinstance(estimand.u_hi, float | int | None)
            )
            and mode == "estimation"
        ):
            error_report += (
                f"Estimand type late requires u_lo and u_hi to be float, ints or None. "
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
    if basis_func_type == "bernstein" and (
        basis_func_options is None or "k_degree" not in basis_func_options
    ):
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


def _error_report_mte_monotone(mte_monotone: str | None) -> str:
    """Return error message if mte_monotone argument is not valid."""
    error_report = ""
    if mte_monotone is None:
        return error_report

    _valid_args = ["increasing", "decreasing"]

    if mte_monotone not in _valid_args:
        error_report += (
            f"MTE monotonicity {mte_monotone} is not valid. "
            f"Only {_valid_args} are valid options."
        )
    return error_report


def _error_report_monotone_response(monotone_response) -> str:
    """Return error message if monotone_response argument is not valid."""
    error_report = ""
    if monotone_response is None:
        return error_report

    _valid_arguments = ["positive", "negative"]

    if monotone_response not in _valid_arguments:
        error_report += (
            f"Monotone response {monotone_response} is not valid. "
            f"Only {_valid_arguments} are valid arguments."
        )
    return error_report


def generate_constant_splines_basis_funcs(u_partition: np.ndarray) -> list[dict]:
    """Generate list with constant spline basis functions corresponding to partition.

    Arguments:
        u_partition: The partition of the unit interval.

    Returns:
        A list of dictionaries containing the basis functions.

    """
    basis_funcs = []

    def _constant_spline(u, lo, hi):
        return np.where((lo <= u) & (u < hi), 1, 0)

    # Make vectorized version of the function

    for i in range(len(u_partition) - 1):
        basis_func = {
            "type": "constant",
            "u_lo": u_partition[i],
            "u_hi": u_partition[i + 1],
            "func": partial(_constant_spline, lo=u_partition[i], hi=u_partition[i + 1]),
        }
        basis_funcs.append(basis_func)

    return basis_funcs


def plot_solution(
    res: PyvmteResult,
    lower_or_upper: str,
    *,
    add_weights: bool = False,
    add_mte: bool = False,
) -> go.Figure:
    """Plot the MTR functions corresponding to the lower or upper bound."""
    num_gridpoints = 1_000

    lines_to_plot = ["lower", "upper"] if lower_or_upper == "both" else [lower_or_upper]

    # ----------------------------------------------------------------------------------
    # Prepare results
    # ----------------------------------------------------------------------------------
    mtr_by_line = {}

    def _mtr_from_bfunc(u: float, coefs: np.ndarray, bfuncs: list[Callable]):
        return np.sum([c * bf(u) for c, bf in zip(coefs, bfuncs, strict=True)], axis=0)

    for line in lines_to_plot:
        optres = res.lower_optres if line == "lower" else res.upper_optres

        n_bfuncs = len(res.basis_funcs)

        # The first n_bfuncs entries of the coefficients correspond to the d == 1
        coefs_d0 = optres.x[:n_bfuncs]
        coefs_d1 = optres.x[n_bfuncs:]

        # Create the mtr functions for d == 0 and d == 1 by multiplying the coefs with
        # the basis functions
        _bfuncs = [bf["func"] for bf in res.basis_funcs]

        mtr_by_line[line] = {
            "d0": partial(_mtr_from_bfunc, coefs=coefs_d0, bfuncs=_bfuncs),
            "d1": partial(_mtr_from_bfunc, coefs=coefs_d1, bfuncs=_bfuncs),
        }

    u_grid = np.linspace(0, 1, num_gridpoints, endpoint=False)

    # ----------------------------------------------------------------------------------
    # Plotting parameters
    # ----------------------------------------------------------------------------------

    line_style = {
        "dash": {
            "upper": "solid",
            "lower": "dash" if lower_or_upper == "both" else "solid",
        },
    }

    # ----------------------------------------------------------------------------------
    # Figure: MTR functions
    # ----------------------------------------------------------------------------------
    n_cols = 2 if lower_or_upper == "both" else 1

    subplot_titles = (
        (f"Lower Bound: {res.lower_bound:.3f}", f"Upper Bound: {res.upper_bound:.3f}")
        if lower_or_upper == "both"
        else ("", "")
    )

    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles)

    _col_counter = 1
    for line in lines_to_plot:
        fig.add_trace(
            go.Scatter(
                x=u_grid,
                y=mtr_by_line[line]["d0"](u_grid),
                mode="lines",
                name="MTR d = 0",
                line={"color": "blue", "dash": line_style["dash"][line]},
                legendgroup=f"{line}",
                legendgrouptitle={"text": f"{line} bound"},
            ),
            row=1,
            col=_col_counter,
        )

        fig.add_trace(
            go.Scatter(
                x=u_grid,
                y=mtr_by_line[line]["d1"](u_grid),
                mode="lines",
                name="MTR d = 1",
                line={"color": "red", "dash": line_style["dash"][line]},
                legendgroup=f"{line}",
            ),
            row=1,
            col=_col_counter,
        )

        if add_mte is True:
            fig.add_trace(
                go.Scatter(
                    x=u_grid,
                    y=mtr_by_line[line]["d1"](u_grid) - mtr_by_line[line]["d0"](u_grid),
                    mode="lines",
                    name="MTE",
                    line={"color": "green", "dash": line_style["dash"][line]},
                    legendgroup=f"{line}",
                ),
                row=1,
                col=_col_counter,
            )

        _col_counter += 1

    if lower_or_upper == "both":
        _sub = ""
    else:
        _bound = res.lower_bound if lower_or_upper == "lower" else res.upper_bound
        _sub = (
            f"<br><sup>{lower_or_upper.capitalize()} "
            f"bound: {_bound:.3f} </sup></br>"
        )

    fig.update_layout(
        title=(f"MTR functions for {lower_or_upper} bound(s){_sub}"),
        xaxis_title="u",
        yaxis_title="MTR",
    )

    if len(lines_to_plot) == 1:
        return fig

    if add_weights is False:
        return fig

    # ----------------------------------------------------------------------------------
    # Add weights (if requested)
    # ----------------------------------------------------------------------------------
    # TODO(@buddejul): Think about this - these are not necessarily the weights for the
    # LP, which also are a function of the basis functions?
    # In case of Bernstein polynomials: Would we need to resolve using constant splines?

    # Weights for target parameter (choice variables in linear program)
    _weights = res.lp_inputs["c"]
    return None

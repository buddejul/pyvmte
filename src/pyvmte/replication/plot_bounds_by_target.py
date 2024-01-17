import numpy as np
import pandas as pd  # type: ignore

from pyvmte.config import SETUP_FIG5, BLD
from pyvmte.utilities import load_paper_dgp
from pyvmte.identification import identification
from pyvmte.estimation.estimation import _generate_basis_funcs, _compute_u_partition

import plotly.graph_objects as go  # type: ignore


def plot_bounds_by_target(data: pd.DataFrame) -> go.Figure:
    """Plot bounds by target based on dataframe."""
    # Plot lines for lower and upper bounds from data_bounds
    data_plot = data[data["u_hi"] > 0.45]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data_plot["u_hi"],
            y=data_plot["upper_bound"],
            name="Upper Bound",
            line_color="green",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=data_plot["u_hi"],
            y=data_plot["lower_bound"],
            name="Lower Bound",
            line_color="blue",
        ),
    )

    fig.update_layout(
        title="Bounds by Target",
        xaxis_title="Target",
        yaxis_title="Bounds",
        legend_title="Bound",
    )

    return fig


def create_bounds_by_target_df(setup: dict, n_gridpoints: int) -> pd.DataFrame:
    """Returns dataframe of bounds for different targets."""

    range_of_targets = np.linspace(0.35, 1, n_gridpoints)
    upper_bounds = np.zeros(len(range_of_targets))
    lower_bounds = np.zeros(len(range_of_targets))

    for i, u_hi in enumerate(range_of_targets):
        target = setup["target"]
        target["u_hi"] = u_hi

        u_partition = _compute_u_partition(target, setup["instrument"]["pscore_z"])
        bfuncs = _generate_basis_funcs("constant", u_partition)
        bounds = identification(
            target=target,
            identified_estimands=setup["identified_estimands"],
            basis_funcs=bfuncs,
            m0_dgp=setup["m0"],
            m1_dgp=setup["m1"],
            instrument=setup["instrument"],
        )

        upper_bounds[i] = bounds["upper_bound"]
        lower_bounds[i] = bounds["lower_bound"]

    bounds_by_target = pd.DataFrame(
        {
            "u_hi": range_of_targets,
            "upper_bound": upper_bounds,
            "lower_bound": lower_bounds,
            "target": "late",
        }
    )

    return bounds_by_target

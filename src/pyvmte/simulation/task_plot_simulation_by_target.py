"""Task for plotting simulation by target."""
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from pytask import Product

from pyvmte.config import (
    BLD,
    DGP_MST,
    IV_MST,
    SETUP_FIG5,
    SIMULATION_RESULTS_DIR,
)
from pyvmte.config_mc_by_target import MONTE_CARLO_BY_TARGET
from pyvmte.simulation.create_bounds_by_target import create_bounds_by_target_df

_DEPENDENCIES = {
    u_hi: BLD
    / "python"
    / "data"
    / "by_target"
    / Path(f"sim_results_figure5_u_hi_{u_hi}.pkl")
    for u_hi in MONTE_CARLO_BY_TARGET.u_hi_range  # type: ignore
}


def task_plot_simulation_by_target(
    u_hi_range: np.ndarray = MONTE_CARLO_BY_TARGET.u_hi_range,  # type: ignore
    path_to_data: dict[str, Path] = _DEPENDENCIES,
    path_to_plot: Annotated[Path, Product] = BLD
    / "python"
    / "figures"
    / "simulation_results_by_target.png",
) -> None:
    """Plot simulation by target."""
    files = list(path_to_data.values())

    dfs = [
        pd.read_pickle(SIMULATION_RESULTS_DIR / "by_target" / f).assign(
            filename=f,
        )
        for f in files
    ]
    df_estimates = pd.concat(dfs, ignore_index=True)

    # From the filename column extract the string between "u_hi" and ".pkl"
    df_estimates["u_hi"] = (
        df_estimates["filename"].astype(str).str.extract(r"u_hi_(.*)\.pkl")
    )
    df_estimates["u_hi"] = df_estimates["u_hi"].astype(float)

    df_estimates = df_estimates[df_estimates["u_hi"].isin(u_hi_range)]

    fig = go.Figure()

    df_mean = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].mean()
    df_mean = df_mean.reset_index()

    df_05 = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].quantile(0.05)
    df_05 = df_05.reset_index()

    df_95 = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].quantile(0.95)
    df_95 = df_95.reset_index()

    if SETUP_FIG5.target.u_lo is None or SETUP_FIG5.target.u_hi is None:
        msg = "Target parameter must have both u_lo and u_hi."
        raise ValueError(msg)

    df_identified = create_bounds_by_target_df(
        setup=SETUP_FIG5,
        instrument=IV_MST,
        m0=DGP_MST.m0,
        m1=DGP_MST.m1,
        n_gridpoints=100,
        u_hi_min=SETUP_FIG5.target.u_lo + 0.01,
        u_hi_max=SETUP_FIG5.target.u_hi,
    )

    df_identified = df_identified[df_identified["u_hi"] <= max(u_hi_range)]

    data_to_plot = {
        "mean": {"data": df_mean, "name": "Mean", "transp": 1.0, "line_dash": "solid"},
        "5th": {
            "data": df_05,
            "name": "5th Percentile",
            "transp": 0.5,
            "line_dash": "solid",
        },
        "95th": {
            "data": df_95,
            "name": "95th Percentile",
            "transp": 0.5,
            "line_dash": "solid",
        },
        "identified": {
            "data": df_identified,
            "name": "True Bound",
            "transp": 1.0,
            "line_dash": "dot",
        },
    }

    bounds_to_plot = {
        "lower_bound": {"color": "blue"},
        "upper_bound": {"color": "green"},
    }

    for data in data_to_plot.values():
        for key, values in bounds_to_plot.items():
            fig.add_trace(
                go.Scatter(
                    x=data["data"]["u_hi"],
                    y=data["data"][key],
                    name=data["name"],
                    mode="lines",
                    line_color=values["color"],
                    line_dash=data["line_dash"],
                    opacity=data["transp"],
                    legendgroup=key,
                    legendgrouptitle_text=key.replace("_", " ").title(),
                ),
            )

    fig.update_layout(
        title_text=(
            "Bound Estimates by Target Parameter"
            "<br><sup>Target LATE(0.35, x), Sharp-Nonparametric Bounds (Figure 5 MST)</sup>"  # noqa: E501
        ),
    )
    fig.update_xaxes(title_text="Upper Bound of Target Parameter")
    fig.update_yaxes(title_text="Bound Estimate")

    fig.add_annotation(
        text=(
            f"Sample Size = {MONTE_CARLO_BY_TARGET.sample_size}, "
            f"Repetitions = {MONTE_CARLO_BY_TARGET.repetitions}"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.2,
        font={"size": 10},
        showarrow=False,
    )

    pio.write_image(fig, path_to_plot)

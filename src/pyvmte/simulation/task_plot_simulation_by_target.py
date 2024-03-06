"""Task for plotting simulation by target."""
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from pytask import Product

from pyvmte.config import (
    BLD,
    DGP_MST,
    IV_MST,
    MONTE_CARLO_BY_TARGET,
    SETUP_FIG5,
    SIMULATION_RESULTS_DIR,
)
from pyvmte.replication.plot_bounds_by_target import create_bounds_by_target_df


class _Arguments(NamedTuple):
    path_to_plot: Path


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
    df_estimates.head()

    # From the filename column extract the string between "u_hi" and ".pkl"
    df_estimates["u_hi"] = (
        df_estimates["filename"].astype(str).str.extract(r"u_hi_(.*)\.pkl")
    )
    df_estimates["u_hi"] = df_estimates["u_hi"].astype(float)

    df_estimates = df_estimates[df_estimates["u_hi"].isin(u_hi_range)]

    fig = go.Figure()

    df_mean = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].mean()
    df_mean = df_mean.reset_index()

    df_10 = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].quantile(0.1)
    df_10 = df_10.reset_index()

    df_90 = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].quantile(0.9)
    df_90 = df_90.reset_index()

    df_identified = create_bounds_by_target_df(
        setup=SETUP_FIG5,
        instrument=IV_MST,
        m0=DGP_MST.m0,
        m1=DGP_MST.m1,
        n_gridpoints=100,
    )

    # Simulation results
    for df in [df_mean, df_10, df_90]:
        if df is df_mean:
            name = "Mean"
        elif df is df_10:
            name = "10th Percentile"
        elif df is df_90:
            name = "90th Percentile"

        transp = 1.0 if df is df_mean else 0.5

        fig.add_trace(
            go.Scatter(
                x=df["u_hi"],
                y=df["upper_bound"],
                name=name,
                mode="lines",
                line_color="green",
                opacity=transp,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=df["u_hi"],
                y=df["lower_bound"],
                name=name,
                mode="lines",
                line_color="blue",
                opacity=transp,
            ),
        )

    # (True) Identified bounds
    fig.add_trace(
        go.Scatter(
            x=df_identified["u_hi"],
            y=df_identified["upper_bound"],
            name=name,
            mode="lines",
            line={"color": "green", "dash": "dot"},
            opacity=transp,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df_identified["u_hi"],
            y=df_identified["lower_bound"],
            name=name,
            mode="lines",
            line={"color": "blue", "dash": "dot"},
            opacity=transp,
        ),
    )

    # Remove legend
    fig.update_layout(showlegend=False)

    # Update title
    fig.update_layout(
        title_text="Sharp Non-Parametric Bound Estimates by Target Parameter",
    )
    fig.update_xaxes(title_text="Upper Bound of Target Parameter")
    fig.update_yaxes(title_text="Bound Estimate")

    pio.write_image(fig, path_to_plot)

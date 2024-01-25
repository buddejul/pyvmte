from typing import Annotated, NamedTuple
from pathlib import Path

import os

import pandas as pd  # type: ignore
import numpy as np

import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore

from pytask import Product

from pyvmte.config import BLD, U_HI_RANGE, SIMULATION_RESULTS_DIR


class _Arguments(NamedTuple):
    path_to_plot: Path


def task_plot_simulation_by_target(
    # path_to_data: Path = SIMULATION_RESULTS_DIR / "by_target",
    range=U_HI_RANGE,
    path_to_plot: Annotated[Path, Product] = BLD
    / "python"
    / "figures"
    / "simulation_results_by_target.png",
) -> None:
    files = [
        f
        for f in os.listdir(SIMULATION_RESULTS_DIR / "by_target")
        if Path.is_file(SIMULATION_RESULTS_DIR / "by_target" / f)
    ]

    dfs = [
        pd.read_pickle(SIMULATION_RESULTS_DIR / "by_target" / f).assign(filename=f)
        for f in files
    ]  # noqa: S301
    df_estimates = pd.concat(dfs, ignore_index=True)
    df_estimates.head()

    # From the filename column extract the string between "u_hi" and ".pkl"
    df_estimates["u_hi"] = df_estimates["filename"].str.extract(r"u_hi_(.*)\.pkl")
    df_estimates["u_hi"] = df_estimates["u_hi"].astype(float)

    df_estimates = df_estimates[df_estimates["u_hi"].isin(range)]

    fig = go.Figure()

    df_mean = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].mean()
    df_mean = df_mean.reset_index()

    df_10 = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].quantile(0.1)
    df_10 = df_10.reset_index()

    df_90 = df_estimates.groupby("u_hi")[["lower_bound", "upper_bound"]].quantile(0.9)
    df_90 = df_90.reset_index()

    for df in [df_mean, df_10, df_90]:
        if df is df_mean:
            name = "Mean"
        elif df is df_10:
            name = "10th Percentile"
        elif df is df_90:
            name = "90th Percentile"

        if df is df_mean:
            transp = 1.0
        else:
            transp = 0.5

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

    # Remove legend
    fig.update_layout(showlegend=False)

    # Update title
    fig.update_layout(
        title_text="Sharp Non-Parametric Bound Estimates by Target Parameter"
    )
    fig.update_xaxes(title_text="Upper Bound of Target Parameter")
    fig.update_yaxes(title_text="Bound Estimate")

    pio.write_image(fig, path_to_plot)

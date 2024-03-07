"""Task for plotting simulation by sample size."""
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from pytask import Product

from pyvmte.config import (
    BLD,
    SAMPLE_SIZES,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    SIMULATION_RESULTS_DIR,
)


class _Arguments(NamedTuple):
    path_to_plot: Path


figures = [SETUP_FIG2, SETUP_FIG3, SETUP_FIG5]

# TODO implement for all figures
_DEPENDENCIES = {
    sample_size: BLD
    / "python"
    / "data"
    / "by_sample_size"
    / Path(f"sim_results_figure5_sample_size_{sample_size}.pkl")
    for sample_size in SAMPLE_SIZES
}


def task_plot_simulation_by_sample_size(
    sample_sizes: np.ndarray = SAMPLE_SIZES,  # type: ignore
    path_to_data: dict[int, Path] = _DEPENDENCIES,
    path_to_plot: Annotated[Path, Product] = BLD
    / "python"
    / "figures"
    / "simulation_results_by_sample_size_lower_bound.png",
) -> None:
    """Plot simulation by target."""
    files = list(path_to_data.values())

    dfs = [
        pd.read_pickle(SIMULATION_RESULTS_DIR / "by_sample_size" / f).assign(
            filename=f,
        )
        for f in files
    ]
    df_estimates = pd.concat(dfs, ignore_index=True)
    df_estimates.head()

    # From the filename column extract the string between "sample_size" and ".pkl"
    df_estimates["sample_size"] = (
        df_estimates["filename"].astype(str).str.extract(r"sample_size_(.*)\.pkl")
    )
    df_estimates["sample_size"] = df_estimates["sample_size"].astype(int)

    df_estimates = df_estimates[df_estimates["sample_size"].isin(sample_sizes)]

    # Keep "upper_bound", "lower_bound", "sample_size"
    df_estimates = df_estimates[["upper_bound", "lower_bound", "sample_size"]]

    fig = go.Figure()

    params = {
        size: {
            "name": f"N = {size}",
            "color": f"#008{100 + int(np.floor(i * 899 / len(sample_sizes) ))}",
        }
        for i, size in enumerate(sample_sizes)
    }

    for sample_size, param in params.items():
        fig.add_trace(
            go.Histogram(
                x=df_estimates[df_estimates["sample_size"] == sample_size][
                    "lower_bound"
                ],
                name=param["name"],
                marker_color=param["color"],
            ),
        )

    fig.update_layout(
        barmode="overlay",
        legend_title_text="Sample Size",
        title_text=(
            "Lower Bound Estimates by Sample Size"
            "<br><sup>Sharp Non-parametric Bounds (Figure 5)</sup>"
        ),
        xaxis_title_text="Lower Bound",
        yaxis_title_text="Count",
    )
    fig.update_traces(opacity=0.75)

    fig.add_vline(SETUP_FIG5.lower_bound, annotation={"text": "True Bound"})

    pio.write_image(fig, path_to_plot)

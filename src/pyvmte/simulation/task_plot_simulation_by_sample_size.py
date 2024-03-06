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
    / "simulation_results_by_sample_size.png",
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

    # From the filename column extract the string between the last "_" and ".pkl"
    df_estimates["u_hi"] = (
        df_estimates["filename"].astype(str).str.extract(r"_([^_]*)\.pkl")
    )
    df_estimates["u_hi"] = df_estimates["u_hi"].astype(float)

    df_estimates = df_estimates[df_estimates["u_hi"].isin(sample_sizes)]

    fig = go.Figure()

    pio.write_image(fig, path_to_plot)

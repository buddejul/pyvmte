"""Task for plotting simulation by sample size."""
from itertools import product
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
from pytask import Product, task

from pyvmte.config import (
    BLD,
    SETUP_FIG2,
    SETUP_FIG3,
    SETUP_FIG5,
    SIMULATION_RESULTS_DIR,
)
from pyvmte.config_mc_by_size import MC_SAMPLE_SIZES

figures = [
    ("figure2", SETUP_FIG2, "IV-Slope Identified (Figure 2 MST)"),
    ("figure3", SETUP_FIG3, "IV-Slope and OLS Identified (Figure 3 MST)"),
    ("figure5", SETUP_FIG5, "Sharp Non-parametric Bounds (Figure 5 MST)"),
]

_DEPENDENCIES = {
    f"{sample_size}_{fig}": SIMULATION_RESULTS_DIR
    / "by_sample_size"
    / Path(f"sim_results_{fig[0]}_sample_size_{sample_size}.pkl")
    for sample_size, fig in product(MC_SAMPLE_SIZES, figures)
}


class _Arguments(NamedTuple):
    path_to_plot: Path
    bound: str
    true_bound: float
    fig_name: str
    fig_subtitle: str


ID_TO_KWARGS = {
    f"{bound}_{fig[0]}": _Arguments(
        path_to_plot=BLD
        / "python"
        / "figures"
        / f"simulation_results_by_sample_size_{bound}_{fig[0]}.png",
        bound=bound,
        true_bound=getattr(fig[1], bound),
        fig_name=fig[0],
        fig_subtitle=fig[2],
    )
    for bound, fig in product(["lower_bound", "upper_bound"], figures)
}

for _id, kwargs in ID_TO_KWARGS.items():

    @task(id=_id, kwargs=kwargs)  # type: ignore
    def task_plot_simulation_by_sample_size(
        path_to_plot: Annotated[Path, Product],
        bound: str,
        true_bound: float,
        fig_name: str,
        fig_subtitle: str,
        sample_sizes: np.ndarray = MC_SAMPLE_SIZES,  # type: ignore
        path_to_data: dict[str, Path] = _DEPENDENCIES,
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

        df_estimates["sample_size"] = (
            df_estimates["filename"].astype(str).str.extract(r"sample_size_(.*)\.pkl")
        )
        df_estimates["sample_size"] = df_estimates["sample_size"].astype(int)
        df_estimates = df_estimates[df_estimates["sample_size"].isin(sample_sizes)]

        df_estimates["figure"] = (
            df_estimates["filename"]
            .astype(str)
            .str.extract(r"results_(.*)_sample_size")
        )

        df_estimates = df_estimates[df_estimates["figure"] == fig_name]

        df_estimates = df_estimates[[bound, "sample_size"]]

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
                    x=df_estimates[df_estimates["sample_size"] == sample_size][bound],
                    name=param["name"],
                    marker_color=param["color"],
                ),
            )

        bound_title = bound.replace("_", " ").title()

        fig.update_layout(
            barmode="overlay",
            legend_title_text="Sample Size",
            title_text=(
                f"{bound_title} Estimates by Sample Size"
                f"<br><sup>{fig_subtitle}</sup>"
            ),
            xaxis_title_text=f"{bound_title} Estimate",
            yaxis_title_text="Count",
        )
        fig.update_traces(opacity=0.75)

        fig.add_vline(true_bound, annotation={"text": "True Bound"})

        pio.write_image(fig, path_to_plot)

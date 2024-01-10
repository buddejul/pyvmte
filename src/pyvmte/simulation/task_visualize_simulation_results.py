from pyvmte.config import BLD
from pyvmte.simulation.simulation_plotting import plot_upper_and_lower_bounds
from typing import Annotated, NamedTuple
from pathlib import Path

from pytask import Product
from pytask import task

import pandas as pd  # type: ignore
import numpy as np
import plotly.io as pio  # type: ignore


class _Arguments(NamedTuple):
    path_to_data: Path
    path_to_output: Annotated[Path, Product]


ID_TO_KWARGS_PLOT = {
    "figure2": _Arguments(
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure2.pkl"),
        path_to_output=BLD / "python" / "figures" / Path("sim_plot_figure2.png"),
    ),
    "figure3": _Arguments(
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure3.pkl"),
        path_to_output=BLD / "python" / "figures" / Path("sim_plot_figure3.png"),
    ),
    "figure5": _Arguments(
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure5.pkl"),
        path_to_output=BLD / "python" / "figures" / Path("sim_plot_figure5.png"),
    ),
}

ID_TO_KWARGS_TABLE = {
    "figure2": _Arguments(
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure2.pkl"),
        path_to_output=BLD / "python" / "tables" / Path("sim_plot_table2.tex"),
    ),
    "figure3": _Arguments(
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure3.pkl"),
        path_to_output=BLD / "python" / "tables" / Path("sim_plot_table3.tex"),
    ),
    "figure5": _Arguments(
        path_to_data=BLD / "python" / "data" / Path("sim_results_figure5.pkl"),
        path_to_output=BLD / "python" / "tables" / Path("sim_plot_table5.tex"),
    ),
}

for id_, kwargs in ID_TO_KWARGS_PLOT.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore
    def task_plot_monte_carlo_upper_lower_bound(
        path_to_data: Path,
        path_to_output: Annotated[Path, Product],
    ) -> None:
        df = pd.read_pickle(path_to_data)
        fig = plot_upper_and_lower_bounds(df)

        pio.write_image(fig, path_to_output)


for id_, kwargs in ID_TO_KWARGS_TABLE.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore
    def task_create_table_by_mode(
        path_to_data: Path,
        path_to_output: Annotated[Path, Product],
    ) -> None:
        df = pd.read_pickle(path_to_data)

        # .describe() and to_latex but only for mean, std, min, max
        df = df.describe().T[["mean", "std", "min", "max"]]
        df = df.round(3)

        table = df.to_latex()

        with open(path_to_output, "w") as f:
            f.write(table.replace("%", "\\%").replace("_", "\\_"))

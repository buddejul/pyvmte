from pyvmte.config import BLD
from pyvmte.simulation.simulation_plotting import plot_upper_and_lower_bounds
from typing import Annotated, NamedTuple
from pathlib import Path

from pytask import Product
from pytask import task

import pandas as pd
import numpy as np
import plotly.io as pio


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

    @task(id=id_, kwargs=kwargs)
    def task_plot_monte_carlo_upper_lower_bound(
        path_to_data: Path,
        path_to_output: Annotated[Path, Product],
    ) -> None:
        df = pd.read_pickle(path_to_data)
        fig = plot_upper_and_lower_bounds(df)

        pio.write_image(fig, path_to_output)


for id_, kwargs in ID_TO_KWARGS_TABLE.items():

    @task(id=id_, kwargs=kwargs)
    def task_create_table_by_mode(
        path_to_data: Path,
        path_to_output: Annotated[Path, Product],
    ) -> None:
        df = pd.read_pickle(path_to_data)

        df["left_mode"] = np.where(df["lower_bound"] < -0.6, 1, 0)

        table = df.groupby("left_mode").describe().T.to_latex()

        with open(path_to_output, "w") as f:
            f.write(table.replace("%", "\\%").replace("_", "\\_"))

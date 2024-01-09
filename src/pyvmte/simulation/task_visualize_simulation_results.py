from pyvmte.config import BLD
from pyvmte.simulation.simulation_plotting import plot_upper_and_lower_bounds

import pandas as pd
import numpy as np
import plotly.io as pio


def task_plot_monte_carlo_upper_lower_bound(
    depends_on=BLD / "python" / "data" / "monte_carlo_simulation_results.pkl",
    produces=BLD / "python" / "figures" / "monte_carlo_upper_lower_bound.png",
):
    df = pd.read_pickle(depends_on)
    fig = plot_upper_and_lower_bounds(df)

    pio.write_image(fig, produces)


def task_create_table_by_mode(
    depends_on=BLD / "python" / "data" / "monte_carlo_simulation_results.pkl",
    produces=BLD / "python" / "tables" / "table_by_mode.tex",
):
    df = pd.read_pickle(depends_on)

    df["left_mode"] = np.where(df["lower_bound"] < -0.6, 1, 0)

    table = df.groupby("left_mode").describe().T.to_latex()

    with open(produces, "w") as f:
        f.write(table.replace("%", "\\%").replace("_", "\\_"))

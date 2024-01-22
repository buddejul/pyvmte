from typing import Annotated
from pathlib import Path
from pytask import Product
import pytask

from pyvmte.replication.plot_bounds_by_target import (
    create_bounds_by_target_df,
    plot_bounds_by_target,
)
from pyvmte.utilities import load_paper_dgp
from pyvmte.config import SETUP_FIG5, BLD

import plotly.io as pio  # type: ignore
import pandas as pd  # type: ignore

DGP = load_paper_dgp()

INSTRUMENT = {
    "support_z": DGP["support_z"],
    "pscore_z": DGP["pscore_z"],
    "pdf_z": DGP["pdf_z"],
}


def task_create_bounds_by_target_df(
    path_to_data: Annotated[Path, Product] = BLD
    / "python"
    / "data"
    / "bounds_by_target.pickle"
):
    bounds_by_target = create_bounds_by_target_df(
        setup=SETUP_FIG5,
        instrument=INSTRUMENT,
        m0=DGP["m0"],
        m1=DGP["m1"],
        n_gridpoints=100,
    )
    bounds_by_target.to_pickle(path_to_data)


@pytask.mark.skip()
def task_plot_bounds_by_target(
    path_to_data: Path = BLD / "python" / "data" / "bounds_by_target.pickle",
    path_to_plot: Annotated[Path, Product] = BLD
    / "python"
    / "figures"
    / "bounds_by_target.png",
):
    bounds_by_target = pd.read_pickle(path_to_data)
    fig = plot_bounds_by_target(bounds_by_target)
    pio.write_image(fig, path_to_plot)

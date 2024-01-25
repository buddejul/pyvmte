"""Task for plotting bounds by target."""
from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore
import plotly.io as pio  # type: ignore
import pytask
from pytask import Product

from pyvmte.config import BLD, SETUP_FIG5, Instrument
from pyvmte.replication.plot_bounds_by_target import (
    create_bounds_by_target_df,
    plot_bounds_by_target,
)
from pyvmte.utilities import load_paper_dgp

DGP = load_paper_dgp()

INSTRUMENT = Instrument(
    support=DGP["support_z"],
    pmf=DGP["pdf_z"],
    pscores=DGP["pscores"],
)


def task_create_bounds_by_target_df(
    path_to_data: Annotated[Path, Product] = BLD
    / "python"
    / "data"
    / "bounds_by_target.pickle",
):
    """Create dataframe of bounds by target."""
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
    """Plot bounds by target."""
    bounds_by_target = pd.read_pickle(path_to_data)
    fig = plot_bounds_by_target(bounds_by_target)
    pio.write_image(fig, path_to_plot)

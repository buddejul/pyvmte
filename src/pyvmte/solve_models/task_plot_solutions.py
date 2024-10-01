"""Plot solutions for different linear programs for visual inspection."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task
from scipy.interpolate import BPoly  # type: ignore[import-untyped]

from pyvmte.classes import Setup
from pyvmte.config import BLD, DGP_MST, IV_MST, SETUP_FIG7, U_PART_MST
from pyvmte.identification import identification
from pyvmte.utilities import generate_bernstein_basis_funcs


def plot_solution(funcs: list[BPoly], solution: np.ndarray, bound: float) -> go.Figure:
    """Plot the solution of the MTE problem."""
    u_grid = np.linspace(0, 1, 1000)

    _sol_d0 = solution[: len(funcs)]
    _sol_d1 = solution[len(funcs) :]

    def _bpoly0(u):
        return np.sum([f(u) * c for f, c in zip(funcs, _sol_d0, strict=True)], axis=0)

    def _bpoly1(u):
        return np.sum([f(u) * c for f, c in zip(funcs, _sol_d1, strict=True)], axis=0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=u_grid,
            y=_bpoly0(u_grid),
            mode="lines",
            name="MTR Solution: d = 0",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=u_grid,
            y=_bpoly1(u_grid),
            mode="lines",
            name="MTR Solution: d = 1",
        ),
    )

    fig.update_yaxes(range=[-0.2, 1])

    fig.add_annotation(
        x=0.5,
        y=-0.1,
        text=f"Bound: {bound:.2f}",
        showarrow=False,
    )

    return fig


def solve_and_plot(k: int, setup, lower_or_upper: str) -> go.Figure:
    """Solve problem and plot the solution."""
    bfuncs = generate_bernstein_basis_funcs(k=k)

    res = identification(
        target=setup.target,
        identified_estimands=setup.identified_estimands,
        basis_funcs=bfuncs,
        m0_dgp=DGP_MST.m0,
        m1_dgp=DGP_MST.m1,
        instrument=IV_MST,
        u_partition=U_PART_MST,
        shape_constraints=setup.shape_constraints,
        debug=True,
    )[lower_or_upper]

    bounds = res.fun

    if lower_or_upper == "upper":
        bounds = -bounds

    funcs_for_plot = [f["func"] for f in bfuncs]

    return plot_solution(funcs_for_plot, res.x, bounds)


def _check_solution_decreasing(
    solution: np.ndarray,
    num_bfuncs: int,
    shape_constraints: tuple[str, str],
    eps: float = 1e-6,
) -> str:
    # Check if the solution is decreasing: The first num_bfuncs elements need to be
    # decreasing and the next num_bfuncs elements need to be decreasing as well.
    _sol_d0 = solution[:num_bfuncs]
    _sol_d1 = solution[num_bfuncs + 1 :]

    msg = ""

    if shape_constraints[0] == "decreasing" and not all(np.diff(_sol_d0) <= eps):
        msg += "The solution for d = 0 is not decreasing. "

    if shape_constraints[1] == "decreasing" and not all(np.diff(_sol_d1) <= eps):
        msg += "The solution for d = 1 is not decreasing. "

    if shape_constraints[0] == "increasing" and not all(np.diff(_sol_d0) >= -eps):
        msg += "The solution for d = 0 is not increasing. "

    if shape_constraints[1] == "increasing" and not all(np.diff(_sol_d1) >= -eps):
        msg += "The solution for d = 1 is not increasing. "

    if msg == "":
        msg = "The solution obeys the shape constraints."

    return msg


class _Arguments(NamedTuple):
    k_degree: int
    bound: str
    path_to_plot: Path
    setup: Setup = SETUP_FIG7


k_to_plot = np.arange(5, 15)

ID_TO_KWARGS = {
    f"fig7_degree_{k}_{bound}": _Arguments(
        k_degree=k,
        path_to_plot=BLD / "solutions" / f"fig7_degree_{k}_{bound}.png",
        bound=bound,
    )
    for k in k_to_plot
    for bound in ["lower", "upper"]
}


for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.checks
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_solutions(
        k_degree: int,
        bound: str,
        path_to_plot: Annotated[Path, Product],
        setup: Setup,
    ) -> None:
        """Task for plotting solutions."""
        fig = solve_and_plot(
            k=k_degree,
            lower_or_upper=bound,
            setup=setup,
        )

        fig.write_image(path_to_plot)

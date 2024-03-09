"""Create bounds by target functions."""
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd  # type: ignore

from pyvmte.config import Instrument, Setup
from pyvmte.estimation.estimation import _compute_u_partition, _generate_basis_funcs
from pyvmte.identification import identification

MIN_X_AXIS = 0.45


def create_bounds_by_target_df(
    setup: Setup,
    instrument: Instrument,
    m0: Callable,
    m1: Callable,
    n_gridpoints: int,
) -> pd.DataFrame:
    """Returns dataframe of bounds for different targets."""
    range_of_targets = np.linspace(0.35, 1, n_gridpoints)
    upper_bounds = np.zeros(len(range_of_targets))
    lower_bounds = np.zeros(len(range_of_targets))

    for i, u_hi in enumerate(range_of_targets):
        target = replace(setup.target, u_hi=u_hi)

        u_partition = _compute_u_partition(target, instrument.pscores)
        bfuncs = _generate_basis_funcs("constant", u_partition)
        bounds = identification(
            target=target,
            identified_estimands=setup.identified_estimands,
            basis_funcs=bfuncs,
            m0_dgp=m0,
            m1_dgp=m1,
            instrument=instrument,
            u_partition=u_partition,
        )

        upper_bounds[i] = bounds["upper_bound"]
        lower_bounds[i] = bounds["lower_bound"]

    return pd.DataFrame(
        {
            "u_hi": range_of_targets,
            "upper_bound": upper_bounds,
            "lower_bound": lower_bounds,
            "target": "late",
        },
    )

"""Test identification of simple model using pyvmte against analytical solutions."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from pyvmte.classes import (  # type: ignore[import-untyped]
    Estimand,
    Instrument,
)
from pyvmte.config import RNG
from pyvmte.simulation.simulation_funcs import monte_carlo_pyvmte
from pyvmte.solutions import (
    no_solution_region,
    solution_simple_model,
)

sample_size = 100_000
repetitions = 1_000

# TODO(@buddejul): Need to handle late in estimation. We need to specify the identified
# estimand using estimated propensity scores. Same for the target.

# --------------------------------------------------------------------------------------
# Preliminary settings
# --------------------------------------------------------------------------------------
num_gridpoints = 1

u_hi_late = 0.2

pscore_lo = 0.4
pscore_hi = 0.6

identified_sharp = [
    Estimand(esttype="cross", dz_cross=(d, z)) for d in [0, 1] for z in [0, 1]
]

# Leave pscores unspecified, they are estimated in the simulation. This corresponds to
# an application where the true propensity scores are unknown and hence the true
# target parameter is unknown.
identified_late = [Estimand(esttype="late")]


instrument = Instrument(
    support=np.array([0, 1]),
    pmf=np.array([0.5, 0.5]),
    pscores=np.array([pscore_lo, pscore_hi]),
)


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "id_set",
        "target_type",
        "u_hi",
        "bfunc_type",
        "shape_restriction",
        "mte_monotone",
        "monotone_response",
    ),
    [
        # # LATE-based identified set, extrapolate to LATE
        #     "idlate",
        #     "late",
        #     u_hi_late,
        #     "constant",
        #     None,
        #     None,
        # ),
        #     "idlate",
        #     "late",
        #     u_hi_late,
        #     "constant",
        #     None,
        #     None,
        # ),
        # # LATE-based identified set, monotone treatment selection
        #     "idlate",
        #     "late",
        #     u_hi_late,
        #     "constant",
        #     None,
        #     "decreasing",
        #     None,
        # ),
        #     "idlate",
        #     "late",
        #     u_hi_late,
        #     "constant",
        #     None,
        #     "increasing",
        #     None,
        # ),
        # # LATE-based identified set, monotone treatment selection
        #     "idlate",
        #     "late",
        #     u_hi_late,
        #     "constant",
        #     None,
        #     None,
        #     "positive",
        # ),
        #     "idlate",
        #     "late",
        #     u_hi_late,
        #     "constant",
        #     None,
        #     None,
        #     "negative",
        # ),
        # Sharp identified set, extrapolate to LATE, no shape constraints
        #     "sharp",
        #     "late",
        #     u_hi_late,
        #     "constant",
        #     None,
        #     None,
        #     None,
        # ),
        # # Sharp identified set, extrapolate to ATE
        #     "sharp",
        #     "ate",
        #     1 - pscore_hi,
        #     "constant",
        #     None,
        #     None,
        # ),
        # Sharp identified set, extrapolate to LATE
        (
            "sharp",
            "late",
            u_hi_late,
            "constant",
            ("decreasing", "decreasing"),
            None,
            None,
        ),
    ],
)
def test_simple_model_estimation(
    id_set: str,
    target_type: str,
    u_hi: float,
    bfunc_type: str,
    shape_restriction: tuple[str, str],
    mte_monotone: str | None,
    monotone_response: str | None,
) -> None:
    """Solve the simple model for a range of parameter values."""

    if id_set == "idlate":
        identified = identified_late
    elif id_set == "sharp":
        identified = identified_sharp

    _sol_lo, _sol_hi = solution_simple_model(
        id_set=id_set,
        target_type=target_type,
        pscore_hi=pscore_hi,
        monotone_response=monotone_response,
        mts=mte_monotone,
        u_hi_late_target=u_hi,
        shape_restrictions=shape_restriction,
    )

    _no_sol = no_solution_region(
        id_set=id_set,
        monotone_response=monotone_response,
        mts=mte_monotone,
        shape_restrictions=shape_restriction,
    )

    # Leave pscores unspecified, they are estimated in the simulation.
    target = Estimand(
        "late",
        u_hi_extra=u_hi,
    )

    w = (pscore_hi - pscore_lo) / (pscore_hi - pscore_lo + u_hi)

    # Draw a random point in the parameter space until in the solution region.
    # Otherwise the simulation will probably fail if tuning parameters are not chosen
    # carefully.
    (
        y1_at,
        y1_c,
        y1_nt,
        y0_at,
        y0_c,
        y0_nt,
    ) = RNG.uniform(size=6)

    while _no_sol(y1_at=y1_at, y1_c=y1_c, y0_c=y0_c, y0_nt=y0_nt) is True:
        (
            y1_at,
            y1_c,
            y1_nt,
            y0_at,
            y0_c,
            y0_nt,
        ) = RNG.uniform(size=6)

    dgp_params = {
        "y1_at": y1_at,
        "y1_c": y1_c,
        "y1_nt": y1_nt,
        "y0_at": y0_at,
        "y0_c": y0_c,
        "y0_nt": y0_nt,
    }

    # The identified set might be empty for some parameter value combinations.
    res = monte_carlo_pyvmte(
        sample_size=sample_size,
        repetitions=repetitions,
        target=target,
        identified_estimands=identified,
        basis_func_type=bfunc_type,
        rng=RNG,
        shape_constraints=shape_restriction,
        mte_monotone=mte_monotone,
        monotone_response=monotone_response,
        dgp="simple_model",
        dgp_params=dgp_params,
    )

    _kwargs = {
        "y1_c": y1_c,
        "y0_c": y0_c,
        "y0_nt": y0_nt,
    }

    expected = np.array([_sol_lo(w=w, **_kwargs), _sol_hi(w=w, **_kwargs)])

    data = pd.DataFrame([res["lower_bounds"], res["upper_bounds"]]).T

    columns = {0: "lower_bound", 1: "upper_bound"}

    data = data.rename(columns=columns)
    actual = np.array(data.mean())

    assert expected == pytest.approx(actual, abs=5 / np.sqrt(sample_size))

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

atol = 1e-05

sample_size = 10_000
repetitions = 100

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

identified_late = [Estimand(esttype="late", u_lo=pscore_lo, u_hi=pscore_hi)]


instrument = Instrument(
    support=np.array([0, 1]),
    pmf=np.array([0.5, 0.5]),
    pscores=np.array([pscore_lo, pscore_hi]),
)

target_late = Estimand(esttype="late", u_lo=pscore_lo, u_hi=pscore_hi + u_hi_late)
target_ate = Estimand(esttype="late", u_lo=0, u_hi=1)


def _at(u: float) -> bool:
    return u <= pscore_lo


def _c(u: float) -> bool:
    return pscore_lo <= u and pscore_hi > u


def _nt(u: float) -> bool:
    return u >= pscore_hi


# Define function factories to avoid late binding
# See https://stackoverflow.com/a/3431699
def _make_m0(y0_c, y0_at, y0_nt):
    def _m0(u):
        return y0_at * _at(u) + y0_c * _c(u) + y0_nt * _nt(u)

    return _m0


def _make_m1(y1_c, y1_at, y1_nt):
    def _m1(u):
        return y1_at * _at(u) + y1_c * _c(u) + y1_nt * _nt(u)

    return _m1


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "id_set",
        "target_type",
        "u_hi",
        "bfunc_type",
        "identified",
        "shape_restriction",
        "mte_monotone",
        "monotone_response",
    ),
    [
        # LATE-based identified set, extrapolate to LATE
        (
            "idlate",
            "late",
            u_hi_late,
            "constant",
            identified_late,
            ("decreasing", "decreasing"),
            None,
            None,
        ),
        (
            "idlate",
            "late",
            u_hi_late,
            "constant",
            identified_late,
            ("increasing", "increasing"),
            None,
            None,
        ),
        # LATE-based identified set, monotone treatment selection
        (
            "idlate",
            "late",
            u_hi_late,
            "constant",
            identified_late,
            None,
            "decreasing",
            None,
        ),
        (
            "idlate",
            "late",
            u_hi_late,
            "constant",
            identified_late,
            None,
            "increasing",
            None,
        ),
        # LATE-based identified set, monotone treatment selection
        (
            "idlate",
            "late",
            u_hi_late,
            "constant",
            identified_late,
            None,
            None,
            "positive",
        ),
        (
            "idlate",
            "late",
            u_hi_late,
            "constant",
            identified_late,
            None,
            None,
            "negative",
        ),
        # Sharp identified set, extrapolate to ATE
        (
            "sharp",
            "ate",
            1 - pscore_hi,
            "constant",
            identified_sharp,
            ("decreasing", "decreasing"),
            None,
            None,
        ),
        # Sharp identified set, extrapolate to LATE
        (
            "sharp",
            "late",
            u_hi_late,
            "constant",
            identified_sharp,
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
    identified: list[Estimand],
    shape_restriction: tuple[str, str],
    mte_monotone: str | None,
    monotone_response: str | None,
) -> None:
    """Solve the simple model for a range of parameter values."""

    _sol_lo, _sol_hi = solution_simple_model(
        id_set=id_set,
        target_type=target_type,
        pscore_hi=pscore_hi,
        monotone_response=monotone_response,
        mts=mte_monotone,
        u_hi_late_target=u_hi,
        shape_restrictions=shape_restriction,
    )

    no_solution_region(
        id_set=id_set,
        monotone_response=monotone_response,
        mts=mte_monotone,
        shape_restrictions=shape_restriction,
    )

    target = Estimand(
        "late",
        u_lo=pscore_lo,
        u_hi=pscore_hi + u_hi,
    )

    w = (pscore_hi - pscore_lo) / (pscore_hi - pscore_lo + u_hi)

    # Generate solution for a meshgrid of parameter values
    # TODO only draw single points.
    (
        y1_at,
        y1_c,
        y1_nt,
        y0_at,
        y0_c,
        y0_nt,
    ) = RNG.uniform(size=6)

    _m1 = _make_m1(y1_at=y1_at, y1_c=y1_c, y1_nt=y1_nt)
    _m0 = _make_m0(y0_at=y0_at, y0_c=y0_c, y0_nt=y0_nt)

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
    )

    _kwargs = {
        "y1_c": y1_c,
        "y0_c": y0_c,
        "y0_nt": y0_nt,
    }

    expected = _sol_lo(w=w, **_kwargs), _sol_hi(w=w, **_kwargs)

    data = pd.DataFrame([res["lower_bounds"], res["upper_bounds"]]).T

    columns = {0: "lower_bound", 1: "upper_bound"}

    data = data.rename(columns=columns)
    actual = data.mean()

    assert expected == pytest.approx(actual, abs=5 / np.sqrt(sample_size))

"""Test identification of simple model using pyvmte against analytical solutions."""


import numpy as np
import pytest
from pyvmte.classes import (  # type: ignore[import-untyped]
    Estimand,
    Instrument,
    PyvmteResult,
)
from pyvmte.config import RNG
from pyvmte.identification import identification
from pyvmte.solutions import (
    no_solution_region,
    solution_simple_model,
)

atol = 1e-05

# --------------------------------------------------------------------------------------
# Preliminary settings
# --------------------------------------------------------------------------------------
num_gridpoints = 3

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

bfunc_1 = {"type": "constant", "u_lo": 0.0, "u_hi": pscore_lo}
bfunc_2 = {"type": "constant", "u_lo": pscore_lo, "u_hi": pscore_hi}
bfunc_3_ate = {"type": "constant", "u_lo": pscore_hi, "u_hi": 1}
bfunc_3 = {"type": "constant", "u_lo": pscore_hi, "u_hi": pscore_hi + u_hi_late}
bfunc_4 = {"type": "constant", "u_lo": pscore_hi + u_hi_late, "u_hi": 1}

BFUNCS_ATE = [bfunc_1, bfunc_2, bfunc_3_ate]
BFUNCS_LATE = [bfunc_1, bfunc_2, bfunc_3, bfunc_4]

UPART_ATE = np.array([0, pscore_lo, pscore_hi, 1])
UPART_LATE = np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi_late, 1])


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
        "bfuncs",
        "u_partition",
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
            BFUNCS_LATE,
            UPART_LATE,
            identified_late,
            ("decreasing", "decreasing"),
            None,
            None,
        ),
        (
            "idlate",
            "late",
            u_hi_late,
            BFUNCS_LATE,
            UPART_LATE,
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
            BFUNCS_LATE,
            UPART_LATE,
            identified_late,
            None,
            "decreasing",
            None,
        ),
        (
            "idlate",
            "late",
            u_hi_late,
            BFUNCS_LATE,
            UPART_LATE,
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
            BFUNCS_LATE,
            UPART_LATE,
            identified_late,
            None,
            None,
            "positive",
        ),
        (
            "idlate",
            "late",
            u_hi_late,
            BFUNCS_LATE,
            UPART_LATE,
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
            BFUNCS_ATE,
            UPART_ATE,
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
            BFUNCS_LATE,
            UPART_LATE,
            identified_sharp,
            ("decreasing", "decreasing"),
            None,
            None,
        ),
    ],
)
def test_simple_model_identification(
    id_set: str,
    target_type: str,
    u_hi: float,
    bfuncs: list[dict[str, float]],
    u_partition: np.ndarray,
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

    no_solution = no_solution_region(
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
    (
        y1_at,
        y1_c,
        y1_nt,
        y0_at,
        y0_c,
        y0_nt,
    ) = np.meshgrid(
        RNG.random(num_gridpoints),
        RNG.random(num_gridpoints),
        RNG.random(num_gridpoints),
        RNG.random(num_gridpoints),
        RNG.random(num_gridpoints),
        RNG.random(num_gridpoints),
    )

    # Flatten each meshgrid
    y1_at_flat = y1_at.flatten()
    y1_c_flat = y1_c.flatten()
    y1_nt_flat = y1_nt.flatten()
    y0_at_flat = y0_at.flatten()
    y0_c_flat = y0_c.flatten()
    y0_nt_flat = y0_nt.flatten()

    results = []

    for y1_at, y1_c, y1_nt, y0_at, y0_c, y0_nt in zip(
        y1_at_flat,
        y1_c_flat,
        y1_nt_flat,
        y0_at_flat,
        y0_c_flat,
        y0_nt_flat,
        strict=True,
    ):
        _m1 = _make_m1(y1_at=y1_at, y1_c=y1_c, y1_nt=y1_nt)
        _m0 = _make_m0(y0_at=y0_at, y0_c=y0_c, y0_nt=y0_nt)

        # The identified set might be empty for some parameter value combinations.
        try:
            res = identification(
                target=target,
                identified_estimands=identified,
                basis_funcs=bfuncs,
                instrument=instrument,
                u_partition=u_partition,
                m0_dgp=_m0,
                m1_dgp=_m1,
                shape_constraints=shape_restriction,
                mte_monotone=mte_monotone,
                monotone_response=monotone_response,
            )
        except TypeError:
            res = PyvmteResult(
                procedure="identification",
                lower_bound=np.nan,
                upper_bound=np.nan,
                basis_funcs=bfuncs,
                method="highs",
                lp_api="scipy",
                lower_optres=None,
                upper_optres=None,
                lp_inputs=None,  # type: ignore[arg-type]
            )

        results.append(res)

    actual_lo = np.array([res.lower_bound for res in results])
    actual_hi = np.array([res.upper_bound for res in results])

    # Put into pandas DataFrame and save to disk
    _kwargs = {
        "y1_c": y1_c_flat,
        "y0_c": y0_c_flat,
        "y0_nt": y0_nt_flat,
    }

    expected_lo = _sol_lo(w=w, **_kwargs)
    expected_hi = _sol_hi(w=w, **_kwargs)

    _idx_no_sol = no_solution(y1_at=y1_at_flat, **_kwargs)
    expected_lo[_idx_no_sol] = np.nan
    expected_hi[_idx_no_sol] = np.nan

    # Get _idx of nan mismatch
    np.where(np.isnan(actual_lo) != np.isnan(expected_lo))
    np.where(np.isnan(actual_hi) != np.isnan(expected_hi))

    # Get _idx of value mismatch
    np.where(np.abs(actual_lo - expected_lo) > atol)
    np.where(np.abs(actual_hi - expected_hi) > atol)

    np.testing.assert_allclose(actual_lo, expected_lo, atol=atol)
    np.testing.assert_allclose(actual_hi, expected_hi, atol=atol)

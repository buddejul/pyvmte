"""Collects solutions to specific MTE models."""

from collections.abc import Callable
from functools import partial

import numpy as np


def solution_simple_model(
    id_set: str,
    target_type: str,
    pscore_hi: float,
    shape_restrictions: tuple[str, str] | None = None,
    monotone_response: str | None = None,
    mts: str | None = None,
    u_hi_late_target: float | None = None,
) -> tuple[Callable, Callable]:
    """Select the solution function for the simple model."""
    _shape_restrictions = [shape_restrictions, monotone_response, mts]

    # Count the number of None values in the shape_restrictions tuple.
    # If the count is not >= 2, raise a ValueError.
    _min_none = 2
    if _shape_restrictions.count(None) < _min_none:
        msg = "At least two of the shape restrictions must be None."
        raise ValueError(msg)

    solution_functions_no_restrictions = {}  # type: ignore[var-annotated]

    solution_functions_shape_restrictions = {
        "idlate": {
            "late": {
                ("decreasing", "decreasing"): (
                    _sol_lo_idlate_decreasing,
                    _sol_hi_idlate_decreasing,
                ),
                ("increasing", "increasing"): (
                    _sol_lo_idlate_increasing,
                    _sol_hi_idlate_increasing,
                ),
            },
        },
        "sharp": {
            "ate": {
                ("decreasing", "decreasing"): (_sol_lo_sharp_ate, _sol_hi_sharp_ate),
            },
            "late": {
                ("decreasing", "decreasing"): (_sol_lo_sharp_late, _sol_hi_sharp_late),
            },
        },
    }

    solution_functions_monotone_response = {
        "idlate": {
            "ate": {
                "positive": (
                    _sol_lo_idlate_monotone_response_positive,
                    _sol_hi_idlate_monotone_response_positive,
                ),
                "negative": (
                    _sol_lo_idlate_monotone_response_negative,
                    _sol_hi_idlate_monotone_response_negative,
                ),
            },
            "late": {
                "positive": (
                    _sol_lo_idlate_monotone_response_positive,
                    _sol_hi_idlate_monotone_response_positive,
                ),
                "negative": (
                    _sol_lo_idlate_monotone_response_negative,
                    _sol_hi_idlate_monotone_response_negative,
                ),
            },
        },
        # TODO(@buddejul): Solve this for monotone response.
        "sharp": {"ate": {}, "late": {}},
    }

    solution_functions_mts = {
        "idlate": {
            "ate": {
                "increasing": (
                    _sol_lo_idlate_mts_increasing,
                    _sol_hi_idlate_mts_increasing,
                ),
                "decreasing": (
                    _sol_lo_idlate_mts_decreasing,
                    _sol_hi_idlate_mts_decreasing,
                ),
            },
            "late": {
                "increasing": (
                    _sol_lo_idlate_mts_increasing,
                    _sol_hi_idlate_mts_increasing,
                ),
                "decreasing": (
                    _sol_lo_idlate_mts_decreasing,
                    _sol_hi_idlate_mts_decreasing,
                ),
            },
        },
        # TODO(@buddejul): Solve this for monotone selection.
        "sharp": {},
    }

    if shape_restrictions is not None:
        _out = solution_functions_shape_restrictions[id_set][target_type][
            shape_restrictions
        ]

    if monotone_response is not None:
        _out = solution_functions_monotone_response[id_set][target_type][  # type: ignore[index]
            monotone_response
        ]

    if mts is not None:
        _out = solution_functions_mts[id_set][target_type][mts]  # type: ignore[index]

    if monotone_response is None and mts is None and shape_restrictions is None:
        _out = solution_functions_no_restrictions[id_set][target_type]

    if _out == {}:
        msg = "No solution function found."
        raise ValueError(msg)

    _kwargs = {
        "pscore_hi": pscore_hi,
        "u_hi_late_target": u_hi_late_target,
    }

    _sol_lo = partial(_out[0], **_kwargs)
    _sol_hi = partial(_out[1], **_kwargs)

    return _sol_lo, _sol_hi


# --------------------------------------------------------------------------------------
# Solution functions
# --------------------------------------------------------------------------------------


def _sol_hi_sharp_ate(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c

    return w * _b_late + (1 - w) * (y1_c - y0_nt)


def _sol_lo_sharp_ate(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c

    return w * _b_late + (1 - w) * (0 - y0_nt)


def _sol_hi_sharp_late(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    return _sol_hi_sharp_ate(
        w=w,
        y1_c=y1_c,
        y0_c=y0_c,
        y0_nt=y0_nt,
        u_hi_late_target=u_hi_late_target,
        pscore_hi=pscore_hi,
    )


def _sol_lo_sharp_late(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    _b_late = y1_c - y0_c

    k = u_hi_late_target / (1 - pscore_hi)

    return w * _b_late + (1 - w) * (0 - np.minimum(y0_c, y0_nt / k))


def _sol_hi_idlate_increasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return (_b_late >= 0) * (w * _b_late + (1 - w)) + (_b_late < 0) * (
        _b_late + (1 - w)
    )


def _sol_lo_idlate_increasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return (_b_late >= 0) * (_b_late - (1 - w)) + (_b_late < 0) * (
        w * _b_late - (1 - w)
    )


def _sol_hi_idlate_decreasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return (_b_late >= 0) * (w * _b_late + (1 - w)) + (_b_late < 0) * (
        w * _b_late + (1 - w) * (1 + _b_late)
    )


def _sol_lo_idlate_decreasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return (_b_late >= 0) * (w * _b_late + (1 - w) * (_b_late - 1)) + (_b_late < 0) * (
        w * _b_late + (1 - w) * (-1)
    )


def _sol_hi_idlate_mts_decreasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi, w
    return y1_c - y0_c


def _sol_hi_idlate_mts_increasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return w * _b_late + (1 - w) * 1


def _sol_lo_idlate_mts_increasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    return y1_c - y0_c


def _sol_lo_idlate_mts_decreasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return w * _b_late + (1 - w) * -1


def _sol_hi_idlate_monotone_response_positive(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return w * _b_late + (1 - w) * 1


def _sol_hi_idlate_monotone_response_negative(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return w * _b_late + (1 - w) * 0


def _sol_lo_idlate_monotone_response_positive(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return w * _b_late + (1 - w) * 0


def _sol_lo_idlate_monotone_response_negative(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c
    return w * _b_late + (1 - w) * (-1)


# --------------------------------------------------------------------------------------
# Functions to indicate region of no solution
# --------------------------------------------------------------------------------------


def _no_solution(y1_at, y1_c, y0_c, y0_nt):
    # The model is not consistent with decreasing MTRs if
    # - y1_at < y1_c or y1_c < y1_nt or
    # - y0_at < y0_c or y0_c < y0_nt
    # Make sure this also works vectorized.
    # Note that while y1_c < y1_nt is also not consistent with decreasing MTRs, the
    # model puts no restrictions on y1_nt since it is not identified.
    return np.logical_or(y1_at < y1_c, y0_c < y0_nt)


def _no_solution_nonsharp(y1_at, y1_c, y0_c, y0_nt):
    return False


def _no_solution_nonsharp_monotone_response(
    monotone_response,
    y1_at,
    y1_c,
    y0_c,
    y0_nt,
):
    if monotone_response == "positive":
        return y1_c - y0_c <= 0
    if monotone_response == "negative":
        return y1_c - y0_c >= 0
    return None

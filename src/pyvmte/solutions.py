"""Collects solutions to specific MTE models."""

from collections.abc import Callable
from functools import partial

import numpy as np

from pyvmte.config import RNG


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

    solution_functions_no_restrictions = {
        "idlate": {"late": (_sol_lo_idlate, _sol_hi_idlate)},
        "sharp": {
            "ate": (_sol_lo_sharp, _sol_hi_sharp),
            "late": (_sol_lo_sharp, _sol_hi_sharp),
        },
    }  # type: ignore[var-annotated]

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
                ("decreasing", "decreasing"): (
                    _sol_lo_sharp_ate_decreasing,
                    _sol_hi_sharp_ate_decreasing,
                ),
            },
            "late": {
                ("decreasing", "decreasing"): (
                    _sol_lo_sharp_late_decreasing,
                    _sol_hi_sharp_late_decreasing,
                ),
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
        "sharp": {
            "ate": {
                "positive": (
                    partial(
                        _sol_lo_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                    partial(
                        _sol_hi_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                ),
                "negative": (
                    partial(
                        _sol_lo_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                    partial(
                        _sol_hi_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                ),
            },
            "late": {
                "positive": (
                    partial(
                        _sol_lo_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                    partial(
                        _sol_hi_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                ),
                "negative": (
                    partial(
                        _sol_lo_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                    partial(
                        _sol_hi_sharp_monotone_response,
                        monotone_response=monotone_response,
                    ),
                ),
            },
        },
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
        "sharp": {
            "late": {
                "increasing": (
                    partial(_sol_lo_sharp_mts, mts=mts),
                    partial(_sol_hi_sharp_mts, mts=mts),
                ),
                "decreasing": (
                    partial(_sol_lo_sharp_mts, mts=mts),
                    partial(_sol_hi_sharp_mts, mts=mts),
                ),
            },
            "ate": {
                "increasing": (
                    partial(_sol_lo_sharp_mts, mts=mts),
                    partial(_sol_hi_sharp_mts, mts=mts),
                ),
                "decreasing": (
                    partial(_sol_lo_sharp_mts, mts=mts),
                    partial(_sol_hi_sharp_mts, mts=mts),
                ),
            },
        },
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


def _sol_hi_sharp(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
) -> float:
    k = u_hi_late_target / (1 - pscore_hi)

    a = np.clip((y0_nt - (1 - k)) / k, a_min=0, a_max=1)

    return w * (y1_c - y0_c) + (1 - w) * (1 - a)


def _sol_lo_sharp(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
) -> float:
    k = u_hi_late_target / (1 - pscore_hi)
    a = np.clip(y0_nt / k, a_min=0, a_max=1)
    return w * (y1_c - y0_c) + (1 - w) * (0 - a)


def _sol_hi_sharp_ate_decreasing(
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


def _sol_lo_sharp_ate_decreasing(
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


def _sol_hi_sharp_late_decreasing(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    return _sol_hi_sharp_ate_decreasing(
        w=w,
        y1_c=y1_c,
        y0_c=y0_c,
        y0_nt=y0_nt,
        u_hi_late_target=u_hi_late_target,
        pscore_hi=pscore_hi,
    )


def _sol_lo_sharp_late_decreasing(
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


def _sol_hi_sharp_mts(
    mts: str,
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del (
        u_hi_late_target,
        pscore_hi,
    )
    _b_late = y1_c - y0_c

    if mts == "decreasing":
        return w * _b_late + (1 - w) * _b_late
    if mts == "increasing":
        return w * _b_late + (1 - w) * (1 - y0_nt)
    return None


def _sol_lo_sharp_mts(
    mts: str,
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    del u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c

    if mts == "decreasing":
        return w * _b_late + (1 - w) * (0 - y0_nt)
    if mts == "increasing":
        return w * _b_late + (1 - w) * _b_late
    return None


def _sol_hi_sharp_monotone_response(
    monotone_response: str,
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
):
    k = u_hi_late_target / (1 - pscore_hi)

    a = np.clip((y0_nt - (1 - k)) / k, a_min=0, a_max=1)

    if monotone_response == "positive":
        return w * (y1_c - y0_c) + (1 - w) * np.clip(1 - a, a_min=0, a_max=1)
    if monotone_response == "negative":
        return w * (y1_c - y0_c) + (1 - w) * np.clip(1 - a, a_max=0, a_min=-1)

    msg = f"Invalid monotone_response: {monotone_response}."
    raise ValueError(msg)


def _sol_lo_sharp_monotone_response(
    monotone_response: str,
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
) -> float:
    k = u_hi_late_target / (1 - pscore_hi)

    a = np.clip(y0_nt / k, a_min=0, a_max=1)

    if monotone_response == "positive":
        return w * (y1_c - y0_c) + (1 - w) * np.clip(0 - a, a_min=0, a_max=1)

    if monotone_response == "negative":
        return w * (y1_c - y0_c) + (1 - w) * np.clip(0 - a, a_max=0, a_min=-1)

    msg = f"Invalid monotone_response: {monotone_response}."
    raise ValueError(msg)


# --------------------------------------------------------------------------------------
# Solution functions for using only the LATE for identification
# --------------------------------------------------------------------------------------
def _sol_lo_idlate(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
) -> float:
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c

    return w * _b_late + (1 - w) * (-1)


def _sol_hi_idlate(
    w: float,
    y1_c: float,
    y0_c: float,
    y0_nt: float,
    u_hi_late_target: float,
    pscore_hi: float,
) -> float:
    del y0_nt, u_hi_late_target, pscore_hi
    _b_late = y1_c - y0_c

    return w * _b_late + (1 - w) * (1)


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
# TODO(@buddejul): Check we cover all cases here.
def no_solution_region(  # noqa: PLR0911
    id_set: str,
    shape_restrictions: tuple[str, str] | None = None,
    monotone_response: str | None = None,
    mts: str | None = None,
):
    """Return function indicating parameter region with no solution."""
    _all_restr = [shape_restrictions, monotone_response, mts]

    if id_set == "sharp":
        if shape_restrictions is not None:
            return partial(
                _no_solution_sharp_shape,
                shape_restrictions=shape_restrictions,
            )
        if monotone_response is not None:
            return partial(
                _no_solution_sharp_monotone_response,
                monotone_response=monotone_response,
            )
        if mts is not None:
            return partial(
                _no_solution_sharp_mts,
                mts=mts,
            )

        return _no_solution_sharp

    if id_set == "idlate":
        if shape_restrictions is not None:
            return partial(
                _no_solution_idlate_shape,
                shape_restrictions=shape_restrictions,
            )
        if monotone_response is not None:
            return partial(
                _no_solution_idlate_monotone_response,
                monotone_response=monotone_response,
            )
        if mts is not None:
            return partial(_no_solution_idlate_mts, mts=mts)

        return _no_solution_idlate

    msg = "Invalid id_set or no function found."
    raise ValueError(msg)


def _no_solution_sharp(y1_at, y1_c, y0_c, y0_nt):
    """Without any restrictions on MTR function the model always has a solution."""
    return False


def _no_solution_sharp_shape(shape_restrictions, y1_at, y1_c, y0_c, y0_nt):
    # The model is not consistent with decreasing MTRs if
    # - y1_at < y1_c or y1_c < y1_nt or
    # - y0_at < y0_c or y0_c < y0_nt
    # Make sure this also works vectorized.
    # Note that while y1_c < y1_nt is also not consistent with decreasing MTRs, the
    # model puts no restrictions on y1_nt since it is not identified.
    if shape_restrictions == ("decreasing", "decreasing"):
        return np.logical_or(y1_at < y1_c, y0_c < y0_nt)
    if shape_restrictions == ("increasing", "increasing"):
        msg = "No solution region not implemented yet."
        raise ValueError(msg)
    return None


def _no_solution_sharp_monotone_response(monotone_response, y1_at, y1_c, y0_c, y0_nt):
    del y1_at, y0_nt
    if monotone_response == "positive":
        return y1_c - y0_c <= 0
    if monotone_response == "negative":
        return y1_c - y0_c >= 0
    return None


def _no_solution_sharp_mts(mts, y1_at, y1_c, y0_c, y0_nt):
    """No solution region for sharp ID set in the simple model with MTS.

    Decreasing MTE function (positive MTS):
    - beta_late needs to be smaller than the largest possible always-taker treatment
        effect which is y1_at - 0.
    - beta_late needs to be larger than the smallest possible never-taker treatment
        effect which is 0 - y0_nt.

    Increasing MTE function (negative MTS):
    - beta_late >= y1_at - 1 (smallest possible always-taker treatment effect).
    - beta_late <= 1 - y0_nt (largest possible never-taker treatment effect).

    """
    _b_late = y1_c - y0_c
    if mts == "decreasing":
        return np.logical_or(_b_late > y1_at - 0, _b_late < 0 - y0_nt)
    if mts == "increasing":
        return np.logical_or(_b_late < y1_at - 1, _b_late > 1 - y0_nt)

    msg = "MTS needs to be either 'increasing' or 'decreasing'."
    raise ValueError(msg)


def _no_solution_idlate(y1_at, y1_c, y0_c, y0_nt):
    """No solution to non-sharp ID set in simple model without any restrictions."""
    return False


def _no_solution_idlate_monotone_response(
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


def _no_solution_idlate_shape(shape_restrictions, y1_at, y1_c, y0_c, y0_nt):
    """No-solution region non-sharp ID set in simple model with in/decreasing MTRs.

    Note if only the complier LATE is used for identification, y1_at and y0_nt are not
    used to put restrictions on the model, hence the no-solution region is empty. For
    example, a trivial feasible solution is that all MTR functions are equivalent. This
    cannot be rejected by the data since no information on y1_at and y0_nt is used.

    """
    if shape_restrictions == ("decreasing", "decreasing"):
        return False
    if shape_restrictions == ("increasing", "increasing"):
        return False

    msg = f"Invalid shape_restrictions: {shape_restrictions}."
    raise ValueError(msg)


def _no_solution_idlate_mts(mts, y1_at, y1_c, y0_c, y0_nt):
    """No solution region for non-sharp ID set in the simple model with MTS.

    Only information on the complier LATE is used. Hence there are no restrictions.

    """
    if mts == "decreasing":
        return False
    if mts == "increasing":
        return False

    msg = f"Invalid mts: {mts}."
    raise ValueError(msg)


# --------------------------------------------------------------------------------------
# Other helper function
# --------------------------------------------------------------------------------------


def draw_valid_simple_model_params(no_solution_region: Callable) -> tuple:
    """Draw parameters for the simple model inside the solution region."""
    while True:
        (
            y1_at,
            y1_c,
            y1_nt,
            y0_at,
            y0_c,
            y0_nt,
        ) = RNG.uniform(size=6)

        if not no_solution_region(y1_at=y1_at, y1_c=y1_c, y0_c=y0_c, y0_nt=y0_nt):
            break

    return y1_at, y1_c, y1_nt, y0_at, y0_c, y0_nt

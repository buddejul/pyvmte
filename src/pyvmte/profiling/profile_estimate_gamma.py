"""Profile the estimation of gamma linear maps."""
import numpy as np
from numba import njit  # type: ignore

from pyvmte.config import RNG, Estimand, Instrument
from pyvmte.estimation.estimation import (
    _compute_u_partition,
    _estimate_instrument_characteristics,
    _estimate_moments_for_weights,
    _generate_array_of_pscores,
    _generate_basis_funcs,
)
from pyvmte.utilities import simulate_data_from_paper_dgp

sample_size = 100_000

d_value = 0
ols_estimand = Estimand(esttype="ols_slope")
iv_estimand = Estimand(esttype="iv_slope")
cross_estimand = Estimand(esttype="cross", dz_cross=(0, 1))
late_estimand = Estimand(esttype="late", u_lo=0.35, u_hi=0.9)

bfunc = {
    "u_lo": 0.35,
    "u_hi": 0.7,
    "type": "constant",
}

data = simulate_data_from_paper_dgp(sample_size=sample_size, rng=RNG)

instrument = _estimate_instrument_characteristics(z_data=data["z"], d_data=data["d"])
data["pscores"] = _generate_array_of_pscores(
    z_data=data["z"],
    support=instrument.support,
    pscores=instrument.pscores,
)

moments = _estimate_moments_for_weights(z_data=data["z"], d_data=data["d"])

u_partition = _compute_u_partition(target=late_estimand, pscore_z=instrument.pscores)

basis_functions = _generate_basis_funcs("constant", u_partition)


def _estimate_weights_estimand(
    estimand: Estimand,
    basis_funcs: list,
    data: dict[str, np.ndarray],
    moments: dict,
) -> np.ndarray:
    """Estimate the weights on each basis function for a single estimand."""
    weights = np.zeros(len(basis_funcs) * 2)

    for d_value in [0, 1]:
        for i, basis_func in enumerate(basis_funcs):
            idx = i + d_value * len(basis_funcs)
            weights[idx] = _estimate_gamma_for_basis_funcs(
                d_value=d_value,
                estimand=estimand,
                basis_func=basis_func,
                data=data,
                moments=moments,
            )

    return weights


# Function creating list of tuples from list of dictionaries
def _create_list_of_tuples(basis_funcs: list) -> list[tuple]:
    """Create list of tuples from list of dictionaries."""
    basis_funcs_tuples = []
    for basis_func in basis_funcs:
        basis_funcs_tuples.append((basis_func["u_lo"], basis_func["u_hi"]))

    return basis_funcs_tuples


# Function creating array of all hi or low points from dictionary
def _create_array_of_points(basis_funcs: list, point: str) -> np.ndarray:
    """Create array of all hi or low points from dictionary."""
    points = []
    for basis_func in basis_funcs:
        points.append(basis_func[point])

    return np.array(points)


@njit()
def _njit__estimate_weights_estimand(
    estimand_type: str,
    basis_funcs_hi: np.ndarray,
    basis_funcs_lo: np.ndarray,
    pscores: np.ndarray,
    expectation_d,
    variance_d,
) -> np.ndarray:
    """Estimate the weights on each basis function for a single estimand."""
    number_bfuncs = len(basis_funcs_hi)

    weights = np.zeros(number_bfuncs * 2, dtype=np.float64)

    for d_value in range(2):
        coef = (d_value - expectation_d) / variance_d

        for i in range(number_bfuncs):
            indicators = np.full(len(pscores), fill_value=False)

            for j in range(len(pscores)):
                if d_value == 0:
                    if basis_funcs_lo[i] >= pscores[j]:
                        indicators[j] = True
                elif basis_funcs_hi[i] <= pscores[j]:
                    indicators[j] = True

            weights[i + d_value * number_bfuncs] = (
                basis_funcs_hi[i] - basis_funcs_lo[i]
            ) * np.mean(coef * indicators)

    return weights


def _optimized_estimate_gamma_for_basis_funcs(
    d_value: int,
    estimand: Estimand,
    basis_func: dict,
    data: dict,
    moments: dict,
    instrument: Instrument,
) -> float:
    """Estimate gamma linear map for basis function (cf.

    S33 in Appendix).

    """
    length = basis_func["u_hi"] - basis_func["u_lo"]

    if estimand.esttype == "ols_slope":
        coef = (d_value - moments["expectation_d"]) / moments["variance_d"]
    if estimand.esttype == "iv_slope":
        coef = (data["z"] - moments["expectation_z"]) / moments["covariance_dz"]
    if estimand.esttype == "cross":
        d_cross = estimand.dz_cross[0]  # type: ignore
        z_cross = estimand.dz_cross[1]  # type: ignore

        if d_value != d_cross:
            return 0
        if d_value == 0:
            # Could use instrment values here!
            if basis_func["u_lo"] < instrument.pscores[0]:
                return 0
            mask1 = basis_func["u_lo"] >= data["pscores"]
            mask2 = data["z"] == z_cross
            cross_indicators = np.logical_and(mask1, mask2)
        if d_value == 1:
            if basis_func["u_hi"] > instrument.pscores[-1]:
                return 0
            mask1 = basis_func["u_hi"] <= data["pscores"]
            mask2 = data["z"] == z_cross
            cross_indicators = np.logical_and(mask1, mask2)

        return length * np.count_nonzero(cross_indicators) / len(cross_indicators)

    if d_value == 0:
        # Create array of 1 if basis_funcs["u_lo"] > data["pscores"] else 0
        indicators = basis_func["u_lo"] >= data["pscores"]
    else:
        indicators = basis_func["u_hi"] <= data["pscores"]

    if estimand.esttype == "ols_slope":
        share = np.count_nonzero(indicators) / len(indicators)
        return length * coef * share
    return length * np.mean(coef * indicators)


def _estimate_gamma_for_basis_funcs(
    d_value: int,
    estimand: Estimand,
    basis_func: dict,
    data: dict,
    moments: dict,
) -> float:
    """Estimate gamma linear map for basis function (cf.

    S33 in Appendix).

    """
    length = basis_func["u_hi"] - basis_func["u_lo"]

    if estimand.esttype == "ols_slope":
        coef = (d_value - moments["expectation_d"]) / moments["variance_d"]
    if estimand.esttype == "iv_slope":
        coef = (data["z"] - moments["expectation_z"]) / moments["covariance_dz"]
    if estimand.esttype == "cross":
        d_cross = estimand.dz_cross[0]  # type: ignore
        z_cross = estimand.dz_cross[1]  # type: ignore
        coef = np.where(d_value == d_cross, data["z"] == z_cross, 0)

    if d_value == 0:
        # Create array of 1 if basis_funcs["u_lo"] > data["pscores"] else 0
        indicators = np.where(basis_func["u_lo"] >= data["pscores"], 1, 0)
    else:
        indicators = np.where(basis_func["u_hi"] <= data["pscores"], 1, 0)

    return length * np.mean(coef * indicators)

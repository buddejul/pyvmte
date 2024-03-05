"""Setup for profiling simulation functions."""
import numpy as np
import pandas as pd  # type: ignore


def simulate_data_from_paper_dgp(sample_size, rng):
    """Simulate data using the dgp from MST 2018 ECMA."""
    data = pd.DataFrame()

    support = np.array([0, 1, 2])
    pmf = np.array([0.5, 0.4, 0.1])
    pscores = np.array([0.35, 0.6, 0.7])

    choices = np.hstack([support.reshape(-1, 1), pscores.reshape(-1, 1)])

    # Draw random ndices
    idx = rng.choice(support, size=sample_size, p=pmf)

    data = choices[idx]

    # Put data into df
    z = np.array(data[:, 0], dtype=int)
    pscores = data[:, 1]

    u = rng.uniform(size=sample_size)
    d = u < pscores

    y = np.empty(sample_size)
    idx = d == 0
    # TODO (@buddejul):  do this properly
    y[idx] = (
        +0.6 * (1 - u[idx]) ** 2 + 0.4 * 2 * u[idx] * (1 - u[idx]) + 0.3 * u[idx] ** 2
    )

    y[~idx] = (
        +0.75 * (1 - u[~idx]) ** 2
        + 0.5 * 2 * u[~idx] * (1 - u[~idx])
        + 0.25 * u[~idx] ** 2
    )

    return {"z": z, "d": d, "y": y, "u": u}


# @njit
# def bern_bas(n, v, x):
#     """Bernstein polynomial basis of degree n and index v at point x."""

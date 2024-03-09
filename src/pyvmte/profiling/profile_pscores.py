"""Setup for profiling pscore functions."""
import numpy as np

from pyvmte.config import RNG

supp = np.array([0, 1, 2])
pmf = np.array([0.5, 0.4, 0.1])
pscores = np.array([0.3, 0.6, 0.7])

z_dict = dict(zip(supp, pscores, strict=True))

size = 100_000

z = RNG.choice(supp, size=size, p=pmf)


def slow_func(z, z_dict):
    """Non-vectorized function."""
    return np.array([z_dict[z_] for z_ in z])


def pscore_function(z: float) -> float:
    """Non-vectorized function."""
    if z == 0:
        return 0.35
    if z == 1:
        return 0.6
    if z == 2:  # noqa: PLR2004
        return 0.7

    msg = "z must be in {0, 1, 2}"
    raise ValueError(msg)


def vectorized_func(z: np.ndarray) -> np.ndarray:
    """Vectorized function."""
    return np.vectorize(pscore_function)(z)


def dict_vectorized(z: np.ndarray, z_dict: dict) -> np.ndarray:
    """Vectorized function using a dictionary."""
    return np.vectorize(z_dict.get)(z)

import numpy as np

supp = np.array([0, 1, 2])
pmf = np.array([0.5, 0.4, 0.1])
pscores = np.array([0.3, 0.6, 0.7])

z_dict = dict(zip(supp, pscores))

size = 100_000

z = np.random.choice(supp, size=size, p=pmf)


def slow_func(z, z_dict):
    return np.array([z_dict[z_] for z_ in z])


def pscore_function(z: int | float) -> float:
    if z == 0:
        return 0.35
    elif z == 1:
        return 0.6
    elif z == 2:
        return 0.7
    else:
        raise ValueError("z must be in {0, 1, 2}")


def vectorized_func(z: np.ndarray) -> np.ndarray:
    return np.vectorize(pscore_function)(z)


def dict_vectorized(z: np.ndarray, z_dict: dict) -> np.ndarray:
    return np.vectorize(z_dict.get)(z)

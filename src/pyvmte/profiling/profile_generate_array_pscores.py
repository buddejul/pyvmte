import numpy as np

supp = np.array([0, 1, 2])
pmf = np.array([0.5, 0.4, 0.1])
pscores = np.array([0.3, 0.6, 0.7])

z_data = np.random.choice(supp, size=100_000, p=pmf)
d_data = np.random.choice([0, 1], size=100_000, p=[0.5, 0.5])


# timeit benchmark: 3.9ms
def _generate_array_of_pscores(z_data: np.ndarray, d_data: np.ndarray) -> np.ndarray:
    """For input data on instrument and treatment generates array of same length with
    estimated propensity scores for each corresponding entry of z."""

    # Estimate propensity scores
    p = _estimate_prop_z(z_data, d_data)

    # Get vector of p corresponding to z
    supp_z = np.unique(z_data)
    return p[np.searchsorted(supp_z, z_data)]


# timeit 1.99ms
def _estimate_prop_z(z_data: np.ndarray, d_data: np.ndarray) -> np.ndarray:
    """Estimate propensity score of z given d."""

    supp_z = np.unique(z_data)

    pscore = []

    for z_val in supp_z:
        pscore.append(np.mean(d_data[z_data == z_val]))

    return np.array(pscore)

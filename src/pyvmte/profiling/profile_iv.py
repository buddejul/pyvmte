"""Setup for profiling iv estimation functions."""
import numpy as np

from pyvmte.config import RNG

sample_size = 100_000

support = np.array([0, 1, 2])

z = RNG.choice(support, size=sample_size, p=[0.5, 0.4, 0.1])
d = RNG.choice([0, 1], size=sample_size, p=[0.65, 0.35])


def _estimate_prop_z(
    z_data: np.ndarray,
    d_data: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Estimate propensity score of z given d."""
    pscore = []

    for z_val in support:
        pscore.append(np.mean(d_data[z_data == z_val]))

    return np.array(pscore)

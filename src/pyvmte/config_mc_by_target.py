"""Settings for Monte Carlo simulations by target."""
import numpy as np

from pyvmte.classes import MonteCarloSetup

MONTE_CARLO_BY_TARGET = MonteCarloSetup(
    sample_size=10_000,
    repetitions=1_000,
    u_hi_range=np.arange(0.40, 1, 0.05),
)

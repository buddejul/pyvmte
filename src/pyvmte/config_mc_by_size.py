"""Settings for Monte Carlo simulations by sample size."""
from pyvmte.classes import MonteCarloSetup

MC_SAMPLE_SIZES = [500, 2500, 10000]

MONTE_CARLO_BY_SIZE = MonteCarloSetup(
    sample_size=1_000,
    repetitions=100,
)

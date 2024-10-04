"""Define custom classes for pyvmte."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import numpy as np
from scipy.optimize import OptimizeResult  # type: ignore[import-untyped]


@dataclass
class Estimand:
    """Target estimand.

    For identification need to specify `u_lo` and `u_hi` if type is late.
    If in addition `u_lo_extra` and `u_hi_extra` are specified, they are added (hi) or
    subtracted (lo) to `u_hi` and `u_lo' internally.

    For estimation, if late is specified, only `u_lo_extra' and/or `u_hi_extra` may be
    provided. In this case, `u_lo' and `u_hi` are estimated from the data.

    """

    # TODO(@buddejul): Unexpected results when both u_hi and u_hi_extra are specified
    # for identification.
    esttype: str
    u_lo: float | None = None
    u_hi: float | None = None
    dz_cross: tuple[int, int] | None = None
    u_lo_extra: float | None = None
    u_hi_extra: float | None = None


@dataclass
class Instrument:
    """Discrete instrument."""

    support: np.ndarray
    pmf: np.ndarray
    pscores: np.ndarray


@dataclass
class DGP:
    """Data Generating Process."""

    m0: Callable
    m1: Callable
    support_z: np.ndarray
    pmf_z: np.ndarray
    pscores: np.ndarray
    joint_pmf_dz: dict[int, dict[int, float]]

    @property
    def expectation_z(self):
        """Expectation of instrument Z."""
        return np.sum(self.support_z * self.pmf_z)

    @property
    def expectation_d(self):
        """Expectation of binary treatment D."""
        return np.sum(self.pscores * self.pmf_z)

    @property
    def variance_d(self):
        """Variance of binary treatment D."""
        return self.expectation_d * (1 - self.expectation_d)

    @property
    def covariance_dz(self):
        """Covariance of binary treatment D and instrument Z."""
        return np.sum(
            [
                self.joint_pmf_dz[d][z]
                * (d - self.expectation_d)
                * (z - self.expectation_z)
                for d in [0, 1]
                for z in self.support_z
            ],
        )


class Setup(NamedTuple):
    """Setup from the paper."""

    target: Estimand
    identified_estimands: list[Estimand]
    lower_bound: float
    upper_bound: float
    shape_constraints: tuple[str, str] | None = None
    polynomial: tuple[str, int] | None = None


class MonteCarloSetup(NamedTuple):
    """Setup for Monte Carlo simulations."""

    sample_size: int
    repetitions: int
    u_hi_range: np.ndarray | None = None


class Bern:
    """Bernstein polynomial of degree n with coefficients on the basis functions."""

    bfunc_type: str = "bernstein"
    lo: float = 0
    hi: float = 1

    def __init__(self, n, coefs):
        """Initialize the Bernstein polynomial with degree n and coefficients coefs."""
        self.n = n
        self.coefs = coefs

    def __call__(self, x):
        """Evaluate the Bernstein polynomial at point x."""
        return sum([self._bern_bas(self.n, i, x) * c for i, c in enumerate(self.coefs)])

    def _bern_bas(self, n, v, x):
        return math.comb(n, v) * x**v * (1 - x) ** (n - v)

    def _indef_integral(self, u: float, i: int) -> float:
        out = 0.0

        for j in range(i + 1, self.n + 1 + 1):
            out += self._bern_bas(self.n + 1, j, u)

        # Construct the integral of the basis polynomials

        return (self.hi - self.lo) / (self.n + 1) * out

    def integrate(self, a, b):
        """Integrate the Bernstein polynomial over the interval [a, b]."""
        out = 0.0

        for i, c in enumerate(self.coefs):
            if c != 0:
                _to_int = partial(self._indef_integral, i=i)
                out += float(c) * (_to_int(b) - _to_int(a))

        return out


class PyvmteResult(NamedTuple):
    """Results return class pyvmte identification and estimation.

    Attributes:
        procedure: Identification or estimation call.
        lower_bound: Lower bound of the identified set.
        upper_bound: Upper bound of the identified set.
        basis_funcs: Basis functions used for the identification.
        method: Method to solve the linear program.
        lp_api: API used to solve the linear program.
        lower_optres: Results of the optimization for the lower bound.
            Refers to second step LP for estimation.
        upper_optres: Results of the optimization for the upper bound.
            Refers to second step LP for estimation.
        lp_inputs: Inputs to the linear program.
            Refers to second step LP for estimation.
        est_u_partition: Estimated partition of the identified set. Estimation only.
        est_beta_hat: Estimated beta hat. Estimation only.
        first_minimal_deviations: Minimal deviations for the first step LP.
            Estimation only.
        first_lp_inputs: Inputs to the first step LP. Estimation only.
        first_optres: Results of the first step LP. Estimation only.

    """

    procedure: str
    success: tuple[bool, bool]
    lower_bound: float | None
    upper_bound: float | None
    target: Estimand
    identified_estimands: list[Estimand]
    basis_funcs: list[dict]
    method: str
    lp_api: str
    lower_optres: OptimizeResult
    upper_optres: OptimizeResult
    lp_inputs: dict
    restrictions: dict
    est_u_partition: np.ndarray | None = None
    est_beta_hat: np.ndarray | None = None
    first_minimal_deviations: float | None = None
    first_lp_inputs: dict | None = None
    first_optres: OptimizeResult | None = None

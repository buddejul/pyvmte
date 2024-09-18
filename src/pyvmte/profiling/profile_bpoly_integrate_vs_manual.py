"""Profile Bpoly.integrate method against manual implementation."""

import math
from functools import partial

from scipy.interpolate import BPoly  # type: ignore[import-untyped]


def _indef_integral(u: float, i: int, n: int, a: float, b: float) -> float:
    """Indefinite integral of generalized Bernstein basis i of degree n polynomial."""
    # Construct basis polynomials from i+1 to n+1
    out = 0.0

    for j in range(i + 1, n + 1 + 1):
        out += bern_bas(n + 1, j, u)

    # Construct the integral of the basis polynomials

    return (b - a) / (n + 1) * out


def bern_bas(n, v, x):
    """Bernstein polynomial basis of degree n and index v at point x."""
    return math.comb(n, v) * x**v * (1 - x) ** (n - v)


x = [0, 1]
c = [[0], [0], [1]]
bp = BPoly(c, x)

indef = partial(_indef_integral, i=2, n=2, a=0, b=1)


def integrate(a, b):
    """Integrate the Bernstein polynomial over the interval [a, b]."""
    return indef(b) - indef(a)


class Bern:
    """Bernstein polynomial of degree n with coefficients on the basis functions."""

    bfunc_type = "bernstein"

    def __init__(self, n, coefs):
        """Initialize the Bernstein polynomial with degree n and coefficients coefs."""
        self.n = n
        self.coefs = coefs

    def __call__(self, x):
        """Evaluate the Bernstein polynomial at point x."""
        return sum([bern_bas(self.n, i, x) * c for i, c in enumerate(self.coefs)])

    def integrate(self, a, b):
        """Integrate the Bernstein polynomial over the interval [a, b]."""
        return integrate(a, b)

"""Profile Bpoly.integrate method against manual implementation."""


from scipy.interpolate import BPoly  # type: ignore[import-untyped]

from pyvmte.classes import Bern

x = [0, 1]
c = [[0.5], [0.2], [0.3]]
bp = BPoly(c, x)

bern = Bern(n=2, coefs=[0.5, 0.2, 0.3])

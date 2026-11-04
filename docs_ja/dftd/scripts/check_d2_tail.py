"""Numerical check (NumPy; SymPy is not installed) of the continuum estimate of the D2
shifted-force truncation (A2, section A2.6).  Large-r limit f -> 1, phi = -c/r^6, density rho:
  E_sf - E_full (per atom) = (rho/2) [ int_0^rc 4 pi r^2 (-phi(rc) - phi'(rc)(r-rc)) dr - int_rc^oo 4 pi r^2 phi dr ]
Expected: 7 pi rho c / (3 rc^3).  Integrals by Gauss-Legendre (exact for the polynomial parts).
"""
import numpy as np

c, rho, rc = 1.7, 0.083, 9.0
phi = lambda r: -c / r ** 6
dphi = lambda r: 6 * c / r ** 7
x, w = np.polynomial.legendre.leggauss(60)
r = 0.5 * rc * (x + 1)
shift = np.sum(w * 0.5 * rc * 4 * np.pi * r ** 2 * (-phi(rc) - dphi(rc) * (r - rc)))
# tail: r = rc / t, t in (0,1], dr = -rc/t^2 dt
t = 0.5 * (x + 1)
rt = rc / t
tail = np.sum(w * 0.5 * 4 * np.pi * rt ** 2 * phi(rt) * rc / t ** 2)
dE = rho / 2 * (shift - tail)
ref = 7 * np.pi * rho * c / (3 * rc ** 3)
print(f"shift part {rho/2*shift:.6e}  sharp-cut part {-rho/2*tail:.6e}  total {dE:.6e}  7 pi rho c/(3 rc^3) = {ref:.6e}")
print(f"relative difference {abs(dE-ref)/ref:.2e}; shift part / (5 pi rho c/(3 rc^3)) = {rho/2*shift/(5*np.pi*rho*c/(3*rc**3)):.12f}")

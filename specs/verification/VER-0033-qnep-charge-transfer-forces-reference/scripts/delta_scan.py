#!/usr/bin/env python3
"""VER-0033 diagnostic: dependence of the FD residual on delta (truncation O(delta^2) vs round-off). Usage: python delta_scan.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ct_forces_reference as c
for i, (N, L, rd, rp) in [(2, (59, 14., 5., 5.)), (3, (12, 7.5, 3.6, 3.7))]:
    pos, types = c.random_config(100 + i, N, L)
    m = c.Model(7 + i, rd, rp)
    H = L * np.eye(3)
    rcK = min(4, .49 * L)
    F, W, _ = c.analytic(m, pos, H, types, rcK)
    for order, dls in ((2, (2e-5, 1e-5, 5e-6, 2.5e-6)), (4, (1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5))):
        for dl in dls:
            print(N, "order", order, dl, "virial", c.mx(W - c.fd_virial(m, pos, H, types, rcK, dl, order)), "force", c.mx(F - c.fd_forces(m, pos, H, types, rcK, dl, order)), flush=True)

"""Exact separation RH (binodal water activity of the 1:1 particle) by bisection on the number of liquids."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from aiomfac_py import Component
from aiomfac_py.phase_equilibrium import PhaseEquilibrium
from aiomfac_py.s2as import smiles_to_components
T = 290.15
systems = {"PEG400": ("OCC" + "OCC" * 8 + "O", 18.01528e-3 + 9 * 44.0526e-3, (0.90, 0.92)),
           "MBTCA": ("OC(=O)CC(C(=O)O)C(C)(C(=O)O)C", 0.20418, (0.81, 0.82))}
for name, (smi, M, (lo, hi)) in systems.items():
    C = Component(2, name, tuple(smiles_to_components([smi]).components[1].subgroups))
    pe = PhaseEquilibrium([C], ["NH4+", "SO4--"], T)
    feed = {name: 1e-3 / M, "NH4+": 2e-3 / 0.13214, "SO4--": 1e-3 / 0.13214}
    nl = lambda rh: pe.solve(feed, rh, solids="none").n_liquids
    assert nl(lo) == 2 and nl(hi) == 1, (nl(lo), nl(hi))
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if nl(mid) == 2: lo = mid
        else: hi = mid
    print(f"{name}: separation RH (binodal a_w, 1:1, 290.15 K) = {0.5*(lo+hi):.4f}")

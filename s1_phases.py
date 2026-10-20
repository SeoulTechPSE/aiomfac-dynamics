"""Phase compositions of water + SOA proxy + AS (1:1 dry mass) along a drying path (aiomfac_py PhaseEquilibrium,
solids suppressed), with AIOMFAC-VISC viscosity of each phase.  Column names follow m3a_peg_as_290K.csv (PEG -> organic)
so that the paper_6 scripts can read the file unchanged."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from aiomfac_py.phase_equilibrium import PhaseEquilibrium
from aiomfac_py.viscosity import aquelec_viscosity, aquorg_viscosity
from soa_common import *
pe = PhaseEquilibrium([ORG], ["NH4+", "SO4--"], T)
feed = {NAME: 1e-3 / M_ORG, "NH4+": 2e-3 / M_AS, "SO4--": 1e-3 / M_AS}
RHS = [0.98, 0.95, 0.92, 0.90, 0.88, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80, 0.78, 0.75, 0.72, 0.70, 0.65, 0.60,
       0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
def props(L):
    n = dict(zip(L.names, L.amounts))
    m = np.array([n["Water"] * MW, n[NAME] * M_ORG, n["SO4--"] * M_AS]); w = m / m.sum()
    r = MODEL.evaluate(list(w), T, basis="mass")
    eo = aquorg_viscosity(MODEL, r, T, {2: ETA0}).eta_pas; ee = aquelec_viscosity(MODEL, r, T, {2: ETA0}).eta_pas
    return w, eo, ee, m.sum()
rows, prev = [], None
for rh in RHS:
    res = pe.solve(feed, rh, solids="none", init=prev); prev = res if res.status == "converged" else None
    rec = {"RH": rh, "status": res.status, "n_liquids": res.n_liquids}
    ph = sorted([props(L) for L in res.liquids], key=lambda p: -p[0][1])
    for tag, p in zip(("org", "aq") if len(ph) == 2 else ("single",), ph):
        w, eo, ee, mass = p
        rec.update({f"{tag}_w_water": w[0], f"{tag}_w_PEG": w[1], f"{tag}_w_AS": w[2], f"{tag}_eta_aquorg": eo,
                    f"{tag}_eta_aquelec": ee, f"{tag}_mass_mg": mass * 1e6})
    rows.append(rec)
    print(f"RH {rh:.2f} L{res.n_liquids} " + "  ".join(f"{k}={v:.3g}" for k, v in rec.items()
          if k.endswith(("w_water", "w_PEG", "w_AS", "eta_aquorg", "mass_mg"))), flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/m3a_peg_as_290K.csv", index=False)
print(NAME, f"C{nC}H{nH}O{nO} M {M_ORG*1e3:.1f} Tg {TG:.1f} eta0 {ETA0:.3g}")

"""Exchange chemical potentials (M3e convention), mixing Gibbs energy and AIOMFAC-VISC viscosity of water + SOA proxy
+ AS on a mass-fraction grid down to w_water = 0.03 (needed for drying to RH ~0.2).
  mu_P = ln a_P / M_P - ln a_w / M_w,  mu_S = (2 ln a_NH4 + ln a_SO4)/M_AS - ln a_w / M_w   (per kg, RT units)
  log10 eta = x_org log10 eta_aquorg + (1 - x_org) log10 eta_aquelec  (as m3f_visc_table.py), for several eta0.
Output: results/<NAME>/tables_290K.npz"""
import warnings, time; warnings.filterwarnings("ignore")
import numpy as np
from aiomfac_py.phase_equilibrium import ExplicitLiquidModel
from aiomfac_py.viscosity import aquelec_viscosity, aquorg_viscosity
from soa_common import *
lm = ExplicitLiquidModel([ORG], ["NH4+", "SO4--"]); lm.set_conditions(T, np.log(0.9))
iW, iP, iN, iS = (lm.names.index(k) for k in ("Water", NAME, "NH4+", "SO4--"))
G = 193
wp = np.linspace(1e-6, 0.96, G); ws = np.linspace(1e-6, 0.96, G)
P, S = np.meshgrid(wp, ws, indexing="ij"); Wt = 1 - P - S
ok = Wt >= 0.03
ETA0S = [ETA0, 1e12, 1e14, 1e16]
EP, ES, AW, GG = (np.full(P.shape, np.nan) for _ in range(4)); L10 = np.full((len(ETA0S),) + P.shape, np.nan)
t0 = time.time()
for a, b in zip(*np.nonzero(ok)):
    n = np.zeros(lm.N); n[iW] = Wt[a, b] / MW; n[iP] = P[a, b] / M_ORG; n[iN] = 2 * S[a, b] / M_AS; n[iS] = S[a, b] / M_AS
    la = lm.ln_a(n, T) - lm._c
    AW[a, b] = la[iW]
    EP[a, b] = la[iP] / M_ORG - la[iW] / MW
    ES[a, b] = (2 * la[iN] + la[iS]) / M_AS - la[iW] / MW
    GG[a, b] = n[iW] * la[iW] + n[iP] * la[iP] + n[iN] * la[iN] + n[iS] * la[iS]
print(f"activities {time.time()-t0:.0f}s", flush=True)
step = 2
for a in range(0, G, step):
    for b in range(0, G, step):
        if not ok[a, b]: continue
        w = [Wt[a, b], P[a, b], S[a, b]]
        r = MODEL.evaluate(w, T, basis="mass")
        nP, nI = w[1] / M_ORG, 3 * w[2] / M_AS; xo = nP / (nP + nI)
        for k, e0 in enumerate(ETA0S):
            eo = aquorg_viscosity(MODEL, r, T, {2: e0}).eta_pas; ee = aquelec_viscosity(MODEL, r, T, {2: e0}).eta_pas
            L10[k, a, b] = xo * np.log10(eo) + (1 - xo) * np.log10(ee)
    if a % 20 == 0: print(f"visc row {a} ({time.time()-t0:.0f}s)", flush=True)
np.savez(f"{OUT}/tables_290K.npz", wp=wp, ws=ws, mu_P=EP, mu_S=ES, lnaw=AW, g=GG,
         visc_wp=wp[::step], visc_ws=ws[::step], log10eta=L10[:, ::step, ::step], eta0=np.array(ETA0S),
         M_PEG=M_ORG, M_AS=M_AS, MW=MW, name=NAME, Tg=TG)
print("done", np.isfinite(EP[ok]).all(), np.nanmin(L10), np.nanmax(L10))

"""Smooth excess Gibbs energy for water + SOA proxy + AS (extends m3f_gex_fit.py to low water content).
g = sum n_i ln a_i (per kg, RT) is split into
  G_id = p(ln p - 1)/M_P + 3 s(ln s - 1)/M_S + w(ln w - 1)/M_W,   w = 1 - p - s
(the water term carries the ln w_w singularity of ln a_w; m3f_gex_fit.py, fitted at w_w >= 0.25, folded it into
the polynomial) and G_ex = sum c_k p^i s^e, e in {j, j+1/2}, fitted mainly to its derivatives (Sobolev LSQ).
Checks: tie lines of the fit vs the AIOMFAC PhaseEquilibrium compositions (s1_phases.py) for RH 0.81-0.20."""
import sys
import numpy as np, pandas as pd
from scipy.optimize import fsolve
from soa_common import OUT
DEG = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WMIN_W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
Tb = np.load(f"{OUT}/tables_290K.npz")
wp, ws = Tb["wp"], Tb["ws"]; MP, MS, MW = float(Tb["M_PEG"]), float(Tb["M_AS"]), float(Tb["MW"])
P, S = np.meshgrid(wp, ws, indexing="ij"); Wt = 1 - P - S
with np.errstate(all="ignore"):
    Gid = P * (np.log(P) - 1) / MP + 3 * S * (np.log(S) - 1) / MS + Wt * (np.log(Wt) - 1) / MW
    EP = Tb["mu_P"] - (np.log(P) / MP - np.log(Wt) / MW)
    ES = Tb["mu_S"] - (3 * np.log(S) / MS - np.log(Wt) / MW)
    Gex = Tb["g"] - Gid
m = np.isfinite(Gex) & (Wt >= WMIN_W) & (S >= 0.01) & (S <= 0.93) & (P <= 0.80)
EXP = [(i, j + 0.5 * h) for i in range(DEG + 1) for j in range(DEG + 1 - i) for h in (0, 1) if not (i == 0 and j + h == 0)]
def basis(p, s):
    p, s = np.atleast_1d(p), np.atleast_1d(s)
    B = np.array([p ** i * s ** e for i, e in EXP]).T
    BP = np.array([i * p ** max(i - 1, 0) * s ** e if i else 0 * p for i, e in EXP]).T
    BS = np.array([e * p ** i * s ** (e - 1) if e else 0 * p for i, e in EXP]).T
    return B, BP, BS
B, BP, BS = basis(P[m], S[m])
sc = np.abs(np.vstack([BP, BS])).max(0)                     # column scaling for conditioning
A = np.vstack([BP, BS, 0.05 * B]) / sc; y = np.concatenate([EP[m], ES[m], 0.05 * Gex[m]])
coef = np.linalg.lstsq(A, y, rcond=None)[0] / sc
rP, rS = BP @ coef - EP[m], BS @ coef - ES[m]
print(f"degree {DEG}, {len(EXP)} terms, w_w >= {WMIN_W}: E_P rms {np.sqrt(np.mean(rP**2)):.3g} max {abs(rP).max():.3g}; "
      f"E_S rms {np.sqrt(np.mean(rS**2)):.3g} max {abs(rS).max():.3g}; |E| scale {np.median(abs(EP[m])):.3g}/{np.median(abs(ES[m])):.3g}")
def G(p, s):
    b, bp, bs = basis(p, s); w = 1 - p - s
    return (float((b @ coef)[0]) + p * (np.log(p) - 1) / MP + 3 * s * (np.log(s) - 1) / MS + w * (np.log(w) - 1) / MW,
            float((bp @ coef)[0]) + np.log(p) / MP - np.log(w) / MW, float((bs @ coef)[0]) + 3 * np.log(s) / MS - np.log(w) / MW)
def lnaw(p, s):
    g, gp, gs = G(p, s); return MW * (g - p * gp - s * gs)
M3A = pd.read_csv(f"{OUT}/m3a_peg_as_290K.csv").set_index("RH")
ties, worst = [], 0
for rh in [0.81, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20]:
    r = M3A.loc[rh]
    x0 = [r.org_w_PEG, r.org_w_AS, max(r.aq_w_PEG, 1e-7), r.aq_w_AS]
    def F(x):
        a, b = G(x[0], x[1]), G(x[2], x[3])
        return [a[1] - b[1], a[2] - b[2], (a[0] - x[0] * a[1] - x[1] * a[2]) - (b[0] - x[2] * b[1] - x[3] * b[2]),
                lnaw(x[0], x[1]) - np.log(rh)]
    x, info, ier, msg = fsolve(F, x0, full_output=True)
    err = max(abs(x[0] - x0[0]), abs(x[1] - x0[1]), abs(x[3] - x0[3])); worst = max(worst, err) if rh <= 0.80 else worst
    ties.append([rh, *x0, *x])
    print(f"RH {rh}: AIOMFAC org {x0[0]:.4f}/{x0[1]:.4f} aq {x0[2]:.2g}/{x0[3]:.4f} | fit org {x[0]:.4f}/{x[1]:.4f} "
          f"aq {x[2]:.2g}/{x[3]:.4f} {'ok' if ier == 1 else msg[:30]}  ln aw(AIOMFAC pts) org {lnaw(x0[0],x0[1])-np.log(rh):+.4f} aq {lnaw(x0[2],x0[3])-np.log(rh):+.4f}")
print("max tie-line deviation (RH<=0.80):", f"{worst:.4f}")
np.savez(f"{OUT}/gex_fit_290K.npz", coef=coef, i=np.array([e[0] for e in EXP]), e=np.array([e[1] for e in EXP]),
         M_PEG=MP, M_AS=MS, MW=MW, water_ideal=1, ties=np.array(ties), deg=DEG)

"""Literature water diffusivities vs the fractional Stokes-Einstein water mobility of the three-mobility model, at 290.15 K.
Writes results/water_diffusion/dw_compare.csv and figures/fig8_water_diffusion.png.

Model side: D_W = D_w,pure(T) (eta_w(T)/eta)^xi_w on the equilibrium MBTCA-rich phase (AIOMFAC phase table, AIOMFAC-VISC
eta of results/MBTCA/tables_290K.npz). Literature side, as a function of a_w (= RH for the equilibrium phase):
  Lienhard 2015 3-MBTCA       binary MBTCA-water, x_w(a_w) from AIOMFAC
  Lienhard 2015 a-pinene SOA  x_w(a_w) from kappa-Koehler (kappa 0.1, 1400 kg m-3, 150 g mol-1: assumptions)
  Price 2015 a-pinene SOM     reported ~2 orders above the Lienhard SOA values (Lienhard et al. 2015): shown as x100
  Price 2014 raffinose/M5AS   organic acids + 7.7 % ammonium sulfate, sucrose: cubic fits, shifted from 296.65 K
The implied xi_w at each a_w is ln(D_pure/D_lit)/ln(eta/eta_w) with the model viscosity of the MBTCA-rich phase."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

import wdiff  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "water_diffusion"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures")
T = 290.15
Dp, ew = wdiff.pure_water(T), wdiff.water_viscosity(T)

# --- MBTCA-water binary: x_w(a_w) from AIOMFAC
from soa_common import MODEL, M_ORG, MW  # noqa: E402
ww = np.r_[np.linspace(0.005, 0.1, 40), np.linspace(0.11, 0.9, 60)]
aw_b, xw_b = [], []
for w in ww:
    r = MODEL.evaluate([w, 1 - w, 1e-10], T, basis="mass")
    aw_b.append(float(r.activity[0])); xw_b.append((w / MW) / (w / MW + (1 - w) / M_ORG))
aw_b, xw_b = np.array(aw_b), np.array(xw_b)
k = np.argsort(aw_b); aw_b, xw_b = aw_b[k], xw_b[k]

# --- equilibrium MBTCA-rich (or single) phase along RH, and its model viscosity
SP = pd.read_csv(os.path.join(HERE, "results/MBTCA/m3a_peg_as_290K.csv"))
VT = np.load(os.path.join(HERE, "results/MBTCA/tables_290K.npz"))
gp, gs = VT["visc_wp"], VT["visc_ws"]


def eta_model(p, s, idx):
    L10 = np.array(VT["log10eta"][idx], float); L10 = np.where(np.isfinite(L10), L10, np.nanmax(L10))
    return 10 ** RegularGridInterpolator((gp, gs), L10, bounds_error=False, fill_value=None)(np.c_[p, s])


rows = []
for _, r in SP.iterrows():
    tag = "org" if r.n_liquids == 2 else "single"
    p, s = r[f"{tag}_w_PEG"], r[f"{tag}_w_AS"]
    rec = {"a_w": r.RH, "phase": tag, "w_MBTCA": p, "w_AS": s, "w_water": r[f"{tag}_w_water"]}
    for idx, e0 in enumerate(VT["eta0"]):
        rec[f"eta_e{idx}"] = float(eta_model(np.array([p]), np.array([s]), idx)[0])
    rows.append(rec)
df = pd.DataFrame(rows).sort_values("a_w").reset_index(drop=True)
aw = df.a_w.values

# --- literature D_w at the same a_w
xw_mb = np.interp(aw, aw_b, xw_b)
lit = {"Lienhard 2015, 3-MBTCA": wdiff.lienhard("3-MBTCA", T, xw_mb)}
kap, rho_s, M_s = 0.1, 1400.0, 0.150
vw_vs = kap * aw / (1 - aw)                                   # kappa-Koehler: V_w/V_s
xw_soa = (vw_vs * 1000 / MW) / (vw_vs * 1000 / MW + rho_s / M_s)
lit["Lienhard 2015, a-pinene SOA"] = wdiff.lienhard("alpha-pinene SOA", T, xw_soa)
lit["Price 2015, a-pinene SOM (~x100)"] = 100 * lit["Lienhard 2015, a-pinene SOA"]
lit["Price 2014, raffinose/M5AS (7.7 % AS)"] = wdiff.price2014("raffinose/M5AS", aw, T)
lit["Price 2014, sucrose"] = wdiff.price2014("sucrose", aw, T)
for name, D in lit.items():
    df[f"Dw[{name}]"] = np.minimum(D, Dp)
    df[f"xi_w[{name}]"] = np.log(Dp / np.minimum(D, Dp)) / np.log(df.eta_e0 / ew)
df.to_csv(os.path.join(OUT, "dw_compare.csv"), index=False)

# --- step 2: one xi_w per data set, least squares in ln D over a_w 0.2-0.8 (eta0 6e10 and 1e16 model viscosities)
fit = []
m = (df.a_w >= 0.2) & (df.a_w <= 0.80)
for name, D in lit.items():
    if "x100" in name:          # only a low-a_w bracket; excluded where it hits the pure-water cap
        m2 = m & (np.minimum(D, Dp) < 0.5 * Dp)
    else:
        m2 = m
    y = np.log(Dp / np.minimum(D, Dp))[m2]
    rec = {"data": name}
    for idx in (0, 3):
        x = np.log(df[f"eta_e{idx}"].values / ew)[m2]
        xi = float(np.sum(x * y) / np.sum(x * x))              # D = Dp (ew/eta)^xi through the pure-water anchor
        rec[f"xi_w_e{idx}"] = xi
        rec[f"rms_log10_e{idx}"] = float(np.sqrt(np.mean((xi * x - y) ** 2)) / np.log(10))
    fit.append(rec)
fit = pd.DataFrame(fit)
fit.to_csv(os.path.join(OUT, "xi_w_fit.csv"), index=False)
print(fit.to_string(float_format=lambda v: f"{v:.2f}"))

# --- step 3 input: D_w(a_w) tables on a fine a_w grid (sph1d.py --dw_model reads this)
ag = np.linspace(0.0, 1.0, 1001)
agc = np.clip(ag, aw_b[0], aw_b[-1])
tab = {"a_w": ag,
       "mbtca": np.minimum(wdiff.lienhard("3-MBTCA", T, np.interp(agc, aw_b, xw_b)), Dp),
       "m5as": np.minimum(wdiff.price2014("raffinose/M5AS", ag, T), Dp)}
vws = kap * np.clip(ag, 0, 0.999) / (1 - np.clip(ag, 0, 0.999))
xs = (vws * 1000 / MW) / (vws * 1000 / MW + rho_s / M_s)
tab["soa_lienhard"] = np.minimum(wdiff.lienhard("alpha-pinene SOA", T, xs), Dp)
tab["som_price"] = np.minimum(100 * tab["soa_lienhard"], Dp)
tab["a_w"][-1] = 1.0
for k_ in ("mbtca", "m5as", "soa_lienhard", "som_price"):
    tab[k_][-1] = Dp
pd.DataFrame(tab).to_csv(os.path.join(OUT, "dw_aw_290K.csv"), index=False)

# --- summary: implied xi_w over the a_w range where the droplet is phase-separated or dry (a_w <= 0.80)
sel = df.a_w <= 0.80
print(f"T {T} K: D_w,pure {Dp:.3e} m2/s, eta_w {ew:.3e} Pa s")
for name in lit:
    v = df.loc[sel, f"xi_w[{name}]"]
    print(f"{name:40s} implied xi_w (eta0 6e10, a_w 0.2-0.8): median {v.median():.2f}, range {v.min():.2f}-{v.max():.2f}")

# --- figure
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for i, (name, D) in enumerate(lit.items()):
    ax[0].semilogy(aw, np.minimum(D, Dp), color=C[i], lw=1.8, ls="--" if "x100" in name else "-", label=name)
for xi, lsty in ((0.5, ":"), (0.75, "-."), (1.0, (0, (1, 3)))):
    ax[0].semilogy(aw, Dp * (ew / df.eta_e0) ** xi, color="#8a8984", lw=1.2, ls=lsty, label=f"model, ξ$_w$ = {xi:g} (η$_0$ 6e10)")
ax[0].set_xlabel("water activity"); ax[0].set_ylabel("D$_w$ (m$^2$ s$^{-1}$)"); ax[0].legend(fontsize=6.5, frameon=False)
ax[0].set_title("Water diffusivity at 290 K, MBTCA-rich phase", fontsize=9)
for i, name in enumerate(lit):
    ax[1].plot(aw[sel], df.loc[sel, f"xi_w[{name}]"], color=C[i], lw=1.8, ls="--" if "x100" in name else "-")
ax[1].set_xlabel("water activity"); ax[1].set_ylabel("implied ξ$_w$ (η$_0$ 6e10)")
ax[1].set_title("ξ$_w$ that matches each data set", fontsize=9); ax[1].set_ylim(0, 1.1)
for a_ in ax:
    a_.grid(True, color="#e4e3df", lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig8_water_diffusion.png"), dpi=200); plt.close()

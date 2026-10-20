"""Figures and summary table for the follow-up runs (PEG 3D baseline, MBTCA + AS drying with viscosity-dependent mobility)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "runs")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # fixed order
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "legend.frameon": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "savefig.dpi": 200, "figure.dpi": 110})


def load(tag):
    f = f"{RUN}/{tag}_hist.csv"
    if not os.path.exists(f):
        return None
    return (pd.read_csv(f), np.load(f"{RUN}/{tag}.npz"), json.load(open(f"{RUN}/{tag}_meta.json")))


rows = []
# ---------------- PEG-400/AS: 3D vs 2D ----------------
PEGD = os.environ.get("M3F_DATA", os.path.join(HERE, "m3f_bundle", "data"))
ph = pd.read_csv(f"{PEGD}/m3a_peg_as_290K.csv").set_index("RH")
m_ratio = (ph.loc[0.80].org_mass_mg + ph.loc[0.80].aq_mass_mg) / ph.loc[0.92].single_mass_mg
print(f"PEG equilibrium mass ratio RH 0.80/0.92 = {m_ratio:.4f}: R_f 2D (sqrt) {m_ratio**0.5:.4f}, 3D (cbrt) {m_ratio**(1/3):.4f}")


def onset(h, meta, w0):
    """first recorded time at which an aqueous (PEG-depleted) region appears: w_P,min < (w0 + w_aq,ref)/2.
    Returns (row index, time, previous snapshot time); the snapshot spacing (~12 %) is the resolution of the onset time."""
    thr = 0.5 * (w0 + meta["aq_ref"][0])
    k = np.nonzero(h.wP_min.values < thr)[0]
    if not len(k):
        return None, np.nan, np.nan
    k = k[0]
    return k, h.t_s.values[k], h.t_s.values[k - 1] if k > 0 else np.nan


def onset_X(z, meta, t):
    """X = r/R of the w_P minimum in the snapshot closest to t (where the aqueous phase first appears)"""
    j = np.argmin(abs(z["t"] * meta["t0_s"] - t))
    return float(z["X"][np.argmin(z["wP"][j])])


def t_dry(h, frac=0.95):
    """time to reach frac of the total radius change"""
    R = h.R.values; tgt = R[0] - frac * (R[0] - R[-1])
    return float(np.interp(-tgt, -R, h.t_s.values))


for tag in sorted(glob.glob(f"{RUN}/peg_*_hist.csv")):
    tag = os.path.basename(tag)[:-9]
    h, z, m = load(tag)
    k, t_on, t_on_prev = onset(h, m, m["w0"][0])
    rows.append({"case": tag, "R_final": h.R.iloc[-1], "onset_s": t_on, "onset_prev_snap_s": t_on_prev,
                 "onset_aw_mean": h.aw_mean.values[k] if k is not None else np.nan,
                 "onset_X": onset_X(z, m, t_on) if k is not None else np.nan,
                 "Pe_evap0": h.Pe_evap.iloc[0], "surf_wP_end": h.wP_edge.iloc[-1], "t95_s": t_dry(h),
                 "layers_final": int(h.peg_layers.iloc[-1]), "mass_err": max(abs(h.PEG_rel - 1).max(), abs(h.AS_rel - 1).max()),
                 "steps": m["steps"], "status": m.get("status", "")})

r_s, r_c = load("peg_sph_D1e-10"), load("peg_cyl_D1e-10")
if r_s and r_c:
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
    for (h, z, m), lab, c in ((r_c, "disc (2D, as paper)", C[0]), (r_s, "sphere (3D)", C[1])):
        ax[0].plot(h.t_s, h.R, color=c, label=lab)
        ax[1].plot(h.t_s, h.aw_surf, color=c, label=lab + " surface")
        ax[1].plot(h.t_s, h.aw_mean, color=c, ls="--", lw=1.2, label=lab + " mean")
    ax[0].axhline(m_ratio ** 0.5, color=C[0], lw=0.8, ls=":"); ax[0].axhline(m_ratio ** (1 / 3), color=C[1], lw=0.8, ls=":")
    for a_ in ax[:2]:
        for ln in a_.lines:
            x, y = ln.get_data(); k = np.asarray(x) > 0; ln.set_data(np.asarray(x)[k], np.asarray(y)[k])
    ax[0].set_xscale("log"); ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("R / R$_0$"); ax[0].legend(fontsize=7)
    ax[0].set_title("Radius (dotted: equilibrium)", fontsize=9)
    ax[1].set_xscale("log"); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("water activity"); ax[1].legend(fontsize=6)
    ax[1].set_title("a$_w$ (RH 0.92 → 0.80)", fontsize=9)
    h, z, m = r_s
    sel = [np.argmin(abs(z["t"] * m["t0_s"] - ts)) for ts in (0.1, 0.2, 0.3, 0.5, 1.0, z["t"][-1] * m["t0_s"])]
    cm = plt.get_cmap("Blues")
    for i, k in enumerate(sel):
        ax[2].plot(z["X"] * z["R"][k], z["wP"][k], color=cm(0.35 + 0.65 * i / (len(sel) - 1)), lw=1.3,
                   label=f"{z['t'][k] * m['t0_s']:.2g} s")
    ax[2].set_xlabel("r / R$_0$"); ax[2].set_ylabel("w$_{PEG}$")
    ax[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="t", title_fontsize=7)
    ax[2].set_title("Sphere: PEG profile", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig1_peg_3d_vs_2d.png"); plt.close()

# ---------------- MBTCA/AS ----------------
SP = pd.read_csv(os.path.join(HERE, "results/MBTCA/m3a_peg_as_290K.csv")).set_index("RH")


def eq_water(rh):
    r = SP.loc[rh]
    if r.n_liquids == 2:
        mo, ma = r.org_mass_mg, r.aq_mass_mg
        return (mo * r.org_w_water + ma * r.aq_w_water) / (mo + ma), (mo + ma) / SP.loc[0.85].single_mass_mg
    return r.single_w_water, r.single_mass_mg / SP.loc[0.85].single_mass_mg


def t_equil(h, rh, tol=0.01):
    """time after which the volume-mean a_w stays within tol of RH"""
    bad = np.nonzero(np.abs(h.aw_mean.values - rh) > tol)[0]
    if len(bad) == 0:
        return 0.0
    if bad[-1] == len(h) - 1:
        return np.inf
    return h.t_s.values[bad[-1] + 1]


def skin(z, k, lim=-2.0):
    """thickness (units R0) of the outer region with mobility factor below 10^lim"""
    lf = np.log10(z["fac"][k]); X = z["X"]
    inside = np.nonzero(lf >= lim)[0]
    if len(inside) == 0:
        return z["R"][k]
    j = inside[-1]
    return (1 - X[j]) * z["R"][k] if j < len(X) - 1 else 0.0


soa = {}
for f in sorted(glob.glob(f"{RUN}/soa_rh*_hist.csv")):
    tag = os.path.basename(f)[:-9]
    h, z, m = load(tag)
    soa[tag] = (h, z, m, json.loads(str(z["args"])))
RINF = {}       # end state = each run's own final radius (see the note on interface-held water in the summary)
RCONST = {a["rh"]: h.R.iloc[-1] for tag, (h, z, m, a) in soa.items() if a["visc"] == 0 and a["ramp_s"] == 0}


def fw(h, rinf):
    """fraction of the evaporable water still in the particle"""
    return (h.R.values ** 3 - rinf ** 3) / (1 - rinf ** 3)


def t_frac(h, rinf, f, snap=False):
    """time at which f_w first reaches f, log-log interpolated between the bracketing snapshots
    (snap=True: the first snapshot at or below f, as in the first version; biased high by up to one snapshot interval)"""
    t = h.t_s.values; v = fw(h, rinf); k = np.nonzero(v <= f)[0]
    if not len(k):
        return np.inf
    k = k[0]
    if snap or k == 0:
        return t[k]
    if v[k] > 0 and t[k - 1] > 0:
        return float(np.exp(np.interp(np.log(f), [np.log(v[k]), np.log(v[k - 1])], [np.log(t[k]), np.log(t[k - 1])])))
    return float(t[k - 1] + (t[k] - t[k - 1]) * (v[k - 1] - f) / (v[k - 1] - v[k]))


def t_cross(t, y, level):
    """first time y rises to level, linearly interpolated (NaN if never)"""
    k = np.nonzero(y >= level)[0]
    if not len(k):
        return np.nan
    k = k[0]
    return t[k] if k == 0 else float(np.interp(level, [y[k - 1], y[k]], [t[k - 1], t[k]]))


def shell_scaling(h, z, m, a, rinf, f_ref=0.05, f_lo=0.005, f_hi=0.1):
    """late decay time tau vs diffusion time of the outer organic-rich shell.
    tau: e-folding of f_w fitted on f_lo <= f_w <= f_hi.  L: surface to the first AS-rich node (w_P below the
    midpoint of the organic/aqueous reference compositions) in the snapshot closest to f_w = f_ref.
    D_shell = D_ref * <factor>; layers are in series, so the length-weighted harmonic mean is the physical average
    (the geometric mean is given for comparison with the first draft of the doc)."""
    t = h.t_s.values; v = fw(h, rinf)
    k = (v >= f_lo) & (v <= f_hi)
    if k.sum() < 3:
        return None
    tau = -1.0 / np.polyfit(t[k], np.log(v[k]), 1)[0]
    vz = (z["R"] ** 3 - rinf ** 3) / (1 - rinf ** 3)
    j = int(np.argmin(abs(np.log(np.maximum(vz, 1e-12)) - np.log(f_ref))))
    r = z["X"] * z["R"][j]; wP = z["wP"][j]; fac = z["fac"][j]
    thr = 0.5 * (m["org_ref"][0] + m["aq_ref"][0])
    aq = np.nonzero(wP < thr)[0]
    if not len(aq):
        return None
    sh = slice(aq[-1] + 1, None)
    dr = np.gradient(r)[sh]
    har = dr.sum() / np.sum(dr / fac[sh])
    geo = 10 ** np.mean(np.log10(fac[sh]))
    L = (r[-1] - r[aq[-1]]) * a["R0_um"] * 1e-6
    t_diff_h = L ** 2 / (a["D"] * har); t_diff_g = L ** 2 / (a["D"] * geo)
    # whole particle: R^2 / (pi^2 D_ref <factor>_harm over 0..R); unlike L, independent of kappa (interface width)
    drA = np.gradient(r); harA = drA.sum() / np.sum(drA / fac)
    t_part = (r[-1] * a["R0_um"] * 1e-6) ** 2 / (np.pi ** 2 * a["D"] * harA)
    return {"t_snap_s": z["t"][j] * m["t0_s"], "fw_snap": vz[j], "L_um": L * 1e6, "fac_harm": har, "fac_geo": geo,
            "L2_D_harm_s": t_diff_h, "L2_D_geo_s": t_diff_g, "tau_s": tau,
            "tau_over_L2D_harm": tau / t_diff_h, "tau_over_L2D_geo": tau / t_diff_g,
            "fac_harm_particle": harA, "R2_pi2D_particle_s": t_part, "tau_over_R2pi2D": tau / t_part}


for tag, (h, z, m, a) in soa.items():
    rh = a["rh"]; ww_eq, mr = eq_water(rh)
    rinf = h.R.iloc[-1]
    t1 = np.argmin(abs(h.t_s - 1.0)); t10 = np.argmin(abs(h.t_s - 10.0))
    rows.append({"case": tag, "RH": rh, "xi": a["visc"], "eta_idx": a["eta_idx"], "wet": a["wet"],
                 "R_final": h.R.iloc[-1], "R_eq_AIOMFAC": mr ** (1 / 3), "R_inf_model": rinf,
                 "fw_floor_vs_constD": ((h.R.iloc[-1] ** 3 - RCONST[rh] ** 3) / (1 - RCONST[rh] ** 3)) if rh in RCONST else np.nan,
                 "t90_s": t_frac(h, rinf, 0.10), "t99_s": t_frac(h, rinf, 0.01),
                 "t90_snap_s": t_frac(h, rinf, 0.10, snap=True), "t99_snap_s": t_frac(h, rinf, 0.01, snap=True),
                 "fw_1s": float(np.interp(1.0, h.t_s, fw(h, rinf))), "fw_10s": float(np.interp(10.0, h.t_s, fw(h, rinf))), "t_surf_s": t_equil(h.assign(aw_mean=h.aw_surf), rh),
                 "t_surf_002_s": t_equil(h.assign(aw_mean=h.aw_surf), rh, tol=0.02),
                 "aw_mean_2s": float(np.interp(2.0, h.t_s, h.aw_mean)), "aw_mean_10s": float(np.interp(10.0, h.t_s, h.aw_mean)),
                 "t_first_snap_s": h.t_s.values[1], "t_surf_wP071_s": t_cross(h.t_s.values, h.wP_edge.values, 0.71),
                 "surf_wP_max": h.wP_edge.max(), "surf_wP_20s": float(np.interp(20.0, h.t_s, h.wP_edge)),
                 "t_mean_s": t_equil(h, rh), "excess_water_1s": h.wW_mean.iloc[t1] - ww_eq,
                 "excess_water_10s": h.wW_mean.iloc[t10] - ww_eq, "excess_water_end": h.wW_mean.iloc[-1] - ww_eq,
                 "logfac_min": h.logfac_min.min(), "skin_R0_end": skin(z, -1), "layers_final": int(h.peg_layers.iloc[-1]),
                 "mass_err": max(abs(h.PEG_rel - 1).max(), abs(h.AS_rel - 1).max()), "steps": m["steps"],
                 "status": m.get("status", "")})

# late-time scaling tau vs L^2/D_shell (step runs with viscosity-dependent mobility)
scal = []
for tag, (h, z, m, a) in soa.items():
    if a["visc"] > 0 and a["ramp_s"] == 0:
        s = shell_scaling(h, z, m, a, h.R.iloc[-1])
        if s:
            scal.append({"case": tag, "RH": a["rh"], "xi": a["visc"], "eta_idx": a["eta_idx"], "wet": a["wet"], **s})
scal = pd.DataFrame(scal)
scal.to_csv(f"{FIG}/scaling.csv", index=False)

ETA0 =np.load(os.path.join(HERE, "results/MBTCA/tables_290K.npz"))["eta0"]


def panel_aw(ax, tags, labels, title):
    for i, (t, lab) in enumerate(zip(tags, labels)):
        if t not in soa:
            continue
        h, z, m, a = soa[t]
        h = h[h.t_s > 0]
        ax.plot(h.t_s, np.maximum(fw(h, h.R.iloc[-1]), 1e-4), color=C[i], label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2, None); ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("t (s)"); ax.set_ylabel("evaporable water remaining")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7)


fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
panel_aw(ax[0], ["soa_rh0.30_xi0", "soa_rh0.30_xi0.5_e0", "soa_rh0.30_xi1_e0"],
         ["constant D", "ξ = 0.5", "ξ = 1 (Stokes–Einstein)"], "Mobility law (RH 0.85 → 0.30)")
panel_aw(ax[1], ["soa_rh0.30_xi1_e0", "soa_rh0.30_xi1_e1", "soa_rh0.30_xi1_e2", "soa_rh0.30_xi1_e3"],
         [f"η$_0$ = {e:.0e} Pa s" for e in ETA0], "Dry viscosity η$_0$ (ξ = 1)")
panel_aw(ax[2], ["soa_rh0.80_xi1_e0", "soa_rh0.50_xi1_e0", "soa_rh0.30_xi1_e0", "soa_rh0.30_xi1_e0_ramp60"],
         ["RH 0.80", "RH 0.50", "RH 0.30", "RH 0.30, 60 s ramp"], "Final RH (ξ = 1)")
for a_ in ax:
    a_.axhline(0.01, color="#8a8984", lw=0.7, ls=":")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_soa_water_retention.png"); plt.close()

# sensitivity: interface width (kappa x0.5, x2; fine mesh, + noise seed) and the eta_ref normalisation across eta0
if any(t.endswith("_k2") for t in soa):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for a_, e in zip(ax[:2], ("e0", "e3")):
        tags = [f"soa_rh0.30_xi1_{e}_fine_k0.5", f"soa_rh0.30_xi1_{e}_fine", f"soa_rh0.30_xi1_{e}_fine_k2"]
        labs = ["κ × 0.5", "κ × 1", "κ × 2"]
        if e == "e0":
            tags.append("soa_rh0.30_xi1_e0_fine_s2"); labs.append("κ × 1, seed 2")
        panel_aw(a_, tags, labs, f"Interface width, η$_0$ = {ETA0[int(e[1])]:.0e} Pa s (n 1200)")
    panel_aw(ax[2], ["soa_rh0.30_xi1_e0"] + [f"soa_rh0.30_xi1_e{i}{s}" for i in (1, 3) for s in ("", "_er")],
             ["η$_0$ 6e10", "1e12, η$_{ref}$ own", "1e12, η$_{ref}$ fixed", "1e16, η$_{ref}$ own", "1e16, η$_{ref}$ fixed"],
             "D·η fixed (η$_{ref}$ 0.0086) vs own η$_{ref}$")
    for a_ in ax:
        a_.axhline(0.01, color="#8a8984", lw=0.7, ls=":")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig6_sensitivity.png"); plt.close()


def profiles(tag, fname, times=(0.05, 0.2, 1, 5, 20, None)):
    if tag not in soa:
        return
    h, z, m, a = soa[tag]
    t_s = z["t"] * m["t0_s"]
    sel = [np.argmin(abs(t_s - (ts if ts else t_s[-1]))) for ts in times]
    sel = sorted(set(sel))
    cm = plt.get_cmap("Blues")
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.0))
    for i, k in enumerate(sel):
        r = z["X"] * z["R"][k]; col = cm(0.35 + 0.65 * i / max(len(sel) - 1, 1)); lab = f"{t_s[k]:.3g} s"
        ax[0].plot(r, z["wP"][k], color=col, lw=1.2, label=lab)
        ax[1].plot(r, 1 - z["wP"][k] - z["wS"][k], color=col, lw=1.2)
        ax[2].plot(r, z["aw"][k], color=col, lw=1.2)
        ax[3].plot(r, np.log10(z["fac"][k]), color=col, lw=1.2)
    for a_, yl in zip(ax, ["w$_{MBTCA}$", "w$_{water}$", "local a$_w$", "log$_{10}$ mobility factor (η$_{ref}$/η)$^ξ$"]):
        a_.set_xlabel("r / R$_0$"); a_.set_ylabel(yl)
    ax[2].axhline(a["rh"], color="#8a8984", lw=0.8, ls="--")
    ax[3].legend(*ax[0].get_legend_handles_labels(), fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="t", title_fontsize=7)
    fig.suptitle(f"{tag}", fontsize=9); plt.tight_layout(); plt.savefig(f"{FIG}/{fname}"); plt.close()


profiles("soa_rh0.30_xi1_e0", "fig3_profiles_rh030_xi1_e0.png")
profiles("soa_rh0.30_xi1_e3", "fig4_profiles_rh030_xi1_e3.png", times=(0.05, 0.3, 3, 30, 300, None))
profiles("soa_rh0.30_xi0", "figS_profiles_rh030_xi0.png")
profiles("soa_rh0.30_xi1_e0_wet0", "figS_profiles_rh030_wet0.png")
profiles("soa_rh0.30_xi1_e0_ramp60", "figS_profiles_rh030_ramp60.png", times=(1, 10, 30, 60, 100, None))
profiles("soa_rh0.80_xi1_e0", "figS_profiles_rh080.png", times=(0.05, 0.2, 0.5, 1, 5, None))

# space-time map of w_MBTCA for the baseline and the high-eta0 case
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
pc = None
for a_, tag in zip(ax, ["soa_rh0.30_xi1_e0", "soa_rh0.30_xi1_e3"]):
    if tag not in soa:
        continue
    h, z, m, a = soa[tag]
    t_s = z["t"] * m["t0_s"]
    k0 = 1
    T, Xg = np.meshgrid(t_s[k0:], z["X"], indexing="ij")
    rr = Xg * z["R"][k0:, None]
    pc = a_.pcolormesh(T, rr, z["wP"][k0:], cmap="viridis", shading="nearest", vmin=0, vmax=0.75, rasterized=True)
    a_.plot(t_s[k0:], z["R"][k0:], color="white", lw=0.8)
    a_.set_xscale("log"); a_.set_xlabel("t (s)"); a_.set_title(tag, fontsize=8.5)
ax[0].set_ylabel("r / R$_0$")
if pc is not None:
    plt.colorbar(pc, ax=ax, label="w$_{MBTCA}$", shrink=0.9)
plt.savefig(f"{FIG}/fig5_spacetime.png", bbox_inches="tight"); plt.close()

df = pd.DataFrame(rows)
df.to_csv(f"{FIG}/summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print(df.to_string(float_format=lambda v: f"{v:.4g}"))
print("\nLate-time scaling (figures/scaling.csv): tau vs L^2/D_shell, D_shell from the harmonic (series) and geometric mean")
print(scal.to_string(float_format=lambda v: f"{v:.3g}"))

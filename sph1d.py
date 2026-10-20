"""Radially symmetric drying of a water + organic + AS droplet: two-field Cahn-Hilliard on a shrinking sphere (or disc).

JAX/NumPy port of paper_6 code/m3g_drying_sph.py (dolfinx) with the same P1 weak form, the same lagged mobility,
evaporation law, wetting term and time-step control:
   d(R^d w_i)/dt - div_X(R^(d-1) Rdot w_i X) - div_X(R^(d-2) L_i grad_X mu_i) = 0,   X = r/R in [0, 1]
   mu_i = dg/dw_i - (kappa_i/R^2) lap_X w_i,  wetting term (H/R) at X = 1,  volume weight X^(d-1)
   dR/dt = -Lambda (a_w,surf - RH(t))/R,   Lambda = D_g c_v/(rho D_ref)     (time unit R0^2/D_ref)
Additions for the viscous-SOA follow-up:
   * thermodynamics from either the paper_6 PEG fit (m3f_gex_fit_290K.npz, no water ideal term) or a fit with the
     water ideal term w(ln w - 1)/M_w split off (gex_fit_290K.npz from s3_gexfit.py, valid down to w_water ~ 0.04);
   * mobility factor (eta_ref/eta(w))^xi with eta from an AIOMFAC-VISC table (xi = 1: Stokes-Einstein as --visc in
     m3g_drying_sph.py; xi < 1: fractional Stokes-Einstein, the usual form for water in viscous organics);
   * a mesh graded towards the surface (--grade), since the drying boundary layer ~ R/Pe gets thin;
   * RH ramp, planar (dim 1) mode for the numerical interfacial tension;
   * --mob3 1: three species mobilities m_i = d_i M_i w_i (eta_ref/eta)^xi_i (organic and AS with --visc, water with
     --xi_w, --dW) combined in the fast-mode (mass-fixed frame) Onsager matrix for the exchange potentials
         Lambda_ij = m_i delta_ij - (m_i w_j + w_i m_j) + w_i w_j (m_P + m_S + m_W),   i, j in {P, S}
     so water has its own mobility (default --mob3 0: the diagonal L_i = M_i w_i (1 - w_i) of m3g).
Newton with the exact Jacobian (12 coloured JVPs, block-banded LU)."""
import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.linalg import solve_banded  # noqa: E402

jax.config.update("jax_enable_x64", True)
WMIN = 1e-3


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="peg", help="peg (paper_6 data) or a folder with gex_fit_290K.npz etc.")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.environ.get("M3F_DATA", os.path.join(here, "m3f_bundle", "data")),
                    help="paper6 m3f_bundle/data (PEG runs); not in the repository")
    ap.add_argument("--rh0", type=float, default=0.92)
    ap.add_argument("--rh", type=float, default=0.80)
    ap.add_argument("--ramp_s", type=float, default=0.0, help="linear RH ramp duration (s); 0: step")
    ap.add_argument("--D", type=float, default=1e-10)
    ap.add_argument("--Dg", type=float, default=2.5e-5)
    ap.add_argument("--rho", type=float, default=1150.0)
    ap.add_argument("--T", type=float, default=290.15)
    ap.add_argument("--R0_um", type=float, default=10.0)
    ap.add_argument("--wet", type=float, default=2.0)
    ap.add_argument("--rh_ref", type=float, default=0.85)
    ap.add_argument("--sigma_num", type=float, default=None, help="numerical tension at kP = 2e-4 (default: M3e meta)")
    ap.add_argument("--kP", type=float, default=2e-4)
    ap.add_argument("--kS", type=float, default=2e-3)
    ap.add_argument("--dP", type=float, default=1.0)
    ap.add_argument("--dS", type=float, default=1.0)
    ap.add_argument("--visc", type=float, default=0.0, help="exponent xi of (eta_ref/eta)^xi; 0: constant mobility")
    ap.add_argument("--eta_idx", type=int, default=0, help="which eta0 column of the SOA viscosity table")
    ap.add_argument("--eta_ref", type=float, default=None,
                    help="viscosity (Pa s) at which the mobility is D (default: the initial state of this run, which grows with "
                         "eta0; pass the eta_idx 0 value 0.0086 to keep D*eta fixed across eta0 as Stokes-Einstein implies)")
    ap.add_argument("--fac_min", type=float, default=1e-14)
    ap.add_argument("--dim", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--grade", type=float, default=1.0, help="ratio of the centre to the surface element size")
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--rseed", type=int, default=1)
    ap.add_argument("--tend_s", type=float, default=None)
    ap.add_argument("--dt0", type=float, default=1e-6)
    ap.add_argument("--dtmax", type=float, default=None)
    ap.add_argument("--dR_max", type=float, default=2e-3)
    ap.add_argument("--nsnap", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(here, "runs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--wmin", type=float, default=1e-3, help="log regularization threshold (m3g: 1e-3)")
    ap.add_argument("--lfloor", type=float, default=1e-4, help="floor of w in the degenerate mobility (m3g: 1e-4)")
    ap.add_argument("--mob3", type=int, default=0, help="1: three-mobility fast-mode matrix (water mobility separate)")
    ap.add_argument("--xi_w", type=float, default=None, help="water exponent of (eta_ref/eta)^xi_w (default: --visc)")
    ap.add_argument("--dW", type=float, default=1.0, help="water mobility prefactor D_W,ref/D_ref (--mob3 1)")
    ap.add_argument("--implicit", type=int, default=1, help="1: R(t) solved with the fields (backward Euler); 0: explicit Rdot as m3g")
    return ap.parse_args(argv)


class Thermo:
    """g(p, s) per kg in RT: fitted G_ex + ideal-like terms (water term only for the SOA fits)."""

    def __init__(self, a):
        import pandas as pd
        if a.system == "peg":
            TB = np.load(os.path.join(a.data, "m3e_table_290K.npz"))
            self.MP, self.MS, self.MW = float(TB["M_PEG"]), float(TB["M_AS"]), float(TB["MW"])
            GF = np.load(os.path.join(a.data, "m3f_gex_fit_290K.npz"))
            self.water = False
            self.phases = pd.read_csv(os.path.join(a.data, "m3a_peg_as_290K.csv")).set_index("RH")
            self.visc_file = os.path.join(a.data, "m3f_visc_table_290K.npz")
            meta = json.load(open(os.path.join(a.data, "m3e/rh0.85_wet2_kP0.0002_n200_s1_meta.json")))
            self.sigma_num = meta["sigma_num"]
        else:
            GF = np.load(os.path.join(a.system, "gex_fit_290K.npz"))
            self.MP, self.MS, self.MW = float(GF["M_PEG"]), float(GF["M_AS"]), float(GF["MW"])
            self.water = bool(int(GF["water_ideal"]))
            self.phases = pd.read_csv(os.path.join(a.system, "m3a_peg_as_290K.csv")).set_index("RH")
            self.visc_file = os.path.join(a.system, "tables_290K.npz")
            f = os.path.join(a.system, "sigma_num.json")
            self.sigma_num = json.load(open(f))["sigma_num"] if os.path.exists(f) else None
        self.c = jnp.array(GF["coef"], float)
        self.i = jnp.array(GF["i"], float)
        self.e = jnp.array(GF["e"], float)

    def gex(self, p, s):
        s = jnp.maximum(s, 1e-4)
        return jnp.sum(self.c * p[..., None] ** self.i * s[..., None] ** self.e, axis=-1)

    def g_full(self, p, s):
        pc, sc = jnp.maximum(p, 1e-6), jnp.maximum(s, 1e-6)
        g = self.gex(p, s) + pc * (jnp.log(pc) - 1) / self.MP + 3 * sc * (jnp.log(sc) - 1) / self.MS
        if self.water:
            wc = jnp.maximum(1 - p - s, 1e-6)
            g = g + wc * (jnp.log(wc) - 1) / self.MW
        return g

    def lnaw(self, p, s):
        gp, gs = jax.grad(lambda x, y: self.g_full(x, y).sum(), argnums=(0, 1))(p, s)
        return self.MW * (self.g_full(p, s) - p * gp - s * gs)


def lnreg(w, wmin=WMIN):
    z = (w - wmin) / wmin
    return jnp.where(w > wmin, jnp.log(jnp.maximum(w, wmin)), np.log(wmin) + z - 0.5 * z ** 2)


class Solver:
    def __init__(self, a, th):
        self.a, self.th = a, th
        n = a.n
        if a.grade == 1.0:
            x = np.linspace(0.0, 1.0, n + 1)
        else:                                           # geometric grading: h_centre / h_surface = grade
            q = a.grade ** (-1.0 / (n - 1))
            h = q ** np.arange(n); x = np.concatenate([[0.0], np.cumsum(h)]); x /= x[-1]
        if a.dim == 1:
            x = x - 0.5                                 # planar slab [-0.5, 0.5]
        self.x = x
        self.h = jnp.array(np.diff(x))
        gq = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10]); wq = np.array([5, 8, 5]) / 18
        self.N = jnp.array(np.stack([1 - gq, gq], 1))                 # (3, 2)
        self.wq = jnp.array(wq)
        Xq = x[:-1, None] + np.diff(x)[:, None] * gq[None, :]           # (n, 3)
        self.Xq = jnp.array(Xq)
        self.w2 = jnp.array(Xq ** (a.dim - 1)) if a.dim > 1 else jnp.ones_like(self.Xq)
        rr = th.phases.loc[a.rh_ref]
        self.org, self.aq = (rr.org_w_PEG, rr.org_w_AS), (rr.aq_w_PEG, rr.aq_w_AS)
        dP, dS = self.org[0] - self.aq[0], self.org[1] - self.aq[1]
        self.dPS, self.den = (dP, dS), a.kP * dP ** 2 + a.kS * dS ** 2
        self.res = jax.jit(self._residual)
        self.jvps = jax.jit(jax.vmap(lambda U, v, *c: jax.jvp(lambda u: self._residual(u, *c), (U,), (v,))[1],
                                     in_axes=(None, 0) + (None,) * 9))
        nd = 4 * (n + 1)
        seeds = np.zeros((12, nd))
        for col in range(3):
            for b in range(4):
                seeds[4 * col + b, 4 * np.arange(col, n + 1, 3) + b] = 1
        self.seeds = jnp.array(seeds)
        r = np.arange(nd); i = r // 4
        self.band_rows, self.band_cols, self.band_seed = [], [], []
        for di in (-1, 0, 1):
            j = i + di; ok = (j >= 0) & (j <= n)
            for b in range(4):
                rows = r[ok]; cols = 4 * j[ok] + b
                self.band_rows.append(rows); self.band_cols.append(cols); self.band_seed.append(4 * (j[ok] % 3) + b)
        self.band_rows = np.concatenate(self.band_rows); self.band_cols = np.concatenate(self.band_cols)
        self.band_seed = np.concatenate(self.band_seed)

    def _residual(self, U, Un, LP, LS, LX, dt, Rn, R1, Rd, H):
        a, th = self.a, self.th
        d = a.dim
        F4 = U.reshape(-1, 4); Fn = Un.reshape(-1, 4)
        h = self.h[:, None]
        def q(v):           # values at quadrature points (n, 3) and derivatives (n, 1)
            return v[:-1, None] * self.N[None, :, 0] + v[1:, None] * self.N[None, :, 1], ((v[1:] - v[:-1]) / self.h)[:, None]
        wp, dwp = q(F4[:, 0]); ws, dws = q(F4[:, 1]); mp, dmp = q(F4[:, 2]); ms, dms = q(F4[:, 3])
        wpn, _ = q(Fn[:, 0]); wsn, _ = q(Fn[:, 1]); lp, _ = q(LP); ls, _ = q(LS); lx, _ = q(LX)
        X, W = self.Xq, self.w2 * self.wq[None, :] * h
        gp, gs = jax.grad(lambda x, y: th.gex(x, y).sum(), argnums=(0, 1))(wp, ws)
        mup = lnreg(wp, a.wmin) / th.MP + gp
        mus = 3 * lnreg(ws, a.wmin) / th.MS + gs
        if th.water:
            lw = lnreg(1 - wp - ws, a.wmin) / th.MW
            mup, mus = mup - lw, mus - lw
        Rd1 = R1 ** (d - 1) if d > 1 else 1.0
        Rd2 = R1 ** (d - 2) if d > 1 else 1.0
        N0, N1 = self.N[None, :, 0], self.N[None, :, 1]
        dN0, dN1 = -1 / h, 1 / h
        def assemble(f_val, f_der):      # sum over quad points, element contributions to the two nodes
            e0 = jnp.sum(W * (f_val * N0 + f_der * dN0), 1)
            e1 = jnp.sum(W * (f_val * N1 + f_der * dN1), 1)
            return jnp.zeros(F4.shape[0]).at[:-1].add(e0).at[1:].add(e1)
        if a.mob3:           # full 2x2 mobility matrix (cross term lx)
            jp, js = Rd2 * (lp * dmp + lx * dms), Rd2 * (lx * dmp + ls * dms)
        else:                # diagonal, same operation order as before: --mob3 0 reproduces earlier runs bit for bit
            jp, js = Rd2 * lp * dmp, Rd2 * ls * dms
        Rqp = assemble((R1 ** d * wp - Rn ** d * wpn) / dt, Rd1 * Rd * wp * X + jp)
        Rqs = assemble((R1 ** d * ws - Rn ** d * wsn) / dt, Rd1 * Rd * ws * X + js)
        Rrp = assemble(mp - mup, -a.kP / R1 ** 2 * dwp)
        Rrs = assemble(ms - mus, -a.kS / R1 ** 2 * dws)
        if d > 1:            # wetting at X = 1 only
            pe, se = F4[-1, 0], F4[-1, 1]
            psi = jnp.clip((a.kP * self.dPS[0] * (pe - self.aq[0]) + a.kS * self.dPS[1] * (se - self.aq[1])) / self.den, 0, 1)
            fac = H / R1 * 6 * psi * (1 - psi) / self.den
            Rrp = Rrp.at[-1].add(fac * a.kP * self.dPS[0]); Rrs = Rrs.at[-1].add(fac * a.kS * self.dPS[1])
        return jnp.stack([Rqp, Rqs, Rrp, Rrs], 1).ravel()

    def newton(self, U0, consts, rtol=1e-9, atol=1e-11, maxit=15):
        U = U0.copy()
        F = np.asarray(self.res(U, *consts)); f0 = np.linalg.norm(F)
        nd = len(U)
        for it in range(maxit + 1):
            fn = np.linalg.norm(F)
            if not np.isfinite(fn):
                return U, -1, it
            if fn <= max(atol, rtol * f0) and it > 0 or fn <= atol:
                return U, 1, it
            if it == maxit:
                break
            Jv = np.asarray(self.jvps(jnp.asarray(U), self.seeds, *consts))       # (12, nd)
            ab = np.zeros((15, nd))
            ab[7 + self.band_rows - self.band_cols, self.band_cols] = Jv[self.band_seed, self.band_rows]
            try:
                dU = solve_banded((7, 7), ab, F, check_finite=False)
            except Exception:
                return U, -2, it
            U = U - dU
            F = np.asarray(self.res(jnp.asarray(U), *consts))
            if np.linalg.norm(dU) <= 1e-10 * np.linalg.norm(U) and np.isfinite(F).all():     # PETSc-like stol
                return U, 2, it + 1
            fnew = np.linalg.norm(F)
            if it >= 1 and fnew < 1e-8 and fnew > 0.3 * fn:   # stagnation at the round-off floor (|mu| ~ 50 for SOA)
                return U, 3, it + 1
        return U, -3, maxit

    def setup_implicit(self, LAM):
        """R(t+dt) as an extra unknown: F_R = R1 - Rn + dt Lambda (a_w(U_edge) - RH)/R1 (backward Euler), so the
        surface water activity and the radius are solved together (the explicit Rdot of m3g overshoots once a thin,
        low-mobility skin forms)."""
        th = self.th
        self.LAM = LAM
        def fu(U, R1, Un, LP, LS, LX, dt, Rn, H):
            return self._residual(U, Un, LP, LS, LX, dt, Rn, R1, (R1 - Rn) / dt, H)
        def fr(pe, se, R1, dt, Rn, rh):
            return R1 - Rn + dt * LAM * (jnp.exp(th.lnaw(pe, se)) - rh) / R1
        self.fu = jax.jit(fu)
        self.dfu_dR = jax.jit(lambda U, R1, *c: jax.jvp(lambda r: fu(U, r, *c), (R1,), (1.0,))[1])
        self.jvpsR = jax.jit(jax.vmap(lambda U, v, R1, *c: jax.jvp(lambda u: fu(u, R1, *c), (U,), (v,))[1],
                                      in_axes=(None, 0, None) + (None,) * 7))
        self.fr = jax.jit(fr)
        self.dfr = jax.jit(jax.grad(fr, argnums=(0, 1, 2)))

    def newton_R(self, U0, R0, c, dt, Rn, rh, rtol=1e-9, atol=1e-11, maxit=15):
        """c = (Un, LP, LS, LX, dt, Rn, H)."""
        U, R1 = U0.copy(), float(R0)
        def resid(U, R1):
            return np.asarray(self.fu(jnp.asarray(U), R1, *c)), float(self.fr(U[-4], U[-3], R1, dt, Rn, rh))
        F, FR = resid(U, R1)
        f0 = np.hypot(np.linalg.norm(F), FR)
        nd = len(U)
        for it in range(maxit + 1):
            fn = np.hypot(np.linalg.norm(F), FR)
            if not np.isfinite(fn):
                return U, R1, -1, it
            if (fn <= max(atol, rtol * f0) and it > 0) or fn <= atol:
                return U, R1, 1, it
            if it == maxit:
                break
            Jv = np.asarray(self.jvpsR(jnp.asarray(U), self.seeds, R1, *c))
            ab = np.zeros((15, nd))
            ab[7 + self.band_rows - self.band_cols, self.band_cols] = Jv[self.band_seed, self.band_rows]
            b = np.asarray(self.dfu_dR(jnp.asarray(U), R1, *c))
            gp, gs, gR = (float(v) for v in self.dfr(U[-4], U[-3], R1, dt, Rn, rh))
            try:
                Y = solve_banded((7, 7), ab, np.c_[F, b], check_finite=False)
            except Exception:
                return U, R1, -2, it
            y, z = Y[:, 0], Y[:, 1]
            cy = gp * y[-4] + gs * y[-3]; cz = gp * z[-4] + gs * z[-3]
            dR = (FR - cy) / (gR - cz)
            dU = y - z * dR
            U = U - dU; R1 = R1 - dR
            F, FR = resid(U, R1)
            fnew = np.hypot(np.linalg.norm(F), FR)
            if np.linalg.norm(dU) <= 1e-10 * np.linalg.norm(U) and abs(dR) < 1e-12 and np.isfinite(fnew):
                return U, R1, 2, it + 1
            if it >= 1 and fnew < 1e-8 and fnew > 0.3 * fn:
                return U, R1, 3, it + 1
        return U, R1, -3, maxit


def run(argv=None):
    a = parse(argv)
    th = Thermo(a)
    p_sat = 610.94 * np.exp(17.625 * (a.T - 273.15) / (a.T - 30.11))
    c_v = p_sat * 0.01801528 / (8.314462618 * a.T)
    LAM = a.Dg * c_v / (a.rho * a.D)
    t0_s = (a.R0_um * 1e-6) ** 2 / a.D
    tend = (a.tend_s / t0_s) if a.tend_s else 10.0 / LAM
    dtmax = a.dtmax or 0.004 / LAM
    sig = a.sigma_num if a.sigma_num is not None else th.sigma_num
    H = a.wet * sig * (a.kP / 2e-4) ** 0.5 if a.dim > 1 else 0.0
    tag = a.tag or (f"d{a.dim}_rh{a.rh0:g}to{a.rh:g}_D{a.D:g}_wet{a.wet:g}_n{a.n}" + (f"_xi{a.visc:g}" if a.visc else ""))
    os.makedirs(a.out, exist_ok=True)
    S = Solver(a, th)
    S.setup_implicit(LAM)
    n = a.n
    r0 = th.phases.loc[a.rh0]
    if a.dim == 1:          # planar slab: org in |x| < 0.25 at the rh_ref tie line
        prof = 0.5 * (1 + np.tanh(-np.abs(S.x) / 0.02 + 0.25 / 0.02))
        wp0 = S.aq[0] + (S.org[0] - S.aq[0]) * prof; ws0 = S.aq[1] + (S.org[1] - S.aq[1]) * prof
        w0 = (float(wp0.mean()), float(ws0.mean()))
    else:
        assert r0.n_liquids == 1, "--rh0 must be single phase"
        w0 = (r0.single_w_PEG, r0.single_w_AS)
        rng = np.random.default_rng(a.rseed)
        wp0 = w0[0] + a.noise * rng.standard_normal(n + 1); ws0 = w0[1] - a.noise * rng.standard_normal(n + 1)
    U = np.zeros(4 * (n + 1))
    U[0::4], U[1::4] = wp0, ws0
    U[2::4] = np.log(np.maximum(wp0, WMIN)) / th.MP
    U[3::4] = 3 * np.log(np.maximum(ws0, WMIN)) / th.MS
    lnaw_nodes = jax.jit(th.lnaw)
    eta_of = None
    if a.visc:
        VT = np.load(th.visc_file)
        if "visc_wp" in VT:
            L10 = np.array(VT["log10eta"][a.eta_idx], float); gp_, gs_ = VT["visc_wp"], VT["visc_ws"]
        else:
            L10 = np.array(VT["log10eta"], float); gp_, gs_ = VT["wp"], VT["ws"]
        L10 = np.where(np.isfinite(L10), L10, np.nanmax(L10))
        I = RegularGridInterpolator((gp_, gs_), L10, bounds_error=False, fill_value=None)
        eta_of = lambda p, s: 10 ** I(np.c_[np.clip(p, gp_[0], gp_[-1]), np.clip(s, gs_[0], gs_[-1])])  # noqa: E731
        eta_ref = a.eta_ref if a.eta_ref else float(eta_of(np.array([w0[0]]), np.array([w0[1]]))[0])
    Xn = S.x
    wvol = Xn ** (a.dim - 1) if a.dim > 1 else np.ones_like(Xn)

    def integ(f):            # trapezoid with the volume weight (diagnostics only)
        return np.trapezoid(f * wvol, Xn)

    def rh_amb(t):
        if a.ramp_s <= 0:
            return a.rh
        return a.rh0 + (a.rh - a.rh0) * min(t * t0_s / a.ramp_s, 1.0)

    xi_w = a.visc if a.xi_w is None else a.xi_w

    def mobility(Uv):
        """nodal (lagged) mobilities: diagonal LP, LS and the cross term LX (0 unless --mob3); fac, facW for output"""
        p, s = Uv[0::4], Uv[1::4]
        fac = np.ones_like(p); facW = fac
        if eta_of is not None:
            eta = eta_of(p, s)
            fac = np.maximum((eta_ref / eta) ** a.visc, a.fac_min)
            facW = np.maximum((eta_ref / eta) ** xi_w, a.fac_min)
        if not a.mob3:
            LP = a.dP * fac * th.MP * np.clip(p, a.lfloor, 1) * np.clip(1 - p, 0, 1)
            LS = a.dS * fac * th.MS * np.clip(s, a.lfloor, 1) * np.clip(1 - s, 0, 1)
            return LP, LS, np.zeros_like(p), fac, facW
        pc, sc = np.clip(p, 0, 1), np.clip(s, 0, 1)
        mP = a.dP * fac * th.MP * np.clip(p, a.lfloor, 1)
        mS = a.dS * fac * th.MS * np.clip(s, a.lfloor, 1)
        mW = a.dW * facW * th.MW * np.clip(1 - p - s, a.lfloor, 1)
        mt = mP + mS + mW
        LP = mP - 2 * mP * pc + pc ** 2 * mt
        LS = mS - 2 * mS * sc + sc ** 2 * mt
        LX = -(mP * sc + pc * mS) + pc * sc * mt
        return LP, LS, LX, fac, facW

    thr = 0.5 * (S.org[0] + S.aq[0])
    hist, snaps = [], []
    mass0 = None
    t_wall = time.time()

    def surface_aw(Uv):
        return float(np.exp(lnaw_nodes(jnp.array(Uv[-4]), jnp.array(Uv[-3]))))

    def record(t, k, R, Rdot, Uv):
        nonlocal mass0
        p, s = Uv[0::4], Uv[1::4]
        aw = np.exp(np.asarray(lnaw_nodes(jnp.array(p), jnp.array(s))))
        mP, mS = R ** a.dim * integ(p), R ** a.dim * integ(s)
        if mass0 is None:
            mass0 = (mP, mS)
        vol = integ(np.ones_like(p))
        LPv, LSv, LXv, fac, facW = mobility(Uv)
        y = p > thr
        rec = {"t": t, "t_s": t * t0_s, "steps": k, "R": R, "Rdot": Rdot, "Pe_evap": -Rdot * R, "RH": rh_amb(t),
               "aw_surf": float(aw[-1]), "aw_mean": integ(aw) / vol, "aw_centre": float(aw[0]),
               "PEG_rel": mP / mass0[0], "AS_rel": mS / mass0[1], "wP_edge": float(p[-1]), "wP_centre": float(p[0]),
               "wS_centre": float(s[0]), "wW_centre": float(1 - p[0] - s[0]), "wW_mean": 1 - integ(p + s) / vol,
               "wP_min": float(p.min()), "wP_max": float(p.max()), "peg_layers": int(np.sum(np.diff(y.astype(int)) == 1) + y[0]),
               "logfac_min": float(np.log10(fac.min())), "logfac_surf": float(np.log10(fac[-1])),
               "logfacW_min": float(np.log10(facW.min()))}
        hist.append(rec)
        snaps.append((t, R, p.copy(), s.copy(), aw.copy(), fac.copy()))
        if not a.quiet:
            print(f"t {t:.4g} ({rec['t_s']:.3g} s, {k}): R {R:.4f}, aw surf {rec['aw_surf']:.4f} mean {rec['aw_mean']:.4f} "
                  f"centre {rec['aw_centre']:.4f}, wP edge {rec['wP_edge']:.3f} [{rec['wP_min']:.3f}, {rec['wP_max']:.3f}], "
                  f"layers {rec['peg_layers']}, log fac min {rec['logfac_min']:.1f}, mass {rec['PEG_rel']:.8f}/{rec['AS_rel']:.8f} "
                  f"({time.time() - t_wall:.0f}s)", flush=True)

    R, t, k, dt = 1.0, 0.0, 0, a.dt0
    Rdot = 0.0 if a.dim == 1 else -LAM * (surface_aw(U) - rh_amb(0)) / R
    if not a.quiet:
        print(f"[{tag}] w0 {w0[0]:.4f}/{w0[1]:.4f}, Lambda {LAM:.3g}, Pe_evap0 {LAM * (a.rh0 - a.rh):.3g}, t0 {t0_s:.3g} s, "
              f"H {H:.3g}, tend {tend:.3g}" + (f", eta_ref {eta_ref:.3g} Pa s" if eta_of else ""), flush=True)
    record(0.0, 0, R, Rdot, U)
    times = list(np.geomspace(tend * 1e-4, tend, a.nsnap))
    nfail = 0
    status = "ok"
    while t < tend - 1e-12:
        Rdot = 0.0 if a.dim == 1 else -LAM * (surface_aw(U) - rh_amb(t)) / R
        dt = min(dt, tend - t, a.dR_max * R / max(abs(Rdot), 1e-12))
        LP, LS, LX, _, _ = mobility(U)
        if a.implicit and a.dim > 1:
            Unew, R1, reason, its = S.newton_R(U, R + dt * Rdot, (jnp.asarray(U), jnp.asarray(LP), jnp.asarray(LS), jnp.asarray(LX), dt, R, H),
                                               dt, R, rh_amb(t + dt))
        else:
            consts = (jnp.asarray(U), jnp.asarray(LP), jnp.asarray(LS), jnp.asarray(LX), dt, R, R + dt * Rdot, Rdot, H)
            Unew, reason, its = S.newton(U, consts); R1 = R + dt * Rdot
        if not (reason > 0 and np.isfinite(Unew).all() and Unew[0::4].min() > -0.02 and Unew[1::4].min() > -0.02):
            dt *= 0.3; nfail += 1
            if dt < 1e-14:
                status = "collapsed"; record(t, k, R, Rdot, U); print("time step collapsed at t", t, flush=True)
                break
            continue
        U = Unew
        Rdot = (R1 - R) / dt
        R = R1; t += dt; k += 1
        dt = min(dt * (1.25 if its <= 4 else 0.8 if its >= 8 else 1.0), dtmax)
        while times and t >= times[0] - 1e-12:
            record(t, k, R, Rdot, U); times.pop(0)
    import pandas as pd
    pd.DataFrame(hist).to_csv(f"{a.out}/{tag}_hist.csv", index=False)
    np.savez_compressed(f"{a.out}/{tag}.npz", X=Xn, t=np.array([s[0] for s in snaps]), R=np.array([s[1] for s in snaps]),
                        wP=np.array([s[2] for s in snaps]), wS=np.array([s[3] for s in snaps]),
                        aw=np.array([s[4] for s in snaps]), fac=np.array([s[5] for s in snaps]), args=json.dumps(vars(a)))
    meta = {"tag": tag, "Lambda": LAM, "t0_s": t0_s, "H": H, "sigma_num": sig, "w0": w0, "org_ref": S.org, "aq_ref": S.aq,
            "steps": k, "rejected": nfail, "status": status, "wall_s": time.time() - t_wall}
    if a.dim == 1:
        p, s = U[0::4], U[1::4]
        meta["sigma_num_planar"] = 0.5 * (a.kP * np.trapezoid(np.gradient(p, Xn) ** 2, Xn)
                                          + a.kS * np.trapezoid(np.gradient(s, Xn) ** 2, Xn))
        meta["plateau_org"] = [float(p[np.abs(Xn) < 0.05].mean()), float(s[np.abs(Xn) < 0.05].mean())]
        meta["plateau_aq"] = [float(p[np.abs(Xn) > 0.45].mean()), float(s[np.abs(Xn) > 0.45].mean())]
    json.dump(meta, open(f"{a.out}/{tag}_meta.json", "w"), indent=1)
    if not a.quiet:
        print("done", tag, json.dumps({k_: v for k_, v in meta.items() if k_ in ("steps", "rejected", "wall_s", "sigma_num_planar")}))
    return hist, meta


if __name__ == "__main__":
    run()

# aiomfac-dynamics — paper6 follow-up 1: viscous SOA + AS drying (1D spherical Cahn–Hilliard)

Results are summarised in the doc "후속 연구 1 예비 결과". This folder holds everything needed to reproduce it.

## Files
- `sph1d.py` — JAX/NumPy port of `code/m3g_drying_sph.py` (same P1 weak form, mobility, evaporation, wetting).
  Additions: thermodynamics with the water ideal term split off (`--system <folder>`), mobility factor
  `(eta_ref/eta)^xi` (`--visc xi`, `--eta_idx`), mesh graded towards the surface (`--grade`), implicit radius
  (`--implicit 1`, default), RH ramp (`--ramp_s`), three-mobility fast-mode matrix with a separate water mobility (`--mob3 1`, `--xi_w`, `--dW`),
  fixed viscosity normalisation (`--eta_ref`; default: each run's
  own initial viscosity, which grows with eta0), planar mode for sigma_num (`--dim 1`), `--wmin`/`--lfloor`
  (log regularisation and mobility floor; paper6 values 1e-3/1e-4, SOA runs 1e-6/1e-7).
  PEG runs read the paper6 bundle data (`--data .../m3f_bundle/data`).
- `soa_common.py`, `s1_phases.py`, `s2_tables.py`, `s3_gexfit.py` — MBTCA + AS tables (aiomfac_py, 290.15 K):
  phase compositions (`m3a_peg_as_290K.csv`, same column names as paper6; "PEG" = organic), activity/viscosity grid
  (`tables_290K.npz`, eta0 = 6e10, 1e12, 1e14, 1e16 Pa s), G_ex fit (`gex_fit_290K.npz`, degree 12).
  `SOA=DTAA` or `SOA=terpenylic` switches the proxy.
- `srh.py` — exact separation RH (binodal a_w) of the 1:1 particles: PEG-400 0.9016, MBTCA 0.8169.
- Run lists (each skips runs whose `_meta.json` exists): `run_soa.sh` all MBTCA runs (`q1.sh` + `q2.sh` split them over
  two cores), `run_peg3d.sh` PEG 3D/2D and H = 0 runs, `run_validation.sh` planar sigma_num and 2D disc checks against
  dolfinx (these use `--implicit 0`, the explicit Rdot of m3g, as when they were run), `run_kappa.sh` interface-width
  sensitivity (kappa x0.5, x2, seed 2; fine mesh), `run_etaref.sh` eta0 runs with D*eta fixed (`--eta_ref`): the adopted convention, used in fig2/4/5,
  `run_size.sh` R0 = 1, 3, 30 um at fixed physical interface width (+ a pure-rescaling check), `run_mob3.sh`
  three-mobility runs (water anchored at pure water: `*_m3aw*`; dW = 1: `*_m3w*`).
- `analyze.py` — figures, `figures/summary.csv` (+ `summary.txt`) and `figures/scaling.csv`.
  t90/t99 are log-interpolated between snapshots (`*_snap_s`: first snapshot at or below the level, as in the first
  draft, up to ~17 % high). PEG rows: onset time, previous snapshot time (= resolution), mean a_w and X at onset,
  final surface w_PEG. `scaling.csv`: tau (e-folding of f_w over 0.1-0.005) vs L^2/D_shell with D_shell from the
  harmonic (series, physical) and geometric means of the mobility factor over the outer organic-rich shell, and
  R^2/(pi^2 D_ref <factor>_harm) over the whole particle. `fig6_sensitivity.png`: kappa and eta_ref sensitivity.
- `results/MBTCA/` — tables, fit, sigma_num. `runs/` — every run: `<tag>_hist.csv` (time series),
  `<tag>.npz` (profiles), `<tag>_meta.json`, `<tag>.log`. (The first SOA runs before the positivity fix are not kept.)

## Environment
Python 3.13, jax 0.x (CPU, float64), numpy, scipy, pandas, matplotlib; aiomfac_py (SeoulTechPSE/aiomfac-python,
editable install) with epam.indigo for SMILES. No dolfinx needed.

## Running in paper6's dolfinx code
`m3g_drying_sph.py --data results/MBTCA` reads the CSV and tables, but its `gex_ufl`/`gid` have no water ideal
term and its m3f_gex file name differs; add `+ w*(ln w - 1)/M_W` to `gid`, `- lnreg(1-wp-ws)/M_W` to both
mu equations, and load `gex_fit_290K.npz`. The implicit radius is not in m3g_drying_sph.py; with low-mobility skins
the explicit Rdot overshoots (surface water < 0).

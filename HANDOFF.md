# Handoff: paper6 follow-up 1 (viscous SOA + AS skin and water trapping)

Paste the "Prompt for the new session" at the end together with this zip (soa_followup1.zip).

## Goal
paper6 (AIOMFAC-based phase-field model of LLPS in drying PEG-400 + AS droplets) proposed four follow-ups; #2 (sub-micron
size-dependent LLPS) is running elsewhere. This work started #1: viscous SOA + AS, mobility D(w) from AIOMFAC-VISC,
Pe_evap >= 30 regime, skin formation, water trapping, layers. It also answered paper6 review items A-1 (3D vs 2D) and
A-2 (H = 0 control, exact binodal a_w).

## State (2026-10-20)
- Done: solver port + validation, MBTCA + AS tables, 14 SOA runs, 8 PEG runs, figures, write-up doc
  ("후속 연구 1 예비 결과", claude.ai artifact on the old account; export it to Markdown/Word and attach).
- All numbers below are in `figures/summary.csv` and `runs/<tag>_hist.csv`.

## Environment (no dolfinx needed)
```
git clone https://github.com/SeoulTechPSE/aiomfac-python.git && pip install -e aiomfac-python
pip install epam.indigo "jax[cpu]" numpy scipy pandas matplotlib
```
PEG runs need the paper6 bundle data (not in this repository): unzip m3f_bundle.zip so that `m3f_bundle/data`
sits next to `sph1d.py` (git-ignored), or set `M3F_DATA=<path>/data`, or pass `--data`. SOA runs use `--system results/MBTCA`.
Typical SOA run: `python3 sph1d.py --system results/MBTCA --rh_ref 0.75 --rh0 0.85 --n 600 --grade 10 --dtmax 2
--nsnap 50 --wmin 1e-6 --lfloor 1e-7 --rh 0.30 --visc 1 --tend_s 200 --tag soa_rh0.30_xi1_e0` (~7-15 min on 1 core).

## Key decisions and why
1. Solver: JAX port of paper6 `code/m3g_drying_sph.py` (conda-forge blocked, no FEniCSx). Same P1 weak form;
   Jacobian by 12 coloured JVPs + banded LU. Validated: sigma_num planar 0.00568 vs 0.00576 (M3e); 2D disc R(t),
   a_w,surf vs the paper6 m3g states to 4 digits.
2. Implicit radius (`--implicit 1`, default): explicit Rdot overshot once a thin low-mobility skin formed
   (surface w_water < 0). R(t+dt) is a Newton unknown (bordered banded solve).
3. Positivity: organic in the AS-rich phase is ~1e-7 at RH 0.3; WMIN 1e-3 / mobility floor 1e-4 let w_P go to
   -0.02 and runs collapsed late. SOA runs use --wmin 1e-6 --lfloor 1e-7; PEG runs keep the paper6 values.
4. SOA proxy MBTCA (O:C 0.75): SRH 0.8169 (PEG-400 0.9016). C19H28O7 dimer, pinic, pinonic acid are immiscible
   with AS at all RH in AIOMFAC (rejected). DTAA (SRH ~0.985) and terpenylic acid (~0.96) are alternatives
   (`SOA=DTAA python3 s1_phases.py` etc.).
5. G_ex fit: degree 12 with the water ideal term w(ln w - 1)/M_w split off (valid w_water >= 0.04). Tie lines within
   0.003 (RH 0.75-0.40), 0.01-0.02 near the critical point; ln a_w within +-0.005 to RH 0.2. Systematic ln a_w bias
   -0.0015 -> final radius 0.8029 vs AIOMFAC 0.7995 at RH 0.3.
6. Viscosity: AIOMFAC-VISC, x_org-weighted aquorg/aquelec (as paper6). The organic phase holds AS (w 0.22-0.29), so
   it is only 10^3.5-10^5.6 Pa s at RH 0.2: semi-solid skin, not glass. Dry eta0 = 6e10 (D2018 Tg 286 K), 1e12,
   1e14, 1e16 as an oligomer sensitivity. Mobility L_i ~ D_ref (eta_ref/eta)^xi, eta_ref 0.0086 Pa s, D_ref 1e-10.
7. Metric: remaining evaporable water f_w = (R^3 - Rinf^3)/(1 - Rinf^3), Rinf = each run's own final radius.
   Layered xi = 1 runs hold ~0.7 % more water than the constant-D run in their diffuse interfaces (artefact).

## Results
| case (RH 0.85 -> ) | t90 (s) | t99 (s) | layers |
| --- | --- | --- | --- |
| 0.30 constant D | 0.16 | 0.28 | 2 |
| 0.30 xi 0.5 | 0.19 | 0.72 | 2 |
| 0.30 xi 1, eta0 6e10 / 1e12 / 1e14 / 1e16 | 0.72 / 1.0 / 2.1 / 4.1 | 12.4 / 24.5 / 61.5 / 175 | 7 |
| 0.30 xi 1, H = 0 | 0.74 | 14.5 | 6 |
| 0.30 xi 1, 60 s ramp | 48 | 70 | 2 |
| 0.50 / 0.80, xi 1 | 0.30 / 0.88 | 1.3 / 3.1 | 3 / 2 |
- Late decay time tau = L^2/D_shell within 0.8-1.1 (L = outer MBTCA-rich shell, 0.55-0.7 um).
- Surface a_w within 0.01 of RH by 0.5 s in all step runs; skin w_MBTCA 0.71-0.74 within 0.1 s for H = 2 and H = 0.
- Mesh check (n 1200, grade 20): R within 5e-4, mean a_w within 1e-3. Mass error < 1.4e-6 everywhere.
- PEG A-1: 3D final R 0.824 (AIOMFAC 0.820) vs 2D 0.749 (0.743); aqueous onset 0.17 s (3D) vs 0.22 s (2D);
  95 % radius change 0.85 vs 1.13 s; onset mean a_w 0.893-0.901 vs binodal 0.9016.
- PEG A-2 (H = 2 / H = 0): onset time 0.155/0.155, 0.173/0.173, 0.138/0.123 s at Pe 0.038/0.38/3.8; with H = 0 the
  AS-rich phase forms at the surface (X = 1) and covers it (final surface w_PEG 0.002). Location/time of onset =
  evaporative enrichment; PEG-rich shell = wetting.

## Next steps (open)
- [ ] kappa_P x0.5, x2 sensitivity of L and t99 (interface width)
- [ ] three-mobility (Maxwell-Stefan-like) model with a separate water mobility (largest uncertainty: xi)
- [ ] DTAA, terpenylic acid proxies (O:C dependence)
- [ ] size dependence R0 = 1, 3, 30 um
- [ ] ramp time scan 6, 20, 60, 200 s (layer-count transition; input for follow-up #4)
- [ ] multicomponent alpha-pinene SOA via the NN surrogate (Yoo 2026b)
- [ ] 3D axisymmetric with flow (follow-up #3) to test whether the 7 layers survive capillary flow

## Prompt for the new session
"Clone SeoulTechPSE/aiomfac-dynamics (read HANDOFF.md and README.md first). Attached are m3f_bundle.zip
(paper6 data; unzip into the repository root) and the exported doc. Unzip, install the environment in HANDOFF.md, check that
`python3 analyze.py` reproduces figures/summary.csv, then continue with the next step: <the item you choose>."

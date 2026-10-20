# Handoff: paper6 follow-up 1 (viscous SOA + AS skin and water trapping)

Paste the "Prompt for the new session" at the end together with this zip (soa_followup1.zip).

## Goal
paper6 (AIOMFAC-based phase-field model of LLPS in drying PEG-400 + AS droplets) proposed four follow-ups; #2 (sub-micron
size-dependent LLPS) is running elsewhere. This work started #1: viscous SOA + AS, mobility D(w) from AIOMFAC-VISC,
Pe_evap >= 30 regime, skin formation, water trapping, layers. It also answered paper6 review items A-1 (3D vs 2D) and
A-2 (H = 0 control, exact binodal a_w).

## State (2026-10-20)
- Done: solver port + validation, MBTCA + AS tables, 11 SOA runs, 7 PEG runs (+4 validation runs), figures, write-up doc
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
| 0.30 constant D | 0.14 | 0.26 | 2 |
| 0.30 xi 0.5 | 0.16 | 0.60 | 2 |
| 0.30 xi 1, eta0 6e10 / 1e12 / 1e14 / 1e16 | 0.68 / 0.97 / 1.8 / 3.6 | 11.6 / 21.1 / 57.5 / 168 | 7 |
| 0.30 xi 1, H = 0 | 0.71 | 12.4 | 6 |
| 0.30 xi 1, 60 s ramp | 43 | 69 | 2 |
| 0.50 / 0.80, xi 1 | 0.28 / 0.77 | 1.2 / 3.0 | 3 / 2 |
(t90/t99 log-interpolated between snapshots; the first draft of the doc used the first snapshot past the level,
`*_snap_s` in summary.csv: 12.4 / 24.5 / 61.5 / 175, H = 0 14.5. The 1 % level is close to the model floor: xi = 1 runs
end 0.6-0.8 % above the constant-D radius, the ramp 0.2 % below; `t_mean_s` (mean a_w within 0.01 of RH: 31 / 51 / 154 /
445 s) gives the same eta0 ratio.)
- Late decay time tau = 0.36-0.42 L^2/D_shell with D_shell from the harmonic (series) mean of the mobility factor over the
  outer MBTCA-rich shell (L 0.55-0.7 um; fine mesh 0.48); the doc's 0.8-1.1 came from a geometric mean. `figures/scaling.csv`.
- Surface a_w within 0.02 of RH by 0.36 s in all step runs (within 0.01 only by 0.62 s); surface w_MBTCA reaches 0.71 at
  0.10-0.11 s (eta0 1e16: first snapshot 0.2 s), peak 0.74, for H = 2 and H = 0.
- Mesh check (n 1200, grade 20): R within 5e-4, mean a_w within 1e-3. Mass error < 1.4e-6 everywhere.
- PEG A-1: 3D final R 0.824 (AIOMFAC 0.820) vs 2D 0.749 (0.743); aqueous onset 0.17 s (3D) vs 0.22 s (2D);
  95 % radius change 0.85 vs 1.13 s; onset mean a_w 0.893-0.903 vs binodal 0.9016
  (onset times resolved to one snapshot, ~12 %).
- PEG A-2 (H = 2 / H = 0): onset time 0.155/0.155, 0.173/0.173, 0.138/0.123 s at Pe 0.038/0.38/3.8; with H = 0 the
  AS-rich phase forms at the surface (X = 1) and covers it (final surface w_PEG 0.002). Location/time of onset =
  evaporative enrichment; PEG-rich shell = wetting.

## Session 2 (2026-10-20): review fixes and sensitivity (branch review-fixes)
- Mobility normalisation: D_ref = 1e-10 was set at each run's own initial viscosity, which grows with eta0 (0.0086,
  0.0100, 0.0128, 0.0163 Pa s), so the high-eta0 runs had D*eta up to 1.9x too large for Stokes-Einstein. With
  `--eta_ref 0.0086` (runs `*_er`): t99 11.6 / 24.1 / 82.6 / 315 s (was 11.6 / 21.1 / 57.5 / 168). The eta0 range
  then spans 27x in t99, more than xi 0.5 vs 1 (19x): the doc's "xi matters far more than eta0" no longer holds.
- Interface width (kappa x0.5 / 1 / 2, n 1200): t99 11.5 / 11.5 / 10.5 s (eta0 6e10), 165 / 169 / 168 s (1e16);
  tau unchanged. Layer count 10 / 6 / 2 and 9 / 7 / 5, L 0.45 / 0.55 / 0.68 um (~kappa^0.3). Seed 2: same 6 layers.
  => drying times are robust; the number of layers (and L) is set by the artificially wide interface, not a prediction.
  The step-vs-ramp layer-count contrast (7 vs 2) must not be offered to experiments as is.
- Scaling: tau does not follow L (tau/(L^2/D_shell) 0.28-0.58 across kappa). The whole-particle time
  R^2/(pi^2 D_ref <factor>_harm) gives tau/t = 0.11-0.19 for all xi = 1 runs at RH 0.30 (tau 5-164 s), independent of
  kappa, eta0, eta_ref and seed: the late drying is set by the bulk organic-phase mobility, not by the outer shell.
- Interface-held water (f_w floor vs constant D): 0.6-0.7 % for all kappa except e0 kappa x2 (2 layers, 0.09 %); the
  floor is not simply proportional to the layer count.

## Next steps (open)
- [x] kappa_P x0.5, x2 sensitivity of L and t99 (interface width) — session 2
- [ ] adopt fixed D*eta (eta_ref 0.0086) for the paper: only the eta0 runs e1-e3 change (`*_er`); all eta_idx 0 runs
  (wet0, ramp, RH 0.5/0.8, xi 0.5, fine) already have eta_ref 0.0086
- [ ] layer count: needs a physical interface width (kappa from sigma ~ mN/m) or report only the coarse structure
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

# Water diffusion in viscous organic (+ AS) matrices: literature constraints at 290 K

Collected 2026-10-21 to constrain the water mobility of the three-mobility model (`sph1d.py --mob3 1`).
Code: `wdiff.py` (formulas), `water_diffusion.py` (comparison, fits, tables, `figures/fig8_water_diffusion.png`).

## Sources used

| Source | System | Method, T | What is used here |
| --- | --- | --- | --- |
| Lienhard et al. 2015, ACP 15, 13599, doi:10.5194/acp-15-13599-2015 | 3-MBTCA, alpha-pinene SOA, levoglucosan(/NH4HSO4), raffinose | EDB, 206-291 K | Table A1 + Appendix A2: modified Vignes D_w(T, x_w) |
| Price et al. 2014, ACP 14, 3817, doi:10.5194/acp-14-3817-2014 | sucrose, levoglucosan, raffinose/M5AS (7.7 wt % AS), MgSO4 | Raman D2O tracer, 296.65 K, 20-80 % RH | Table 1, Eq. 4: log10 D_w = a + b a_w + c a_w^2 + d a_w^3 |
| Price et al. 2015, Chem. Sci. 6, 4876, doi:10.1039/c5sc00685f | water-soluble alpha-pinene SOM | Raman, 240-280 K | main text only (ESI Table S2 could not be downloaded): at a_w 0.75 (~250 Pa s, 280 K) Stokes-Einstein underpredicts D_w by 2-3 orders, by >= 8 orders below a_w 0.3; Lienhard et al. 2015 note these values are ~2 orders above their SOA data |
| Holz et al. 2000, PCCP 2, 4740 | pure water | NMR + tracer, 0-100 C | D = 1.635e-8 [(T/215.05) - 1]^2.063 m2 s-1 (1.866e-9 at 290.15 K; reproduces 2.299e-9 at 298.15 K) |

Not obtained: Price 2015 ESI (Table S2, VFT fit), Zobrist et al. 2011 (sucrose) and Song et al. 2020 (saccharides, SE breakdown)
full texts. They would refine the SOM bracket and the sugar comparison, not the MBTCA/AS curves below.

## Formulas (m2 s-1)

Lienhard 2015: D_w = D_w(T,0)^(1 - x_w a) D_w(T,1)^(x_w a), ln a = (1 - x_w)^2 [A + 3B - 4B (1 - x_w)],
ln D_w(T,0) = ln D_w(Tg,0) + E_act/R (1/Tg - 1/T), A = a1 + a2 min(T, Ta), B = b1 + b2 min(T, Tb).
3-MBTCA: ln D_w(Tg,0) = -24.86 (cm2 s-1), E_act 64.5 kJ mol-1, Tg 305 K, a1 -5.033, a2 0.015, Ta 300 K, B = 0;
D_w(290.15 K, dry) = 4.35e-16. D_w(T,1) here is Holz 2000 (the paper uses Smith and Kay 1999).

Price 2014 at 296.65 K, (a, b, c, d): sucrose (-20.89, 25.92, -26.97, 13.25), levoglucosan (-18.41, 31.10, -44.43, 23.12),
raffinose/M5AS (-17.21, 24.00, -32.50, 17.02). Shifted to 290.15 K with an Arrhenius factor whose activation energy
is interpolated in a_w between 65 kJ mol-1 (dry, Lienhard's SOA value) and that of pure water (18 kJ mol-1): an assumption,
the factor is 0.55 (dry) to 0.85 (a_w 1).

## Results at 290.15 K (`dw_compare.csv`, `xi_w_fit.csv`)

- Two independent data sets agree on D_w(a_w): Lienhard 3-MBTCA (EDB, salt-free) and Price raffinose/M5AS (Raman,
  with AS): 1.8e-14 vs 1.6e-14 at a_w 0.2, 3.7e-11 for both at a_w 0.75.
- Against the model viscosity of the equilibrium MBTCA-rich phase (which holds 22-29 wt % AS, so it is far less viscous
  than binary MBTCA-water at the same a_w), the anchored fractional Stokes-Einstein exponent that fits a_w 0.2-0.8 is
  0.90 (3-MBTCA) and 0.84 (M5AS) for eta0 6e10, with 0.34-0.40 decades rms; 0.67 for the Price 2015 SOM bracket; 1.17
  (poor fit) for the Lienhard SOA; 1.38 for sucrose. With the eta0 1e16 viscosities the same data give 0.51-1.05:
  xi_w is not transferable between eta0 values.
- The pointwise implied xi_w varies 0.8-1.2 with a_w for 3-MBTCA and M5AS. A single xi_w is a compromise; the model
  should take D_W(a_w) directly (`dw_aw_290K.csv`, `sph1d.py --dw_model`).

## Caveats

- x_w(a_w) for alpha-pinene SOA uses kappa-Koehler with kappa 0.1, 1400 kg m-3, 150 g mol-1 (assumptions).
- Price 2014 fits are for 20-80 % RH; below a_w 0.2 they are extrapolations.
- D_w(a_w) of the AS-containing MBTCA-rich phase is taken equal to that of the matrices measured at the same a_w; no
  measurement of MBTCA + AS exists in these sources.
- The Price 2015 x100 bracket is meaningful at low a_w only (it reaches the pure-water value above a_w ~0.8).

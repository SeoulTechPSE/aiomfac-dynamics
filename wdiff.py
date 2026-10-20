"""Measured water diffusion coefficients D_w (m2 s-1) in aqueous organic (+ inorganic) matrices, for constraining the
water mobility of the three-mobility model (sph1d.py --mob3 1). Sources and validity: results/water_diffusion/README.md.

  pure_water(T)                     Holz et al. 2000, Speedy-Angell fit (reproduces 2.299e-9 at 298.15 K)
  water_viscosity(T)                Vogel equation for liquid water (1.08e-3 Pa s at 290 K)
  lienhard(system, T, x_w)          Lienhard et al. 2015 (ACP 15, 13599), EDB, modified Vignes in the water MOLE fraction
  price2014(system, a_w, T)         Price et al. 2014 (ACP 14, 3817), Raman isotope tracer at 296.65 K, cubic in a_w
"""
import numpy as np

R = 8.314462618


def pure_water(T):
    """self-diffusion of liquid water (Holz, Heil and Sacco 2000): D0 [(T/Ts) - 1]^gamma"""
    return 1.635e-8 * (T / 215.05 - 1.0) ** 2.063


def water_viscosity(T):
    """Vogel equation, Pa s"""
    return 2.414e-5 * 10 ** (247.8 / (T - 140.0))


# Lienhard et al. 2015, Table A1: ln[D_w(Tg,0)/(cm2 s-1)], E_act (kJ mol-1), Tg (K), a1, a2 (K-1), Ta, b1, b2, Tb;
# M = molar mass (kg mol-1) used for x_w (alpha-pinene SOA: 150 g mol-1, as assumed for the fit in the paper);
# T range = temperatures of the data behind each fit.
LIENHARD = {
    "levoglucosan":         dict(lnD=-27.06, E=157.9, Tg=249.0, a1=-52.77, a2=0.211, Ta=243.0, b1=8.561, b2=0.027, Tb=243.0, M=0.16214, T=(215, 253)),
    "levoglucosan/NH4HSO4": dict(lnD=-45.58, E=142.2, Tg=206.5, a1=-44.96, a2=0.185, Ta=243.0, b1=16.57, b2=-0.063, Tb=243.0, M=None, T=(200, 263)),
    "raffinose":            dict(lnD=-15.76, E=80.1, Tg=378.3, a1=-139.9, a2=0.347, Ta=273.5, b1=17.00, b2=0.00, Tb=273.5, M=0.50442, T=(210, 300)),
    "3-MBTCA":              dict(lnD=-24.86, E=64.5, Tg=305.0, a1=-5.033, a2=0.015, Ta=300.0, b1=0.00, b2=0.00, Tb=300.0, M=0.20418, T=(240, 300)),
    "alpha-pinene SOA":     dict(lnD=-26.60, E=65.5, Tg=270.0, a1=-18.31, a2=0.063, Ta=273.0, b1=-10.65, b2=0.039, Tb=273.0, M=0.150, T=(190, 300)),
}


def lienhard_dry(system, T):
    """D_w(T, 0): Arrhenius through D_w(Tg, 0), m2 s-1"""
    p = LIENHARD[system]
    return np.exp(p["lnD"] + p["E"] * 1e3 / R * (1 / p["Tg"] - 1 / T)) * 1e-4


def lienhard(system, T, x_w, wet=None):
    """D_w(T, x_w) = D_w(T,0)^(1 - x_w alpha) D_w(T,1)^(x_w alpha), ln alpha = (1-x_w)^2 [A + 3B - 4B (1-x_w)].
    wet: D_w(T,1); default the Holz pure-water value (the paper uses the VFT fit of Smith and Kay 1999)."""
    p = LIENHARD[system]
    A = p["a1"] + p["a2"] * min(T, p["Ta"])
    B = p["b1"] + p["b2"] * min(T, p["Tb"])
    x = np.asarray(x_w, float)
    xa = x * np.exp((1 - x) ** 2 * (A + 3 * B - 4 * B * (1 - x)))
    D1 = pure_water(T) if wet is None else wet
    return lienhard_dry(system, T) ** (1 - xa) * D1 ** xa


# Price et al. 2014, Table 1: log10(D_w/(m2 s-1)) = a + b a_w + c a_w^2 + d a_w^3 at 23.5 C (20-80 % RH data).
# raffinose/M5AS = 50 wt % raffinose pentahydrate, 42.3 % C3-C6 dicarboxylic acids, 7.7 % ammonium sulfate.
PRICE2014 = {
    "sucrose":         (-20.89, 25.92, -26.97, 13.25),
    "levoglucosan":    (-18.41, 31.10, -44.43, 23.12),
    "raffinose/M5AS":  (-17.21, 24.00, -32.50, 17.02),
}
T_PRICE2014 = 296.65


def price2014(system, a_w, T=None, E_dry=65.0e3):
    """D_w(a_w) at 296.65 K; with T, shifted by an Arrhenius factor whose activation energy is interpolated in a_w
    between E_dry (default 65 kJ mol-1, Lienhard's SOA value) and that of pure water near room temperature (an assumption,
    the factor is 0.55-0.85 between 296.65 and 290 K)."""
    a, b, c, d = PRICE2014[system]
    aw = np.asarray(a_w, float)
    D = 10 ** (a + b * aw + c * aw ** 2 + d * aw ** 3)
    if T is None:
        return D
    E_w = R * np.log(pure_water(T_PRICE2014) / pure_water(T)) / (1 / T - 1 / T_PRICE2014)
    E = (1 - aw) * E_dry + aw * E_w
    return D * np.exp(-E / R * (1 / T - 1 / T_PRICE2014))

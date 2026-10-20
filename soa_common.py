"""Shared definitions for the SOA-proxy + AS system (MBTCA by default; env SOA=DTAA/terpenylic for alternatives)."""
import os, re
import numpy as np
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.viscosity import predict_tg_derieux2018, pure_organic_viscosity_vtf
from indigo import Indigo
T = 290.15
CAND = {"MBTCA": "OC(=O)CC(C(=O)O)C(C)(C(=O)O)C",
        "DTAA": "CC(=O)OC(C)(C)C(CC(=O)O)CC(=O)O",
        "terpenylic": "OC(=O)C1(C)CCC(=O)C1CC(=O)O"}
NAME = os.environ.get("SOA", "MBTCA")
SMILES = CAND[NAME]
_m = Indigo().loadMolecule(SMILES)
_f = {k: int(v or 1) for k, v in re.findall(r"([A-Z][a-z]?)(\d*)", _m.grossFormula().replace(" ", ""))}
nC, nH, nO = _f["C"], _f["H"], _f.get("O", 0)
M_ORG = _m.molecularWeight() * 1e-3
MW, M_AS = 0.01801528, 0.13214
ORG = Component(2, NAME, tuple(smiles_to_components([SMILES]).components[1].subgroups))
WATER = Component(1, "Water", ((16, 1),))
AS = Component(3, "AS", ((204, 2), (261, 1)))
MODEL = ActivityModel([WATER, ORG, AS])
TG = predict_tg_derieux2018(nC, nH, nO)
ETA0 = float(os.environ.get("ETA0", pure_organic_viscosity_vtf(T, TG)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", NAME)
os.makedirs(OUT, exist_ok=True)

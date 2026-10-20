cd "$(dirname "$0")"
S=results/MBTCA
B="--system $S --rh_ref 0.75 --rh0 0.85 --n 600 --grade 10 --dtmax 2 --nsnap 50"
# A: final RH (Stokes-Einstein, D2018 eta0)
[ -f runs/soa_rh0.80_xi1_e0_meta.json ] || python3 sph1d.py $B --rh 0.80 --visc 1 --tend_s 20  --tag soa_rh0.80_xi1_e0 > runs/soa_rh0.80_xi1_e0.log 2>&1
[ -f runs/soa_rh0.50_xi1_e0_meta.json ] || python3 sph1d.py $B --rh 0.50 --visc 1 --tend_s 100 --tag soa_rh0.50_xi1_e0 > runs/soa_rh0.50_xi1_e0.log 2>&1
[ -f runs/soa_rh0.30_xi1_e0_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --tend_s 200 --tag soa_rh0.30_xi1_e0 > runs/soa_rh0.30_xi1_e0.log 2>&1
# C: mobility law at RH 0.30
[ -f runs/soa_rh0.30_xi0_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 0   --tend_s 200 --tag soa_rh0.30_xi0 > runs/soa_rh0.30_xi0.log 2>&1
[ -f runs/soa_rh0.30_xi0.5_e0_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 0.5 --tend_s 200 --tag soa_rh0.30_xi0.5_e0 > runs/soa_rh0.30_xi0.5_e0.log 2>&1
# B: dry viscosity (oligomer content) at RH 0.30
[ -f runs/soa_rh0.30_xi1_e1_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --eta_idx 1 --tend_s 400 --tag soa_rh0.30_xi1_e1 > runs/soa_rh0.30_xi1_e1.log 2>&1
[ -f runs/soa_rh0.30_xi1_e2_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --eta_idx 2 --tend_s 1000 --tag soa_rh0.30_xi1_e2 > runs/soa_rh0.30_xi1_e2.log 2>&1
[ -f runs/soa_rh0.30_xi1_e3_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --eta_idx 3 --tend_s 2000 --tag soa_rh0.30_xi1_e3 > runs/soa_rh0.30_xi1_e3.log 2>&1
# D: neutral surface
[ -f runs/soa_rh0.30_xi1_e0_wet0_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --wet 0 --tend_s 200 --tag soa_rh0.30_xi1_e0_wet0 > runs/soa_rh0.30_xi1_e0_wet0.log 2>&1
# E: slow ramp (60 s) vs step
[ -f runs/soa_rh0.30_xi1_e0_ramp60_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --ramp_s 60 --tend_s 260 --tag soa_rh0.30_xi1_e0_ramp60 > runs/soa_rh0.30_xi1_e0_ramp60.log 2>&1
# F: mesh convergence
[ -f runs/soa_rh0.30_xi1_e0_fine_meta.json ] || python3 sph1d.py --system $S --rh_ref 0.75 --rh0 0.85 --n 1200 --grade 20 --dtmax 2 --nsnap 50 --rh 0.30 --visc 1 --tend_s 200 --tag soa_rh0.30_xi1_e0_fine > runs/soa_rh0.30_xi1_e0_fine.log 2>&1

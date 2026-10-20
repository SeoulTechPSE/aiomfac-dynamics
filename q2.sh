cd "$(dirname "$0")"
S=results/MBTCA
B="--system $S --rh_ref 0.75 --rh0 0.85 --n 600 --grade 10 --dtmax 2 --nsnap 50 --wmin 1e-6 --lfloor 1e-7"
[ -f runs/soa_rh0.30_xi1_e1_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --eta_idx 1 --tend_s 400 --tag soa_rh0.30_xi1_e1 > runs/soa_rh0.30_xi1_e1.log 2>&1
[ -f runs/soa_rh0.30_xi1_e2_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --eta_idx 2 --tend_s 1000 --tag soa_rh0.30_xi1_e2 > runs/soa_rh0.30_xi1_e2.log 2>&1
[ -f runs/soa_rh0.30_xi1_e3_meta.json ] || python3 sph1d.py $B --rh 0.30 --visc 1 --eta_idx 3 --tend_s 2000 --tag soa_rh0.30_xi1_e3 > runs/soa_rh0.30_xi1_e3.log 2>&1
[ -f runs/soa_rh0.30_xi1_e0_fine_meta.json ] || python3 sph1d.py --system $S --rh_ref 0.75 --rh0 0.85 --n 1200 --grade 20 --dtmax 2 --nsnap 50 --wmin 1e-6 --lfloor 1e-7 --rh 0.30 --visc 1 --tend_s 200 --tag soa_rh0.30_xi1_e0_fine > runs/soa_rh0.30_xi1_e0_fine.log 2>&1
[ -f runs/peg_sph_D1e-10_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-10 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_sph_D1e-10 > runs/peg_sph_D1e-10.log 2>&1
[ -f runs/peg_cyl_D1e-10_meta.json ] || python3 sph1d.py --dim 2 --n 1000 --D 1e-10 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_cyl_D1e-10 > runs/peg_cyl_D1e-10.log 2>&1
[ -f runs/peg_sph_D1e-09_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-9 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_sph_D1e-09 > runs/peg_sph_D1e-09.log 2>&1
[ -f runs/peg_sph_D1e-11_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-11 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_sph_D1e-11 > runs/peg_sph_D1e-11.log 2>&1
[ -f runs/peg_sph_D1e-10_wet0_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-10 --rh0 0.92 --rh 0.80 --wet 0 --nsnap 80 --tag peg_sph_D1e-10_wet0 > runs/peg_sph_D1e-10_wet0.log 2>&1

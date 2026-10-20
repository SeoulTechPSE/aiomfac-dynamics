cd "$(dirname "$0")"
[ -f runs/peg_sph_D1e-10_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-10 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_sph_D1e-10 > runs/peg_sph_D1e-10.log 2>&1
[ -f runs/peg_cyl_D1e-10_meta.json ] || python3 sph1d.py --dim 2 --n 1000 --D 1e-10 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_cyl_D1e-10 > runs/peg_cyl_D1e-10.log 2>&1
[ -f runs/peg_sph_D1e-09_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-9 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_sph_D1e-09 > runs/peg_sph_D1e-09.log 2>&1
[ -f runs/peg_sph_D1e-11_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-11 --rh0 0.92 --rh 0.80 --wet 2 --nsnap 80 --tag peg_sph_D1e-11 > runs/peg_sph_D1e-11.log 2>&1
[ -f runs/peg_sph_D1e-10_wet0_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-10 --rh0 0.92 --rh 0.80 --wet 0 --nsnap 80 --tag peg_sph_D1e-10_wet0 > runs/peg_sph_D1e-10_wet0.log 2>&1
[ -f runs/peg_sph_D1e-09_wet0_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-9 --rh0 0.92 --rh 0.80 --wet 0 --nsnap 80 --tag peg_sph_D1e-09_wet0 > runs/peg_sph_D1e-09_wet0.log 2>&1
[ -f runs/peg_sph_D1e-11_wet0_meta.json ] || python3 sph1d.py --dim 3 --n 1000 --D 1e-11 --rh0 0.92 --rh 0.80 --wet 0 --nsnap 80 --tag peg_sph_D1e-11_wet0 > runs/peg_sph_D1e-11_wet0.log 2>&1

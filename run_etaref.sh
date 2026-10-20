cd "$(dirname "$0")"
# eta0 sensitivity with D*eta fixed (Stokes-Einstein): eta_ref = initial-state viscosity of the eta_idx 0 run (0.0086 Pa s).
# The first eta0 runs (soa_rh0.30_xi1_e1..3) normalised D at each run's own initial viscosity (0.010, 0.0128, 0.0163 Pa s).
S=results/MBTCA
B="--system $S --rh_ref 0.75 --rh0 0.85 --n 600 --grade 10 --dtmax 2 --nsnap 50 --wmin 1e-6 --lfloor 1e-7 --rh 0.30 --visc 1 --eta_ref 0.008603994797084933"
run() { tag=$1; shift; [ -f runs/${tag}_meta.json ] || python3 sph1d.py $B "$@" --tag $tag > runs/$tag.log 2>&1; }
run soa_rh0.30_xi1_e1_er --eta_idx 1 --tend_s 400 &
run soa_rh0.30_xi1_e2_er --eta_idx 2 --tend_s 1000 &
run soa_rh0.30_xi1_e3_er --eta_idx 3 --tend_s 2000 &
wait

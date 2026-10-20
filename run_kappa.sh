cd "$(dirname "$0")"
# Interface-width sensitivity: kappa_P, kappa_S x0.5 and x2 (H scales with sqrt(kappa) inside sph1d.py), all on the
# fine mesh (layer count differs between n 600 and n 1200), plus a noise-seed control. Runs in parallel; ~15-60 min.
S=results/MBTCA
B="--system $S --rh_ref 0.75 --rh0 0.85 --n 1200 --grade 20 --dtmax 2 --nsnap 50 --wmin 1e-6 --lfloor 1e-7 --rh 0.30 --visc 1"
run() { tag=$1; shift; [ -f runs/${tag}_meta.json ] || python3 sph1d.py $B "$@" --tag $tag > runs/$tag.log 2>&1; }
run soa_rh0.30_xi1_e0_fine_k0.5 --kP 1e-4 --kS 1e-3 --tend_s 200 &
run soa_rh0.30_xi1_e0_fine_k2   --kP 4e-4 --kS 4e-3 --tend_s 200 &
run soa_rh0.30_xi1_e0_fine_s2   --rseed 2           --tend_s 200 &
run soa_rh0.30_xi1_e3_fine      --eta_idx 3                     --tend_s 2000 &
run soa_rh0.30_xi1_e3_fine_k0.5 --eta_idx 3 --kP 1e-4 --kS 1e-3 --tend_s 2000 &
run soa_rh0.30_xi1_e3_fine_k2   --eta_idx 3 --kP 4e-4 --kS 4e-3 --tend_s 2000 &
wait

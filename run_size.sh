cd "$(dirname "$0")"
# Size dependence. sph1d.py is nondimensional (X = r/R, time unit R0^2/D_ref, kappa in units of R0^2), so changing only
# --R0_um is an exact rescaling of time (the "naive" run checks this). With a fixed PHYSICAL interface width the
# nondimensional kappa scales as (10 um / R0)^2 (H follows sqrt(kappa) inside sph1d.py); that is the real size test.
# eta0 6e10 (eta_ref = its own initial viscosity 0.0086 Pa s = the D*eta-fixed convention). dtmax and tend scale with R0^2.
S=results/MBTCA
B="--system $S --rh_ref 0.75 --rh0 0.85 --grade 10 --nsnap 50 --wmin 1e-6 --lfloor 1e-7 --rh 0.30 --visc 1"
run() { tag=$1; shift; [ -f runs/${tag}_meta.json ] || python3 sph1d.py $B "$@" --tag $tag > runs/$tag.log 2>&1; }
run soa_rh0.30_xi1_e0_R1       --R0_um 1  --n 600  --kP 2e-2     --kS 2e-1     --dtmax 0.02 --tend_s 2 &
run soa_rh0.30_xi1_e0_R1_naive --R0_um 1  --n 600  --kP 2e-4     --kS 2e-3     --dtmax 0.02 --tend_s 2 &
run soa_rh0.30_xi1_e0_R3       --R0_um 3  --n 600  --kP 2.222e-3 --kS 2.222e-2 --dtmax 0.18 --tend_s 18 &
run soa_rh0.30_xi1_e0_R30      --R0_um 30 --n 3000 --kP 2.222e-5 --kS 2.222e-4 --dtmax 18   --tend_s 1800 &
wait

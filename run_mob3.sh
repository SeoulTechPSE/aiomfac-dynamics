cd "$(dirname "$0")"
# Three-mobility (fast-mode) model: organic and AS follow Stokes-Einstein (xi = 1), water its own exponent xi_w
# (fractional Stokes-Einstein), all normalised at eta_ref = 0.0086 Pa s (D*eta fixed) with D_W,ref = D_ref (--dW 1).
# xi0_m3: constant mobilities, i.e. the effect of the fast-mode matrix alone (vs soa_rh0.30_xi0).
S=results/MBTCA
B="--system $S --rh_ref 0.75 --rh0 0.85 --n 600 --grade 10 --dtmax 2 --nsnap 50 --wmin 1e-6 --lfloor 1e-7 --rh 0.30 --mob3 1 --eta_ref 0.008603994797084933"
run() { tag=$1; shift; [ -f runs/${tag}_meta.json ] || python3 sph1d.py $B "$@" --tag $tag > runs/$tag.log 2>&1; }
run soa_rh0.30_xi0_m3            --visc 0                      --tend_s 200 &
run soa_rh0.30_xi1_e0_m3w1       --visc 1 --xi_w 1             --tend_s 200 &
run soa_rh0.30_xi1_e0_m3w0.75    --visc 1 --xi_w 0.75          --tend_s 200 &
run soa_rh0.30_xi1_e0_m3w0.5     --visc 1 --xi_w 0.5           --tend_s 200 &
run soa_rh0.30_xi1_e3_m3w1       --visc 1 --xi_w 1    --eta_idx 3 --tend_s 2000 &
run soa_rh0.30_xi1_e3_m3w0.75    --visc 1 --xi_w 0.75 --eta_idx 3 --tend_s 2000 &
run soa_rh0.30_xi1_e3_m3w0.5     --visc 1 --xi_w 0.5  --eta_idx 3 --tend_s 2000 &
wait
# Water mobility anchored at pure water (self-diffusion ~1.9e-9 m2/s, viscosity ~1.08 mPa s at 290 K, approximate)
# and extrapolated to eta_ref with the same fractional Stokes-Einstein exponent:
#   dW = D_W(eta_ref)/D_ref = 19 (1.08e-3/8.604e-3)^xi_w = 2.39 (xi_w 1), 4.02 (0.75), 6.73 (0.5).
# The dW = 1 runs above are a prefactor sensitivity (and the xi_w >= 0.75 ones did not reach equilibrium by tend).
run soa_rh0.30_xi1_e0_m3aw1      --visc 1 --xi_w 1    --dW 2.39 --tend_s 1000 &
run soa_rh0.30_xi1_e0_m3aw0.75   --visc 1 --xi_w 0.75 --dW 4.02 --tend_s 1000 &
run soa_rh0.30_xi1_e0_m3aw0.5    --visc 1 --xi_w 0.5  --dW 6.73 --tend_s 1000 &
run soa_rh0.30_xi1_e3_m3aw1      --visc 1 --xi_w 1    --dW 2.39 --eta_idx 3 --tend_s 10000 &
run soa_rh0.30_xi1_e3_m3aw0.75   --visc 1 --xi_w 0.75 --dW 4.02 --eta_idx 3 --tend_s 10000 &
run soa_rh0.30_xi1_e3_m3aw0.5    --visc 1 --xi_w 0.5  --dW 6.73 --eta_idx 3 --tend_s 10000 &
wait

cd "$(dirname "$0")"
# Three-mobility model with a MEASURED water diffusivity D_W(a_w) at the local water activity (--dw_model; tables from
# water_diffusion.py, 290.15 K): Lienhard 2015 3-MBTCA, Price 2014 raffinose/M5AS (with AS), and the alpha-pinene SOA
# brackets (Lienhard 2015, lower; Price 2015 ~x100, upper). Organic and AS keep Stokes-Einstein (xi 1), D*eta fixed.
S=results/MBTCA
B="--system $S --rh_ref 0.75 --rh0 0.85 --n 600 --grade 10 --dtmax 2 --nsnap 50 --wmin 1e-6 --lfloor 1e-7 --rh 0.30 --visc 1 --mob3 1 --eta_ref 0.008603994797084933"
run() { tag=$1; shift; [ -f runs/${tag}_meta.json ] || python3 sph1d.py $B "$@" --tag $tag > runs/$tag.log 2>&1; }
run soa_rh0.30_xi1_e0_dw_mbtca        --dw_model mbtca        --tend_s 2000 &
run soa_rh0.30_xi1_e0_dw_m5as         --dw_model m5as         --tend_s 2000 &
run soa_rh0.30_xi1_e0_dw_som_price    --dw_model som_price    --tend_s 2000 &
run soa_rh0.30_xi1_e0_dw_soa_lienhard --dw_model soa_lienhard --tend_s 5000 &
run soa_rh0.30_xi1_e3_dw_mbtca        --dw_model mbtca --eta_idx 3 --tend_s 10000 &
run soa_rh0.30_xi1_e3_dw_m5as         --dw_model m5as  --eta_idx 3 --tend_s 10000 &
wait

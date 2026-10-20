cd "$(dirname "$0")"
# Validation against paper6 dolfinx (doc "수치 기법" table). These ran before --implicit existed, i.e. with the
# explicit Rdot of m3g_drying_sph.py: --implicit 0 reproduces them.
# planar interface: sigma_num (PEG 0.00568 vs M3e 0.00576) and the MBTCA value used by the SOA runs
[ -f runs/planar_peg_meta.json ] || python3 sph1d.py --dim 1 --n 400 --kP 2e-4 --kS 2e-3 --tend_s 0.02 --dtmax 2e-4 --nsnap 60 --implicit 0 --quiet --tag planar_peg > runs/planar_peg.log 2>&1
[ -f runs/planar_mbtca_meta.json ] || python3 sph1d.py --system results/MBTCA --rh_ref 0.75 --dim 1 --n 400 --kP 2e-4 --kS 2e-3 --tend_s 0.02 --dtmax 2e-4 --nsnap 60 --implicit 0 --quiet --tag planar_mbtca > runs/planar_mbtca.log 2>&1
# 2D disc vs the paper6 m3g states (kP 8e-4, sigma_num 0.00589 as in m3g)
for D in 1e-10 1e-11; do
  [ -f runs/cyl_val_D${D}_meta.json ] || python3 sph1d.py --dim 2 --n 400 --D $D --rh0 0.92 --rh 0.80 --wet 2 --kP 8e-4 --kS 8e-3 --sigma_num 0.00589 --tend_s 0.55 --nsnap 120 --implicit 0 --quiet --tag cyl_val_D$D > runs/cyl_val_D$D.log 2>&1
done

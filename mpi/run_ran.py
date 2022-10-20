import sys, os

sys.path.insert(0, '/data/groups/jeltema/zhou/y3clshear/src')

from measure_clshear import cut_catalog, cut_source, R_bin, run_DS

from constants import Y3_MASTER_DIR, Y3_MASTER_NAME, PROJECT_DIR

import numpy as np

from mpi4py import MPI
import sys
import time as t

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from tqdm import tqdm, trange

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    h = 0.7
    cosmo = FlatLambdaCDM(H0=100. * h, Om0=0.3)

    zbin = np.linspace(0, 2, 500)
    distbin = (cosmo.comoving_distance(zbin).value) * 10**6 * h
    dist_func = interp1d(zbin, distbin, fill_value="extrapolate")

    center_data = np.loadtxt(os.path.join(
        PROJECT_DIR, './data/kmeans_centers_npix100_desy3.dat'),
                             unpack=True)

    print("Creating R bins")
    R, Rmid = R_bin(Rmin, Rmax, NBINS)

    print("Start run_ran.py")
    zmin = float(sys.argv[1])
    zmax = float(sys.argv[2])
    run_id = int(sys.argv[3])
    outdir = sys.argv[4]
    lmin = float(sys.argv[5])
    lmax = float(sys.argv[6])
    Rmin = float(sys.argv[7])
    Rmax = float(sys.argv[8])
    NBINS = int(sys.argv[9])
    select_src = sys.argv[10]

    ran_dir = os.path.join(
        PROJECT_DIR,
        './catalogs/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_randcat_z0.10-0.95_lgt020_vl02.fit'
    )

    src_dir = os.path.join(Y3_MASTER_DIR, Y3_MASTER_NAME)
    dat_save_dir = PROJECT_DIR

    z_bin_edge = np.linspace(zmin, zmax, 50)
    z_bin_intervals = np.array([(z_bin_edge[i], z_bin_edge[i + 1])
                                for i in np.arange(len(z_bin_edge) - 1)])
    lam_bin_edge = np.linspace(lmin, lmax, 50)
    lam_bin_intervals = np.array([(lam_bin_edge[i], lam_bin_edge[i + 1])
                                  for i in np.arange(len(lam_bin_edge) - 1)])

    zlam_list = np.array([(zinteval, lam_interval)
                          for zinteval in z_bin_intervals
                          for lam_interval in lam_bin_intervals])

    N_RANDOMS = 20

    len_cluster = zlam_list.shape[0]
    N_step = round(len_cluster / mpi_size)

    if mpi_rank == mpi_size - 1:
        zlam_list = zlam_list[N_step * mpi_rank:N_step * (mpi_rank + 1) +
                              len_cluster % mpi_size]
        N_zlam = len(zlam_list)
    else:
        zlam_list = zlam_list[N_step * mpi_rank:]
        N_zlam = len(zlam_list)

    empty_array = np.zeros((N_zlam * N_RANDOMS, len(Rmid)))
    empty_pz_array = np.zeros(N_zlam * N_RANDOMS, len(Rmid), 301)

    ran_top_array, ran_top_im_array, ran_bottom_array, ran_weight_array = empty_array, empty_array, empty_array, empty_array
    ran_pz_bpz_array, ran_pz_dnf_array = empty_pz_array, empty_pz_array

    for i, zlam in enumerate(zlam_list):

        zmin, zmax = zlam[0]
        lmin, lmax = zlam[1]

        dict_rand_cut = {
            "ZTRUE": [zmin, zmax],
            "LAMBDA_IN": [lmin, lmax]
        }  ## dictionary for the randoms cut

        print("Start cutting randoms.")

        ran_cat = cut_catalog(ran_dir,
                              dict_rand_cut,
                              random=True,
                              N_randoms=N_RANDOMS)

        print("Start calculating randoms.")

        for run_id in trange(N_RANDOMS):
            
            ran_top, ran_top_im, ran_bottom, ran_weight, ran_pz_bpz, ran_pz_dnf = run_DS(
                RA_ran[run_id],
                DEC_ran[run_id],
                Z_ran[run_id],
                R,
                src_cat,
                select_src='dnf',
                comoving=True,
                cut_with_jk=True,
                cosmo=cosmo,
                center_data=center_data,
                h=h,
                NBINS=NBINS)
            
            ran_top_array[i * N_RANDOMS + run_id] = ran_top
            ran_top_im_array[i * N_RANDOMS + run_id] = ran_top_im
            ran_bottom_array[i * N_RANDOMS + run_id] = ran_bottom
            ran_weight_array[i * N_RANDOMS + run_id] = ran_weight
            ran_pz_bpz_array[i * N_RANDOMS + run_id] = ran_pz_bpz
            ran_pz_dnf_array[i * N_RANDOMS + run_id] = ran_pz_dnf

    save_dir = "/data/groups/jeltema/zhou/y3clshear/output"
    out_path = os.path.join(save_dir, f"output_random_{mpi_rank}.npz")

    np.savez(
        out_path,
        top=ran_top_array,
        top_im=ran_top_im_array,
        bottom=ran_bottom_array,
        weight=ran_weight_array,
        pz_bpz=ran_pz_bpz_array,
        pz_dnf_array=ran_pz_dnf_array,
    )

    print(f"Successfully dumped rank {mpi_rank}")

    t2 = t.time()
    print("Time spent calculating", (t2 - t1))

    # Need some write function.0

    # print('DS=', top / bottom,
    #       'time taken for calculating DSigma = %.2f seconds' % (t2 - t1))
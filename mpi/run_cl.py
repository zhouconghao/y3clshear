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

    h = 0.7
    cosmo = FlatLambdaCDM(H0=100. * h, Om0=0.3)

    zbin = np.linspace(0, 2, 500)
    distbin = (cosmo.comoving_distance(zbin).value) * 10**6 * h
    dist_func = interp1d(zbin, distbin, fill_value="extrapolate")

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    center_data = np.loadtxt(os.path.join(
        PROJECT_DIR, './data/kmeans_centers_npix100_desy3.dat'),
                             unpack=True)

    print("Start run_cl.py")
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

    len_dir = os.path.join(
        PROJECT_DIR,
        './catalogs/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl02_catalog.fit'
    )
    ran_dir = os.path.join(
        PROJECT_DIR,
        './catalogs/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_randcat_z0.10-0.95_lgt020_vl02.fit'
    )
    src_dir = os.path.join(Y3_MASTER_DIR, Y3_MASTER_NAME)
    dat_save_dir = PROJECT_DIR

    dict_lens_cut = {
        "Z_LAMBDA": [zmin, zmax],
        "LAMBDA_CHISQ": [lmin, lmax]
    }  ## dictionary for the lenses cut

    dict_rand_cut = {
        "ZTRUE": [zmin, zmax],
        "LAMBDA_IN": [lmin, lmax]
    }  ## dictionary for the randoms cut

    print("Creating R bins")
    R, Rmid = R_bin(Rmin, Rmax, NBINS)

    len_cat = cut_catalog(len_dir, dict_lens_cut)
    N_lens_total = len(len_cat)
    N_step = round(N_lens_total / mpi_size)

    if mpi_rank != (mpi_size - 1):
        len_cat = len_cat[mpi_rank * N_step:(mpi_rank + 1) * N_step]
        N_lens = len(len_cat)
    else:
        len_cat = len_cat[mpi_rank * N_step:]
        N_lens = len(len_cat)

    RA_len = len_cat["RA"]
    DEC_len = len_cat["DEC"]
    Z_len = len_cat["Z"]
    id_len = len_cat["mem_match_id"]

    print("Start cutting source.")

    t1 = t.time()

    src_cat = cut_source(src_dir,
                         select_src,
                         run_jk_kmeans=True,
                         jk_dir=dat_save_dir + "jk_src_%s" % (select_src),
                         run_healpy=True,
                         func_dist=dist_func)
    t2 = t.time()

    print("Time spent cutting source ", (t2 - t1))

    t1 = t.time()

    top_array = np.zeros((N_lens, len(Rmid)))
    top_im_array = np.zeros((N_lens, len(Rmid)))
    bottom_array = np.zeros((N_lens, len(Rmid)))
    weight_array = np.zeros((N_lens, len(Rmid)))
    pz_bpz_array = np.zeros((N_lens, len(Rmid), 301))
    pz_dnf_array = np.zeros((N_lens, len(Rmid), 301))

    for run_id in trange(N_lens):
        top, top_im, bottom, weight, pz_bpz, pz_dnf = run_DS(
            RA_len[run_id],
            DEC_len[run_id],
            Z_len[run_id],
            R,
            src_cat,
            select_src='dnf',
            comoving=True,
            cut_with_jk=True,
            cosmo=cosmo,
            center_data=center_data,
            h=h,
            NBINS=NBINS)

        top_array[run_id, :] = top
        top_im_array[run_id, :] = top_im
        bottom_array[run_id, :] = bottom
        weight_array[run_id, :] = weight
        pz_bpz_array[run_id, :] = pz_bpz
        pz_dnf_array[run_id, :] = pz_dnf

    save_dir = "/data/groups/jeltema/zhou/y3clshear/output"
    out_path = os.path.join(save_dir, f"output_{mpi_rank}.npz")

    np.savez(
        out_path,
        id=id_len,
        top=top_array,
        top_im=top_im_array,
        bottom=bottom_array,
        weight=weight_array,
        pz_bpz=pz_bpz_array,
        pz_dnf_array=pz_dnf_array,
    )

    print(f"Successfully dumped rank {mpi_rank}")

    t2 = t.time()
    print("Time spent calculating", (t2 - t1))

    # Need some write function.0

    # print('DS=', top / bottom,
    #       'time taken for calculating DSigma = %.2f seconds' % (t2 - t1))
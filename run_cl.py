from measure_clshear import cut_catalog, cut_source, R_bin, run_DS
import numpy as np

from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD
import sys

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()
print("MPI size is", mpi_size)

import kmeans_radec

center_data = np.loadtxt('./data/kmeans_centers_npix100_desy3.dat',
                         unpack=True)

km = kmeans_radec.KMeans(
    np.loadtxt('./data/kmeans_centers_npix100_desy3.dat')
)  ### could be any kmeans JK patch centers  ==> this is for cutting galaxies fast

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

h = 0.7
cosmo = FlatLambdaCDM(H0=100. * h, Om0=0.3)

if __name__ == "__main__":

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

    len_dir = './catalogs/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl02_catalog.fit'
    ran_dir = './catalogs/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_randcat_z0.10-0.95_lgt020_vl02.fit'
    src_dir = '/project/projectdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5'
    dat_save_dir = '/global/cscratch1/sd/zchusre/'

    dict_lens_cut = {
        "Z_LAMBDA": [zmin, zmax],
        "LAMBDA_CHISQ": [lmin, lmax]
    }  ## dictionary for the lenses cut

    dict_rand_cut = {
        "ZTRUE": [zmin, zmax],
        "LAMBDA_IN": [lmin, lmax]
    }  ## dictionary for the randoms cut

    R, Rmid = R_bin(Rmin, Rmax, NBINS)

    t1 = t.time()

    if mpi_rank == 0:
        len_cat = cut_catalog(len_dir, dict_lens_cut)
        N_lens = len(len_cat)
        N_step = round(N_lens / mpi_size)

        for i in mpi_size:
            if i != mpi_size - 1:
                comm.send(len_cat[i * N_step, (i + 1) * N_step],
                          dest=i,
                          tag=99)

            else:
                comm.send(len_cat[(i * N_step), :], dest=i, tag=99)

    else:
        len_cat = comm.recv(source=0, tag=99)

    N_rand = 20 * N_lens

    ran_cat = cut_catalog(ran_dir,
                          dict_rand_cut,
                          random=True,
                          N_randoms=N_rand)

    t2 = t.time()
    print('reading lenses and randoms took %.2f seconds' % (t2 - t1))

    RA_len = len_cat["RA"]
    DEC_len = len_cat["DEC"]
    Z_len = len_cat["Z"]

    RA_ran = ran_cat["RA"]
    DEC_ran = ran_cat["DEC"]
    Z_ran = ran_cat["ZTRUE"]

    t1 = t.time()

    print("Start cutting source.")
    src_cat = cut_source(src_dir,
                         select_src,
                         run_jk_kmeans=False,
                         jk_dir=dat_save_dir + "jk_src_%s" % (select_src),
                         run_healpy=True)

    t2 = t.time()

    print('reading and preparing source catalog took %.2f seconds' % (t2 - t1))

    t1 = t.time()

    top, top_im, bottom, weight, pz_bpz, pz_dnf = run_DS(RA_ran[run_id],
                                                         DEC_ran[run_id],
                                                         Z_ran[run_id],
                                                         R,
                                                         src_cat,
                                                         select_src='bin1',
                                                         comoving=True,
                                                         cut_with_jk=True)

    # Need some write function.

    t2 = t.time()

    print('DS=', top / bottom,
          'time taken for calculating DSigma = %.2f seconds' % (t2 - t1))
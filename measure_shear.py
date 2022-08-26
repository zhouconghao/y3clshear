# based on code from Tae

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import h5py as h5
import astropy.io.fits as pf
import numpy as np
import healpy as hp
from astropy import units as u
from astropy import cosmology
from astropy.cosmology import FlatLambdaCDM
h = 0.7
cosmo = FlatLambdaCDM(H0=100.*h, Om0=0.3)
import os
import sys
from scipy.interpolate import interp1d
from scipy import integrate
import treecorr
import kmeans_radec
center_data = np.loadtxt('./data/kmeans_centers_npix100_desy3.dat',unpack=True)
km = kmeans_radec.KMeans(np.loadtxt('./data/kmeans_centers_npix100_desy3.dat'))  ### could be any kmeans JK patch centers  ==> this is for cutting galaxies fast
from astropy.coordinates import SkyCoord
import time as t

def R_bin(Rm,RM,N):
    R = np.logspace(np.log10(Rm),np.log10(RM),N+1)
    Rmid = np.sqrt(R[1:]*R[:-1])
    return R, Rmid

def cut_catalog(catdir_name, dict_cut, mask=None, nside=None, nest=True, dict_radec_name=None, random=False, N_randoms=None): ## only for fits files
    data = pf.open(catdir_name)[1].data
    var_list = dict_cut.keys() ##dict_cut: 'keys' for the dictionary = column name in the fits file for selection // 'values' for the dictionary: min and max for that quantity
    for i, var in enumerate(var_list):
        if i==0:
            ind = (data[var]>dict_cut[var][0])*(data[var]<=dict_cut[var][1])
        else:
            ind *= (data[var]>dict_cut[var][0])*(data[var]<=dict_cut[var][1])
    data = data[ind]
    if mask is not None:
        RAtem = data[dict_radec_name['RA']]
        DECtem = data[dict_radec_name['DEC']]
        THETA = (90.0 - DECtem)*np.pi/180.    ### THETA, PHI ==> HEALPIX PIXEL
        PHI = RAtem*np.pi/180.
        PIX = hp.ang2pix(nside, THETA, PHI, nest=nest)
        data = data[(mask[PIX]==1)]
    if random==True:
        subsample = np.random.choice(len(data), size=N_randoms, replace=False)
        data = data[subsample]
    return data

def dist_comoving():
    zbin = np.linspace(0,2,2001)
    distbin = (cosmo.comoving_distance(zbin).value)*10**6*h
    func_dist = interp1d(zbin,distbin,fill_value='extrapolate')
    return func_dist

def cut_source(catdir_name, select_src, run_jk_kmeans=False, jk_dir=None, run_healpy=True):
    f = h5.File(catdir_name,'r')
    if (select_src == "dnf")|(select_src == "bpz"):
        sel_src = np.array(f['index/select'])
        mask_1p = np.array(f['index/select_1p'])
        mask_1m = np.array(f['index/select_1m'])
        mask_2p = np.array(f['index/select_2p'])
        mask_2m = np.array(f['index/select_2m'])
    elif select_src == "bin1":
        sel_src = np.array(f['index/select_bin1'])
        mask_1p = np.array(f['index/select_1p_bin1'])
        mask_1m = np.array(f['index/select_1m_bin1'])
        mask_2p = np.array(f['index/select_2p_bin1'])
        mask_2m = np.array(f['index/select_2m_bin1'])
    elif select_src == "bin2":
        sel_src = np.array(f['index/select_bin2'])
        mask_1p = np.array(f['index/select_1p_bin2'])
        mask_1m = np.array(f['index/select_1m_bin2'])
        mask_2p = np.array(f['index/select_2p_bin2'])
        mask_2m = np.array(f['index/select_2m_bin2'])
    elif select_src == "bin3":
        sel_src = np.array(f['index/select_bin3'])
        mask_1p = np.array(f['index/select_1p_bin3'])
        mask_1m = np.array(f['index/select_1m_bin3'])
        mask_2p = np.array(f['index/select_2p_bin3'])
        mask_2m = np.array(f['index/select_2m_bin3'])
    elif select_src == "bin4":
        sel_src = np.array(f['index/select_bin4'])
        mask_1p = np.array(f['index/select_1p_bin4'])
        mask_1m = np.array(f['index/select_1m_bin4'])
        mask_2p = np.array(f['index/select_2p_bin4'])
        mask_2m = np.array(f['index/select_2m_bin4'])

    dgamma = 2*0.01
    R11s = ( np.array(f['catalog/metacal/unsheared/e_1'])[mask_1p].mean() -
         np.array(f['catalog/metacal/unsheared/e_1'])[mask_1m].mean() )/dgamma
    R22s = ( np.array(f['catalog/metacal/unsheared/e_2'])[mask_2p].mean() -
         np.array(f['catalog/metacal/unsheared/e_2'])[mask_2m].mean() )/dgamma
    Rs = 0.5*(R11s+R22s)

    ra = np.array(f['catalog/gold/ra'])[sel_src]
    dec = np.array(f['catalog/gold/dec'])[sel_src]
    R11 = np.array(f['catalog/metacal/unsheared/R11'])[sel_src]
    R22 = np.array(f['catalog/metacal/unsheared/R22'])[sel_src]
    R12 = np.array(f['catalog/metacal/unsheared/R12'])[sel_src]
    R21 = np.array(f['catalog/metacal/unsheared/R21'])[sel_src]
    e1 = np.array(f['catalog/metacal/unsheared/e_1'])[sel_src]
    e2 = np.array(f['catalog/metacal/unsheared/e_2'])[sel_src]
    z_bpz = np.array(f['catalog/bpz/unsheared/zmean_sof'])[sel_src]
    z_dnf = np.array(f['catalog/dnf/unsheared/zmean_sof'])[sel_src]
    zmc_dnf = np.array(f['catalog/dnf/unsheared/zmc_sof'])[sel_src]
    zmc_bpz = np.array(f['catalog/bpz/unsheared/zmc_sof'])[sel_src]
    w_e = np.array(f['catalog/metacal/unsheared/weight'])[sel_src]
    e1_mean = np.average(e1,weights=w_e)
    e2_mean = np.average(e2,weights=w_e)
    e1 = e1 - e1_mean
    e2 = e2 - e2_mean
    hpix = hp.ang2pix(512, ra, dec, nest=True, lonlat=True)  ### nside 512, nested ==> each heal-pixel is about 0.12 deg in diameter

    func_dist = dist_comoving()
    dist_bpz = func_dist(z_bpz)
    dist_dnf = func_dist(z_dnf)
    distmc_bpz = func_dist(zmc_bpz)
    distmc_dnf = func_dist(zmc_dnf)

    if jk_dir is not None:
        if run_jk_kmeans:
            ratem = ra.copy()
            ratem[ratem>250.] -= 360.
            radec = np.vstack((ratem,dec)).T
            nstep = int(len(ra)/4000000 + 1) ### 4,000,000 galaxies per time (memory issue ==> if it can handle more increase it)
            jk = np.zeros(len(ra))
            for i in range(nstep):
                print(i,'of',nstep,'done calculating jk_src')
                if i != nstep-1:
                    jk[i*4000000:(i+1)*4000000] = km.find_nearest(radec[i*4000000:(i+1)*4000000])
                else:
                    jk[i*4000000:] = km.find_nearest(radec[i*4000000:])
            np.savez(jk_dir, jk=jk)
        else:
            jk = np.load(jk_dir+'.npz')['jk']  ## if already saved in npz file format with the column name "jk"
        return ra, dec, R11, R22, R12, R21, e1, e2, z_bpz, z_dnf, zmc_bpz, zmc_dnf, dist_bpz, dist_dnf, distmc_bpz, distmc_dnf, w_e, jk, hpix, Rs
    
    else:
        return ra, dec, R11, R22, R12, R21, e1, e2, z_bpz, z_dnf, zmc_bpz, zmc_dnf, dist_bpz, dist_dnf, distmc_bpz, distmc_dnf, w_e, hpix, Rs


def run_DS(RA, DEC, Z, R, src_cat, select_src='dnf', comoving=True, cut_with_jk=False):
    if cut_with_jk:
        ra, dec, R11, R22, R12, R21, e1, e2, z_bpz, z_dnf, zmc_bpz, zmc_dnf, dist_bpz, dist_dnf, distmc_bpz, distmc_dnf, w_e, jk, hpix, Rs = src_cat
    else:
        ra, dec, R11, R22, R12, R21, e1, e2, z_bpz, z_dnf, zmc_bpz, zmc_dnf, dist_bpz, dist_dnf, distmc_bpz, distmc_dnf, w_e, hpix, Rs = src_cat

    MpcPerDeg = (h*60.*(cosmo.kpc_comoving_per_arcmin(Z))/1000.).value
    if comoving==False: ##if using physical distance
        MpcPerDeg /= (1.+Z)

    pos_lens = SkyCoord(ra=RA*u.deg,dec=DEC*u.deg)
    if cut_with_jk:
        pos_jk = SkyCoord(ra=center_data[0]*u.deg,dec=center_data[1]*u.deg)
        jk_dist = pos_lens.separation(pos_jk).degree   ### distances from the lens to each jk patch center
        ind = jk_dist < 15.+R[-1]/MpcPerDeg   ## 15 = approx. double the size of the jk patch length + extra for make sure we don't miss anything
        jk_neigh = np.arange(100)[ind]  ## 100 = number of jk patches
        ind_jk_src = np.in1d(jk,jk_neigh)

        ra = ra[ind_jk_src]
        dec = dec[ind_jk_src]
        R11 = R11[ind_jk_src]
        R21 = R21[ind_jk_src]
        R12 = R12[ind_jk_src]
        R22 = R22[ind_jk_src]
        e1 = e1[ind_jk_src]
        e2 = e2[ind_jk_src]
        z_bpz = z_bpz[ind_jk_src]
        z_dnf = z_dnf[ind_jk_src]
        zmc_dnf = zmc_dnf[ind_jk_src]
        zmc_bpz = zmc_bpz[ind_jk_src]
        dist_bpz = dist_bpz[ind_jk_src]
        dist_dnf = dist_dnf[ind_jk_src]
        distmc_bpz = distmc_bpz[ind_jk_src]
        distmc_dnf = distmc_dnf[ind_jk_src]
        w_e = w_e[ind_jk_src]
        hpix = hpix[ind_jk_src]

    vec_lens = hp.ang2vec(RA,DEC,lonlat=True)
    hpix_select = hp.query_disc(nside=512, vec=vec_lens, radius=np.radians(R[-1]/MpcPerDeg+0.5), nest=True)  #+0.5 to make sure we don't miss any galaxies due to the finite hpix size
    ind_hpix = np.isin(hpix,hpix_select)

    ra = ra[ind_hpix]
    dec = dec[ind_hpix]
    R11 = R11[ind_hpix]
    R21 = R21[ind_hpix]
    R12 = R12[ind_hpix]
    R22 = R22[ind_hpix]
    e1 = e1[ind_hpix]
    e2 = e2[ind_hpix]
    z_bpz = z_bpz[ind_hpix]
    z_dnf = z_dnf[ind_hpix]
    zmc_dnf = zmc_dnf[ind_hpix]
    zmc_bpz = zmc_bpz[ind_hpix]
    dist_bpz = dist_bpz[ind_hpix]
    dist_dnf = dist_dnf[ind_hpix]
    distmc_bpz = distmc_bpz[ind_hpix]
    distmc_dnf = distmc_dnf[ind_hpix]
    w_e = w_e[ind_hpix]
    pos_src = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
    deg_src = pos_lens.separation(pos_src).degree

    if select_src == 'dnf':
        ind_behind = z_dnf > Z + 0.1
        R11 = R11[ind_behind]
        R12 = R12[ind_behind]
        R21 = R21[ind_behind]
        R22 = R22_tem[ind_behind]
        e1 = e1[ind_behind]
        e2 = e2[ind_behind]
        z_bpz = z_bpz[ind_behind]
        z_dnf = z_dnf[ind_behind]
        zmc_dnf = zmc_dnf[ind_behind]
        zmc_bpz = zmc_bpz[ind_behind]
        dist_bpz = dist_bpz[ind_behind]
        dist_dnf = dist_dnf[ind_behind]
        distmc_bpz = distmc_bpz[ind_behind]
        distmc_dnf = distmc_dnf[ind_behind]
        w_e = w_e[ind_behind]
        deg_src = deg_src[ind_behind]
        pos_src = pos_src[ind_behind]
    
    pz_bin_bpz = np.linspace(-0.005,3.005,302)
    pz_bin_dnf = pz_bin_bpz
    top = np.zeros(NBINS)
    top_im = np.zeros(NBINS)
    bottom = np.zeros(NBINS)
    weight = np.zeros(NBINS)
    pz_bpz = np.zeros((NBINS,301))
    pz_dnf = np.zeros((NBINS,301))

    Dist = (cosmo.comoving_distance(Z).value*10**6)*h
    if comoving==False:
        Dist /= (1.+Z)
        dist_bpz /= (1.+z_bpz)
        dist_dnf /= (1.+z_dnf)
        distmc_bpz /= (1.+zmc_bpz)
        distmc_dnf /= (1.+zmc_dnf)

    ang_rad = np.pi/2. + pos_lens.position_angle(pos_src).radian
    cos2phi = np.cos(2.*ang_rad)
    sin2phi = np.sin(2.*ang_rad)
    et = -e1*cos2phi - e2*sin2phi
    ex = e1*sin2phi - e2*cos2phi
    R_rot = R11*cos2phi**2 + R22*sin2phi**2 +(R12+R21)*sin2phi*cos2phi + Rs
    wt = w_e*(1./(1.663*10**12))*(1.+Z)*Dist*(1.-Dist/dist_dnf)
    sci = (1./(1.663*10**12))*(1.+Z)*Dist*(1.-Dist/distmc_dnf)
    if comoving==False:
        wt /= 1.+Z
        sci /= 1.+Z
    wt[wt<0.] = 0.
    sci[sci<0.] = 0.
    top_src = wt*et
    top_im_src = wt*ex
    bottom_src = wt*sci*R_rot
    for i in range(len(R)-1):
        thmin = np.arctan(R[i]*10**6 / Dist) * (180./np.pi)
        thmax = np.arctan(R[i+1]*10**6 / Dist) * (180./np.pi)
        ind_th = (deg_src>=thmin)*(deg_src<thmax)
        top[i] = np.sum(top_src[ind_th])
        top_im[i] = np.sum(top_im_src[ind_th])
        bottom[i] = np.sum(bottom_src[ind_th])
        weight[i] = np.sum(wt[ind_th])
        hist_bpz, _ = np.histogram(zmc_bpz[ind_th],bins=pz_bin_bpz,weights=bottom_src[ind_th])
        pz_bpz[i] = hist_bpz
        hist_dnf, _ = np.histogram(zmc_dnf[ind_th],bins=pz_bin_dnf,weights=bottom_src[ind_th])
        pz_dnf[i] = hist_dnf
    return top, top_im, bottom, weight, pz_bpz, pz_dnf

if __name__=="__main__":

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
    dat_save_dir = '/global/cscratch1/sd/taeshin/'
    dict_lens_cut = {"Z_LAMBDA":[zmin,zmax], "LAMBDA_CHISQ":[lmin,lmax]}   ## dictionary for the lenses cut
    dict_rand_cut = {"ZTRUE":[zmin,zmax],"LAMBDA_IN":[lmin,lmax]}  ## dictionary for the randoms cut
    
    R, Rmid = R_bin(Rmin,Rmax,NBINS)

    t1 = t.time()

    len_cat = cut_catalog(len_dir, dict_lens_cut)
    N_lens = len(len_cat)
    N_rand = 20*N_lens
    ran_cat = cut_catalog(ran_dir, dict_rand_cut, random=True, N_randoms=N_rand)

    t2 = t.time()
    print('reading lenses and randoms took %.2f seconds'%(t2-t1))

    RA_len = len_cat["RA"]
    DEC_len = len_cat["DEC"]
    Z_len = len_cat["Z"]
    RA_ran = ran_cat["RA"]
    DEC_ran = ran_cat["DEC"]
    Z_ran = ran_cat["ZTRUE"]

    t1 = t.time()

    src_cat = cut_source(src_dir, select_src, run_jk_kmeans=False, jk_dir=dat_save_dir+"jk_src_%s"%(select_src), run_healpy=True)
    
    t2 = t.time()
    print('reading and preparing source catalog took %.2f seconds'%(t2-t1))
    
    t1 = t.time()
    
    top, top_im, bottom, weight, pz_bpz, pz_dnf = run_DS(RA_ran[run_id], DEC_ran[run_id], Z_ran[run_id], R, src_cat, select_src='bin1', comoving=True, cut_with_jk=True)
    
    t2 = t.time()
    print('DS=',top/bottom,'time taken for calculating DSigma = %.2f seconds'%(t2-t1))
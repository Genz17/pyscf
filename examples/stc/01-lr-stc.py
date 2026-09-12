import sys
import numpy
import argparse

import pyscf
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.pbc.df.fft_stc import FFTDF_STC
from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc.df.aft import AFTDF
from pyscf.pbc.df.aft_stc import AFTDF_STC
from pyscf.pbc.df.rsdf import RSGDF
from pyscf.pbc.df.rsdf_stc import density_fit

numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
numpy.set_printoptions(suppress=True, precision=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ke-cutoff", type=int, default=30)
    parser.add_argument("--kmesh", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--odr", type=float, default=4.)
    args = parser.parse_args()

    atom = '''
    C    0.0000000000    0.0000000000    0.0000000000
    C    0.8917500000    0.8917500000    0.8917500000
    '''
    a = numpy.asarray([
        [1.7835000000, 1.7835000000, 0.0000000000],
        [0.0000000000, 1.7835000000, 1.7835000000],
        [1.7835000000, 0.0000000000, 1.7835000000]
    ])

    basis = f'./GTHbasis/cc-pvdz.dat'
    pseudo = 'gth-hf-rev'

    kmesh = numpy.asarray(args.kmesh) # mesh in reciprocal space
    kmesh_label = "_".join(str(k) for k in kmesh)

    print("++++++++++++")
    print("the kmesh is", kmesh)
    print("the odr is", args.odr)
    print("++++++++++++")
    cell = pyscf.pbc.gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo)
    cell.verbose = 1
    cell.ke_cutoff = args.ke_cutoff
    #cell.precision=1e-14
    cell.build()
    kpts = cell.make_kpts(kmesh, time_reversal_symmetry=False)

    kmf = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='smooth_vcut_ws')
    kmf.with_df = RSGDF(cell, kpts=kpts, exxdiv = 'smooth_vcut_ws')
    kmf.verbose = 4
    kmf.kernel()
    dm = kmf.make_rdm1()

    omega = 0.2


    kmf_fft_ewald = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='ewald')
    kmf_fft_ewald.with_df = FFTDF(cell, kpts=kpts)
    vj, vk = kmf_fft_ewald.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)

    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by FFTDF_EWALD: ", exchange_energy)

    kmf_stc = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='vcut_sph')
    kmf_stc.with_df = FFTDF(cell, kpts=kpts)
    kmf_stc.with_df.omega_dot_Rc = args.odr
    vj_stc, vk_stc = kmf_stc.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)
    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_stc) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by FFTDF_SPH: ", exchange_energy)


    kmf_fft_ws = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='vcut_ws')
    kmf_fft_ws.with_df = FFTDF(cell, kpts=kpts)
    vj_ws, vk_ws = kmf_fft_ws.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega) 

    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_ws) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by FFTDF_TC_WS: ", exchange_energy)

    kmf_stc = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='vcut_ws')
    kmf_stc.with_df = FFTDF_STC(cell, kpts=kpts)
    kmf_stc.with_df.omega_dot_Rc = args.odr
    vj_stc, vk_stc = kmf_stc.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)
    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_stc) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by FFTDF_STC_WS: ", exchange_energy)

    kmf_fft_ws = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='smooth_vcut_ws')
    kmf_fft_ws.with_df = FFTDF(cell, kpts=kpts)
    vj_ws, vk_ws = kmf_fft_ws.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)

    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_ws) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by FFTDF_TC_WS: ", exchange_energy)

    kmf_fft_ws = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='smooth_vcut_sph')
    kmf_fft_ws.with_df = FFTDF(cell, kpts=kpts)
    vj_ws, vk_ws = kmf_fft_ws.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)
    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_ws) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by FFTDF_TC_SPH: ", exchange_energy)


    kmf_aft_ws = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='vcut_ws')
    kmf_aft_ws.with_df = AFTDF_STC(cell, kpts=kpts)
    vj_ws, vk_ws = kmf_aft_ws.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)

    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_ws) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by AFTDF_STC: ", exchange_energy)

    kmf_aft_ws = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='smooth_vcut_ws')
    kmf_aft_ws.with_df = AFTDF(cell, kpts=kpts)
    vj_ws, vk_ws = kmf_aft_ws.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)
    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_ws) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by AFTDF_STC_WS: ", exchange_energy)

    kmf_aft_ws = pyscf.pbc.scf.KRHF(cell, kpts=kpts, exxdiv='smooth_vcut_sph')
    kmf_aft_ws.with_df = AFTDF(cell, kpts=kpts)
    vj_ws, vk_ws = kmf_aft_ws.get_jk(dm_kpts=dm, with_j = False, with_k = True, omega = omega)
    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk_ws) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by AFTDF_STC_SPH: ", exchange_energy)


    kmf_rsdf_ws = scf.KRHF(cell, kpts=kpts, exxdiv='smooth_vcut_ws')
    kmf_rsdf_ws.with_df = RSGDF(cell, kpts=kpts, exxdiv = 'smooth_vcut_ws')
    _, vk_ws = kmf_rsdf_ws.get_jk(dm_kpts=dm, with_j=False, with_k=True, omega=omega)
    exchange_energy = (-0.25 * numpy.einsum('kij,kji->', dm, vk_ws) / len(kpts))
    print("Exchange Energy by RSDF_STC_WS:", exchange_energy)

    kmf_rsdf_ws = scf.KRHF(cell, kpts=kpts, exxdiv='smooth_vcut_sph')
    kmf_rsdf_ws.with_df = RSGDF(cell, kpts=kpts, exxdiv = 'smooth_vcut_sph')
    _, vk_ws = kmf_rsdf_ws.get_jk(dm_kpts=dm, with_j=False, with_k=True, omega=omega)
    exchange_energy = (-0.25 * numpy.einsum('kij,kji->', dm, vk_ws) / len(kpts))
    print("Exchange Energy by RSDF_STC_SPH:", exchange_energy)


    kmf_rsdf_ws = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    kmf_rsdf_ws = density_fit(kmf_rsdf_ws, exxdiv='vcut_ws', omega_dot_Rc=args.odr)
    _, vk_ws = kmf_rsdf_ws.get_jk(dm_kpts=dm, with_j=False, with_k=True, omega=omega)
    exchange_energy = (-0.25 * numpy.einsum('kij,kji->', dm, vk_ws) / len(kpts))
    print("Exchange Energy by RSDF_STC:", exchange_energy)


    #print(numpy.linalg.norm(vk - vk_stc)/numpy.linalg.norm(vk))
    #print(numpy.linalg.norm(vk_ws - vk_stc)/numpy.linalg.norm(vk_ws))

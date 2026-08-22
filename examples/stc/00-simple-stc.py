import sys
import numpy
import pyscf
from pyscf import lib
from pyscf.pbc import gto, scf, df
from pyscf.pbc.tools.k2gamma import translation_vectors_for_kmesh
from pyscf.gto.moleintor import getints4c, getints
import copy
from pyscf.pbc.df.rsdf_helper import _binary_search
import time
import os 
import ctypes
import argparse

numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
numpy.set_printoptions(suppress=True, precision=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ke-cutoff", type=int, default=30)
    parser.add_argument("--kmesh", type=int, nargs=3, default=[1, 1, 1])
    args = parser.parse_args()

    #atom = '''
    #H 0 0 0
    #H 0.74 0 0
    #'''
    #a = numpy.asarray([
    #    [8,0,0],
    #    [0,8,0],
    #    [0,0,8]],dtype=numpy.float64)

    #atom = '''
    #B    0.0000000000    0.0000000000    0.0000000000
    #N    1.2504457494    0.7217977157    0.0000000000
    #'''
    #a = numpy.asarray([
    #       [2.5008385811,    0.0000000000,    0.0000000000],
    #       [1.2504192905,    2.1657897420,    0.0000000000],
    #       [0.0000000000,    0.0000000000,    20]
    #])

    atom = '''
    C    0.0000000000    0.0000000000    0.0000000000
    C    0.8917500000    0.8917500000    0.8917500000
    '''
    a = numpy.asarray([
        [1.7835000000, 1.7835000000, 0.0000000000],
        [0.0000000000, 1.7835000000, 1.7835000000],
        [1.7835000000, 0.0000000000, 1.7835000000]
    ])

    #atom = '''
    #Si 0 0 0
    #Si 1.9200424 1.10853699 0.78385403
    #'''
    #a = numpy.asarray([
    #    [3.84008327, 0, 0],
    #    [1.92004163, 3.32560966, 0],
    #    [1.92004163, 1.10853655, 3.13541486]
    #])

    #atom = '''
    #B 0.0000000000   -1.4438092827    1.5876967120
    #B 0.0000000000    1.4438092827   -1.5876967120
    #N 0.0000000000   -1.4438092827   -1.5876967120
    #N 0.0000000000    1.4438092827    1.5876967120
    #'''
    #a = numpy.asarray([
    #    [1.2503755171, -2.1657139241, -0.0000000000],
    #    [1.2503755171,  2.1657139241, 0.0000000000],
    #    [0.0000000000, -0.0000000000, 6.3507868482]
    #])


    basis = f'./GTHbasis/minao_gth.dat'
    pseudo = 'gth-pbe'

    kmesh = numpy.asarray(args.kmesh) # mesh in reciprocal space
    kmesh_label = "_".join(str(k) for k in kmesh)

    print("++++++++++++")
    print("the kmesh is", kmesh)
    print("++++++++++++")
    cell = pyscf.pbc.gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo)
    cell.verbose = 1
    cell.ke_cutoff = args.ke_cutoff
    #cell.precision=1e-14
    cell.build()
    kpts = cell.make_kpts(kmesh, time_reversal_symmetry=False)
    kmf = pyscf.pbc.scf.KRHF(cell,kpts=kpts, exxdiv='stc_ws_3')
    kmf.with_df = df.FFTDF(cell, kpts=kpts)
    chkfileName = './diamond_chk/diamond_%d_%s.chk' % (cell.ke_cutoff, kmesh_label)
    try:
        assert 1 == 0
        cell, scf_res = scf.chkfile.load_scf(chkfileName)
        cell.verbose = 0
        for key,v in scf_res.items():
            setattr(kmf, key, v)
            kmf.kpts = kpts
        kmf.converged = True
    except BaseException:
        print("========================")
        print("sth is wrong, build from scratch")
        print("========================")
        kmf.chkfile = chkfileName
        kmf.kernel()
    dm = kmf.make_rdm1()
    vj, vk = kmf.get_jk(dm_kpts=dm)
    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by DM K original: ", exchange_energy)


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

    atom = '''
    C    0.0000000000    0.0000000000    0.0000000000
    C    0.8917500000    0.8917500000    0.8917500000
    '''
    a = numpy.asarray([
        [1.7835000000, 1.7835000000, 0.0000000000],
        [0.0000000000, 1.7835000000, 1.7835000000],
        [1.7835000000, 0.0000000000, 1.7835000000]
    ])


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
    kmf.kernel()
    dm = kmf.make_rdm1()
    vj, vk = kmf.get_jk(dm_kpts=dm)
    exchange_energy = - 0.25 * numpy.einsum('kij,kji -> ', dm, vk) / numpy.prod(numpy.array(kmesh))
    print("Exchange Energy by DM K original: ", exchange_energy)


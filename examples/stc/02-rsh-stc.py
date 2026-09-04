import sys
import numpy
import argparse

import pyscf
from pyscf import lib
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.df.fft_stc import FFTDF_STC
from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc.df.rsdf import RSGDF

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

    basis = f'./GTHbasis/minao_gth.dat'
    pseudo = 'gth-hf-rev'

    kmesh = numpy.asarray(args.kmesh) # mesh in reciprocal space
    kmesh_label = "_".join(str(k) for k in kmesh)

    print("++++++++++++")
    print("the kmesh is", kmesh)
    print("the odr is", args.odr)
    print("++++++++++++")
    cell = pyscf.pbc.gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo)
    cell.verbose = 0
    cell.ke_cutoff = args.ke_cutoff
    #cell.precision=1e-14
    cell.build()
    kpts = cell.make_kpts(kmesh, time_reversal_symmetry=False)

    kmf_stc = dft.KRKS(
        cell,
        kpts=kpts,
        xc="CAM-B3LYP",
        exxdiv="vcut_ws",
    )

    kmf_stc.with_df = FFTDF_STC(cell, kpts=kpts)
    kmf_stc.with_df.Rc_type = "ws"
    kmf_stc.with_df.omega_dot_Rc = args.odr
    energy = kmf_stc.kernel()

    print("STC Converged:", kmf_stc.converged)
    print("STC Total energy:", energy)


    kmf_ws = dft.KRKS(
        cell,
        kpts=kpts,
        xc="CAM-B3LYP",
        exxdiv="vcut_ws",
    )

    kmf_ws.with_df = FFTDF(cell, kpts=kpts)
    energy = kmf_ws.kernel()

    print("WS Converged:", kmf_ws.converged)
    print("WS Total energy:", energy)


    kmf_ewald = dft.KRKS(
        cell,
        kpts=kpts,
        xc="CAM-B3LYP",
        exxdiv="ewald",
    )

    kmf_ewald.with_df = FFTDF(cell, kpts=kpts)
    energy = kmf_ewald.kernel()

    print("EWALD Converged:", kmf_ewald.converged)
    print("EWALD Total energy:", energy)




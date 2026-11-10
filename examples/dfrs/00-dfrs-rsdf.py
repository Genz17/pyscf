#!/usr/bin/env python

import argparse
from time import perf_counter

import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.df import dfrs, rsdf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--basis", default="cc-pvdz", help="Orbital basis (default: %(default)s)"
)
parser.add_argument(
    "--kmesh", nargs=3, type=int, default=[4, 4, 4], metavar=("NX", "NY", "NZ"),
    help="K-point mesh (default: %(default)s)",
)
parser.add_argument(
    "--max-memory", type=float, default=gto.Cell.max_memory, metavar="MB",
    help="Cell memory budget in MB (default: %(default)s)",
)
args = parser.parse_args()

box_length = 10.0  # Angstrom.
center = box_length / 2
kmesh = args.kmesh

# Water molecule with the oxygen at the center of a cubic periodic box.
cell = gto.M(
    a=box_length * np.eye(3),
    atom=[
        ("O", (center, center, center)),
        ("H", (center, center - 0.757, center + 0.587)),
        ("H", (center, center + 0.757, center + 0.587)),
    ],
    unit="Angstrom",
    basis=args.basis,
    pseudo=None,
    max_memory=args.max_memory,
    precision=1e-8,
    verbose=4,
)
kpts = cell.make_kpts(kmesh)

energies = {}
timings = {}
for name, mydf in (
    ("DFRS", dfrs.GDF(cell, kmesh=kmesh)),
    ("RSDF", rsdf.RSDF(cell, kpts)),
):
    print(f"\n{name} KRHF")
    mydf.auxbasis = cell.basis + "-jkfit"
    mf = scf.KRHF(cell, kpts).density_fit(with_df=mydf)
    mf.conv_tol = 1e-9
    start = perf_counter()  # Includes the density-fitting build and SCF.
    energies[name] = mf.kernel()
    timings[name] = perf_counter() - start

print(f"\nDFRS energy: {energies['DFRS']:.12f} Ha")
print(f"RSDF energy: {energies['RSDF']:.12f} Ha")
print(f"DFRS - RSDF: {energies['DFRS'] - energies['RSDF']:.3e} Ha")
print(f"DFRS time (DF + SCF): {timings['DFRS']:.2f} s")
print(f"RSDF time (DF + SCF): {timings['RSDF']:.2f} s")

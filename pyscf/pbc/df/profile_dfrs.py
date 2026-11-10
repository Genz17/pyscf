#!/usr/bin/env python3
import argparse
import resource
import sys
import threading
import time

import numpy as np
import psutil
from pyscf import lib
from pyscf.pbc import gto

import dfrs  # Your local dfrs.py


MIB = 1024**2


def lifetime_peak_rss():
    """OS high-water RSS in bytes; supports macOS and Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def profile_build(mydf, interval):
    process = psutil.Process()
    stop = threading.Event()
    rss_before = process.memory_info().rss
    peak_rss = rss_before
    hwm_before = lifetime_peak_rss()

    def sample():
        nonlocal peak_rss
        while not stop.wait(interval):
            peak_rss = max(peak_rss, process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    try:
        mydf.build()  # Includes auxiliary-cell setup and pair screening.
    finally:
        elapsed = time.perf_counter() - start
        rss_after = process.memory_info().rss
        hwm_after = lifetime_peak_rss()
        stop.set()
        monitor.join()
        peak_rss = max(peak_rss, rss_after)

        print("\nMemory profile")
        print(f"Build wall time:               {elapsed:12.2f} s")
        for label, value in [
            ("RSS before build", rss_before),
            ("Sampled peak RSS", peak_rss),
            ("Peak above baseline", peak_rss - rss_before),
            ("RSS after build", rss_after),
            ("RSS increase after build", rss_after - rss_before),
            ("OS lifetime peak before", hwm_before),
            ("OS lifetime peak after", hwm_after),
        ]:
            print(f"{label + ':':30s} {value / MIB:12.2f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmesh", nargs=3, type=int, required=True,
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--basis", required=True)
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.01,
                        help="RSS sampling interval in seconds")
    parser.add_argument("--precision", type=float, default=1e-8)
    parser.add_argument("--pair-threshold", type=float, default=None,
                        help="Default: cell.precision; 0 retains all pairs")
    args = parser.parse_args()

    if min(args.kmesh) < 1 or args.threads < 1:
        parser.error("kmesh entries and threads must be positive")
    if not np.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be finite and positive")
    if not np.isfinite(args.precision) or args.precision <= 0:
        parser.error("precision must be finite and positive")
    if args.pair_threshold is not None:
        if not np.isfinite(args.pair_threshold) or args.pair_threshold < 0:
            parser.error("pair-threshold must be finite and nonnegative")

    lib.num_threads(args.threads)

    # Primitive diamond cell: FCC lattice with two carbon atoms.
    a = 3.5668  # Angstrom; conventional cubic lattice constant
    cell = gto.M(
        a=0.5 * a * np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]),
        atom=[
            ("C", (0.0, 0.0, 0.0)),
            ("C", (a / 4, a / 4, a / 4)),
        ],
        unit="Angstrom",
        basis=args.basis,
        pseudo="gth-pade",
        precision=args.precision,
        verbose=4,
    )

    mydf = dfrs.GDF(cell, kmesh=args.kmesh)
    if args.auxbasis is not None:
        mydf.auxbasis = args.auxbasis
    if args.pair_threshold is not None:
        mydf.pair_threshold = args.pair_threshold

    print(f"\nUsing: {dfrs.__file__}", flush=True)
    print(f"kmesh={args.kmesh}, basis={args.basis}, "
          f"OpenMP threads={lib.num_threads()}", flush=True)

    profile_build(mydf, args.interval)

    # Returns the existing tensor without copying or rebuilding.
    cderi = mydf.cderi_array()
    nk, naux, npair = cderi.shape

    print("\nCDERI")
    print(f"Primitive-cell AOs:            {cell.nao_nr()}")
    print(f"Primitive-cell auxiliary AOs:  {naux}")
    print(f"Retained AO pairs:             "
          f"{npair} / {nk * cell.nao_nr()**2}")
    print(f"Shape (translations, aux, pair): {cderi.shape}")
    print(f"dtype:                        {cderi.dtype}")
    print(f"Final tensor storage:         {cderi.nbytes / MIB:.2f} MiB")
    print(f"Pair-index storage:           {mydf.pairs.nbytes / MIB:.2f} MiB")

    # _make_cderi initially allocates complex128 factors.
    factor_bytes = 16 * nk * naux * npair
    print(f"Complex working tensor alone: {factor_bytes / MIB:.2f} MiB")

    return mydf, cderi


if __name__ == "__main__":
    mydf, cderi = main()


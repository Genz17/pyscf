#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

import numpy

from pyscf import lib
from pyscf.df import df_jk
from pyscf.lib import logger


def get_j(mypari, dm, hermi=1, direct_scf_tol=1e-13, omega=None):
    if omega is not None:
        raise NotImplementedError('range-separated Coulomb is not supported')

    mypari._build_auxmol()
    mydf = mypari._j_df
    mydf.max_memory = mypari.max_memory
    mydf.stdout = mypari.stdout
    mydf.verbose = mypari.verbose
    vj = df_jk.get_j(mydf, dm, hermi, direct_scf_tol)
    return vj


def get_k(mypari, dm, hermi=1, mo_coeff=None, mo_occ=None, omega=None):
    # This version does not exploit NBX sparsity. Each packed AO atom-pair
    # block is evaluated, metric-corrected, and scattered to dense G in C.
    if omega is not None:
        raise NotImplementedError('range-separated exchange is not supported')
    if hermi != 1:
        raise NotImplementedError('PARI K only supports hermi=1')

    dm_tag = dm
    dm = numpy.asarray(dm)
    nao = mypari.mol.nao_nr()
    if dm.shape != (nao, nao):
        raise NotImplementedError('PARI K only supports one density matrix')
    if numpy.iscomplexobj(dm):
        raise NotImplementedError('complex density matrices are not supported')

    if mo_coeff is None:
        mo_coeff = getattr(dm_tag, 'mo_coeff', None)
        if mo_occ is None:
            mo_occ = getattr(dm_tag, 'mo_occ', None)
    mo_coeff = _factor_dm(dm, mo_coeff, mo_occ)
    if numpy.iscomplexobj(mo_coeff):
        raise NotImplementedError('complex orbitals are not supported')

    if mypari.df_coeff is None:
        mypari.build()

    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mypari)
    mol = mypari.mol
    auxmol = mypari.auxmol
    layout = mypari.aopair_layout
    auxslice = auxmol.aoslice_by_atom()
    nocc = mo_coeff.shape[1]
    naux = auxmol.nao_nr()

    tnames = ('Dmat', 'Jmat', 'Gmat', 'Hmat', 'Lmat')
    tspans = numpy.zeros((5,2))
    dtype = numpy.result_type(mo_coeff, numpy.double)
    max_naux = numpy.max(auxslice[:,3] - auxslice[:,2])
    mo_coeff = numpy.asarray(mo_coeff, dtype=dtype, order='C')
    moT = numpy.asarray(mo_coeff.T, order='C')
    Dbuf = numpy.empty(nocc*max_naux*nao, dtype=dtype)
    Hbuf = numpy.empty(nocc*max_naux*nao, dtype=dtype)
    Gbuf = numpy.empty(nao*max_naux*nao, dtype=dtype)
    j2cbuf = numpy.empty(naux*max_naux, dtype=dtype)
    Lmat = numpy.zeros((nao, nao), dtype=dtype)
    for A in range(mol.natm):
        naux_A = auxslice[A,3] - auxslice[A,2]

        tick = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        out = Dbuf[:nocc*naux_A*nao]
        Dmat = mypari.build_d(mo_coeff, A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[0] += tock - tick

        tick = tock
        out = j2cbuf[:naux*naux_A]
        j2c = mypari.fill_j2c(A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[1] += tock - tick

        tick = tock
        out = Gbuf[:nao*naux_A*nao]
        Gmat = mypari.fill_g(A, j2c, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[2] += tock - tick

        tick = tock
        Hmat = Hbuf[:nocc*naux_A*nao].reshape(nocc,naux_A,nao)
        lib.dot(moT, Gmat.reshape(nao,naux_A*nao),
                c=Hmat.reshape(nocc,naux_A*nao))
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[3] += tock - tick

        tick = tock
        lib.dot(Dmat.reshape(nocc*naux_A,nao).T,
                Hmat.reshape(nocc*naux_A,nao), c=Lmat, beta=1)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[4] += tock - tick

    vk = Lmat + Lmat.T
    for name, tspan in zip(tnames, tspans):
        cpu0 = logger.process_clock() - tspan[0]
        wall0 = logger.perf_counter() - tspan[1]
        log.timer('PARI K ' + name, cpu0, wall0)

    pair_factor = 2 - (layout.pair_atoms[:,0] ==
                       layout.pair_atoms[:,1])
    log.debug('PARI K calls: D %d sparse dgemm; '
              'G %d fused pair jobs/%d metric products; '
              'H %d dgemm; L %d dgemm',
              len(mypari.df_coeff.d_source_shell),
              mol.natm*layout.npair, mol.natm*pair_factor.sum(),
              mol.natm, mol.natm)
    log.timer('PARI K', *t0)
    return vk


def get_k_version1(mypari, dm, hermi=1, mo_coeff=None, mo_occ=None,
                   omega=None):
    # Version 1 does not exploit NBX sparsity. Three-center integrals
    # are stored in packed AO-pair form and unpacked into a dense G panel.
    if omega is not None:
        raise NotImplementedError('range-separated exchange is not supported')
    if hermi != 1:
        raise NotImplementedError('PARI K only supports hermi=1')

    dm_tag = dm
    dm = numpy.asarray(dm)
    nao = mypari.mol.nao_nr()
    if dm.shape != (nao, nao):
        raise NotImplementedError('PARI K only supports one density matrix')
    if numpy.iscomplexobj(dm):
        raise NotImplementedError('complex density matrices are not supported')

    if mo_coeff is None:
        mo_coeff = getattr(dm_tag, 'mo_coeff', None)
        if mo_occ is None:
            mo_occ = getattr(dm_tag, 'mo_occ', None)
    mo_coeff = _factor_dm(dm, mo_coeff, mo_occ)
    if numpy.iscomplexobj(mo_coeff):
        raise NotImplementedError('complex orbitals are not supported')

    if mypari.df_coeff is None:
        mypari.build()

    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mypari)
    mol = mypari.mol
    auxmol = mypari.auxmol
    layout = mypari.aopair_layout
    auxslice = auxmol.aoslice_by_atom()
    nocc = mo_coeff.shape[1]
    naux = auxmol.nao_nr()

    tnames = ('Dmat', 'Jmat', 'Gmat', 'Emat',
              'Gunpack', 'Hmat', 'Lmat')
    tspans = numpy.zeros((7,2))
    dtype = numpy.result_type(mo_coeff, numpy.double)
    max_naux = numpy.max(auxslice[:,3] - auxslice[:,2])
    mo_coeff = numpy.asarray(mo_coeff, dtype=dtype, order='C')
    moT = numpy.asarray(mo_coeff.T, order='C')
    Dbuf = numpy.empty(nocc*max_naux*nao, dtype=dtype)
    Hbuf = numpy.empty(nocc*max_naux*nao, dtype=dtype)
    Gbuf = numpy.empty(nao*max_naux*nao, dtype=dtype)
    j3cbuf = numpy.empty(layout.naopair*max_naux, dtype=dtype)
    j2cbuf = numpy.empty(naux*max_naux, dtype=dtype)
    Lmat = numpy.zeros((nao, nao), dtype=dtype)
    for A in range(mol.natm):
        naux_A = auxslice[A,3] - auxslice[A,2]

        tick = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        out = Dbuf[:nocc*naux_A*nao]
        Dmat = mypari.build_d(mo_coeff, A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[0] += tock - tick

        tick = tock
        out = j2cbuf[:naux*naux_A]
        j2c = mypari.fill_j2c(A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[1] += tock - tick

        tick = tock
        out = j3cbuf[:layout.naopair*naux_A]
        j3c = mypari.fill_aux_e2_sparse(A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[2] += tock - tick

        tick = tock
        mypari.metric_contract(j3c, j2c)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[3] += tock - tick

        tick = tock
        out = Gbuf[:nao*naux_A*nao]
        Gmat = mypari.unpack_aopair(j3c, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[4] += tock - tick

        tick = tock
        Hmat = Hbuf[:nocc*naux_A*nao].reshape(nocc,naux_A,nao)
        lib.dot(moT, Gmat.reshape(nao,naux_A*nao),
                c=Hmat.reshape(nocc,naux_A*nao))
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[5] += tock - tick

        tick = tock
        lib.dot(Dmat.reshape(nocc*naux_A,nao).T,
                Hmat.reshape(nocc*naux_A,nao), c=Lmat, beta=1)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[6] += tock - tick

    vk = Lmat + Lmat.T
    for name, tspan in zip(tnames, tspans):
        cpu0 = logger.process_clock() - tspan[0]
        wall0 = logger.perf_counter() - tspan[1]
        log.timer('PARI K ' + name, cpu0, wall0)

    pair_factor = 2 - (layout.pair_atoms[:,0] ==
                       layout.pair_atoms[:,1])
    log.debug('PARI K calls: D %d sparse dgemm; '
              'E %d metric products; G %d C unpack; '
              'H %d dgemm; L %d dgemm',
              len(mypari.df_coeff.d_source_shell),
              mol.natm*pair_factor.sum(), mol.natm,
              mol.natm, mol.natm)
    log.timer('PARI K', *t0)
    return vk


def get_k_slow(mypari, dm, hermi=1, mo_coeff=None, mo_occ=None, omega=None):
    if omega is not None:
        raise NotImplementedError('range-separated exchange is not supported')
    if hermi != 1:
        raise NotImplementedError('PARI K only supports hermi=1')

    dm_tag = dm
    dm = numpy.asarray(dm)
    nao = mypari.mol.nao_nr()
    if dm.shape != (nao, nao):
        raise NotImplementedError('PARI K only supports one density matrix')
    if numpy.iscomplexobj(dm):
        raise NotImplementedError('complex density matrices are not supported')

    if mo_coeff is None:
        mo_coeff = getattr(dm_tag, 'mo_coeff', None)
        if mo_occ is None:
            mo_occ = getattr(dm_tag, 'mo_occ', None)
    mo_coeff = _factor_dm(dm, mo_coeff, mo_occ)
    if numpy.iscomplexobj(mo_coeff):
        raise NotImplementedError('complex orbitals are not supported')

    if mypari.df_coeff is None:
        mypari.build()

    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mypari)
    mol = mypari.mol
    auxmol = mypari.auxmol
    layout = mypari.aopair_layout
    df_coeff = mypari.df_coeff
    auxslice = auxmol.aoslice_by_atom()
    nocc = mo_coeff.shape[1]
    naux = auxmol.nao_nr()

    tnames = ('Dmat', 'Jmat', 'Gmat', 'Emat',
              'Gunpack', 'Hmat', 'Lmat')
    tspans = numpy.zeros((7,2))

    dtype = numpy.result_type(mo_coeff, df_coeff._data.dtype)
    max_naux = numpy.max(auxslice[:,3] - auxslice[:,2])
    moT = numpy.asarray(mo_coeff.T, dtype=dtype, order='C')
    # Dense work arrays use (AO,aux,AO) and (occ,aux,AO).
    Dbuf = numpy.empty(nocc*max_naux*nao, dtype=dtype)
    Hbuf = numpy.empty(nocc*max_naux*nao, dtype=dtype)
    Gbuf = numpy.empty(nao*max_naux*nao, dtype=dtype)
    j2cbuf = numpy.empty(naux*max_naux, dtype=dtype)
    j3cbuf = numpy.empty(layout.naopair*max_naux, dtype=dtype)
    Lmat = numpy.zeros((nao, nao), dtype=dtype)
    for A in range(mol.natm):
        aux0, aux1 = auxslice[A,2:]
        naux_A = aux1 - aux0
        tick = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        Dmat = Dbuf[:nocc*naux_A*nao].reshape(nocc,naux_A,nao)
        Gmat = Gbuf[:nao*naux_A*nao].reshape(nao,naux_A,nao)
        Gmat.fill(0)
        for pair, (B, C) in enumerate(layout.pair_atoms):
            if A == B:
                _unpack_aopair(
                    Gmat, df_coeff.left(B, C), layout, pair)
            elif A == C:
                _unpack_aopair(
                    Gmat, df_coeff.right(B, C), layout, pair)
        lib.dot(moT, Gmat.reshape(nao,naux_A*nao),
                c=Dmat.reshape(nocc,naux_A*nao))
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[0] += tock - tick    # Dmat

        tick = tock
        out = j3cbuf[:layout.naopair*naux_A]
        j3c = mypari.fill_aux_e2_sparse(A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[2] += tock - tick    # Gmat

        tick = tock
        out = j2cbuf[:naux*naux_A]
        j2c = mypari.fill_j2c(A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[1] += tock - tick    # Gmat

        tick = tock
        for pair, (B, C) in enumerate(layout.pair_atoms):
            pair0, pair1 = layout.pair_aopair_loc[pair:pair+2]
            pair_slice = slice(pair0, pair1)

            Gpair = j3c[pair_slice]
            auxB = slice(*auxslice[B,2:])
            Gpair -= .5 * lib.dot(df_coeff.left(B, C), j2c[auxB])
            if B != C:
                auxC = slice(*auxslice[C,2:])
                Gpair -= .5 * lib.dot(df_coeff.right(B, C), j2c[auxC])
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[3] += tock - tick    # Emat

        tick = tock
        Gmat.fill(0)
        for pair in range(layout.npair):
            pair0, pair1 = layout.pair_aopair_loc[pair:pair+2]
            _unpack_aopair(
                Gmat, j3c[pair0:pair1], layout, pair)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[4] += tock - tick    # Gunpack

        tick = tock
        Hmat = Hbuf[:nocc*naux_A*nao].reshape(nocc,naux_A,nao)
        lib.dot(moT, Gmat.reshape(nao,naux_A*nao),
                c=Hmat.reshape(nocc,naux_A*nao))
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[5] += tock - tick    # Hmat

        tick = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        lib.dot(Dmat.reshape(nocc*naux_A,nao).T,
                Hmat.reshape(nocc*naux_A,nao), c=Lmat, beta=1)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[6] += tock - tick    # Lmat

    vk = Lmat + Lmat.T
    for name, tspan in zip(tnames, tspans):
        cpu0 = logger.process_clock() - tspan[0]
        wall0 = logger.perf_counter() - tspan[1]
        log.timer('PARI K ' + name, cpu0, wall0)

    pair_nshlpr = numpy.diff(layout.shlpr_loc)
    pair_factor = 2 - (layout.pair_atoms[:,0] ==
                       layout.pair_atoms[:,1])
    log.debug('PARI K calls: D %d shell-pair scatter/%d dgemm; '
              'E %d metric products; G %d shell-pair unpack; '
              'H %d dgemm; L %d dgemm',
              numpy.dot(pair_factor, pair_nshlpr), mol.natm,
              mol.natm*pair_factor.sum(),
              mol.natm*layout.nshlpr, mol.natm, mol.natm)
    log.timer('PARI K', *t0)
    return vk


def get_jk(mypari, dm, hermi=1, with_j=True, with_k=True,
           direct_scf_tol=1e-13, mo_coeff=None, mo_occ=None, omega=None):
    vj = vk = None
    if with_j:
        vj = get_j(mypari, dm, hermi, direct_scf_tol, omega)
    if with_k:
        vk = get_k(mypari, dm, hermi, mo_coeff, mo_occ, omega)
    return vj, vk


def _factor_dm(dm, mo_coeff=None, mo_occ=None):
    if mo_coeff is not None and mo_occ is not None:
        mo_coeff = numpy.asarray(mo_coeff)
        mo_occ = numpy.asarray(mo_occ)
        if mo_coeff.ndim != 2 or mo_coeff.shape[0] != dm.shape[0]:
            raise ValueError('mo_coeff has incompatible shape')
        if mo_occ.shape != (mo_coeff.shape[1],):
            raise ValueError('mo_occ has incompatible shape')
        if numpy.any(mo_occ < -1e-12):
            raise ValueError('negative occupations are not supported')
        mask = mo_occ > 1e-12
        return mo_coeff[:,mask] * numpy.sqrt(mo_occ[mask])

    if mo_coeff is not None:
        mo_coeff = numpy.asarray(mo_coeff)
        if (mo_coeff.ndim == 2 and mo_coeff.shape[0] == dm.shape[0] and
            mo_coeff.shape[1] <= dm.shape[1]):
            dm1 = lib.dot(mo_coeff, mo_coeff.T)
            if numpy.allclose(dm1, dm, rtol=1e-8, atol=1e-10):
                return mo_coeff

    occ, coeff = numpy.linalg.eigh(dm)
    if occ[0] < -1e-10:
        raise ValueError('density matrix is not positive semidefinite')
    mask = occ > 1e-12
    return coeff[:,mask] * numpy.sqrt(occ[mask])


def _unpack_aopair(out, packed, layout, pair):
    shlpr0, shlpr1 = layout.shlpr_loc[pair:pair+2]
    aopair0 = layout.aopair_loc[shlpr0]
    ao_loc = layout.ao_loc

    for ijsh in range(shlpr0, shlpr1):
        ish, jsh = layout.shlpr[ijsh]
        i0, i1 = ao_loc[ish:ish+2]
        j0, j1 = ao_loc[jsh:jsh+2]
        row0 = layout.aopair_loc[ijsh] - aopair0
        row1 = layout.aopair_loc[ijsh+1] - aopair0
        block = packed[row0:row1]

        di = i1 - i0
        dj = j1 - j0
        naux = packed.shape[1]
        if ish == jsh:
            buf = numpy.empty((di, dj, naux), dtype=packed.dtype)
            idx = numpy.triu_indices(di)
            buf[idx] = block
            buf[(idx[1],idx[0])] = block
            out[i0:i1,:,j0:j1] = buf.transpose(0,2,1)
        else:
            buf = block.reshape(di, dj, naux)
            out[i0:i1,:,j0:j1] = buf.transpose(0,2,1)
            out[j0:j1,:,i0:i1] = buf.transpose(1,2,0)

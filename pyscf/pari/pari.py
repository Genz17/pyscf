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

import ctypes
import numpy
import scipy.linalg

from pyscf import df
from pyscf import gto
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.lib import logger


libpari = lib.load_library('libpari')


class AOPAIR_LAYOUT:
    '''Sparse AO shell-pair layout grouped by canonical atom pairs.'''

    SPARSE = 0
    DENSE = 1

    def __init__(self, mol, shlpr_mask):
        shlpr_mask = numpy.asarray(shlpr_mask)
        if shlpr_mask.shape != (mol.nbas, mol.nbas):
            raise ValueError('shlpr_mask has incompatible shape')
        if shlpr_mask.dtype != numpy.bool_:
            raise TypeError('shlpr_mask must have boolean dtype')
        if not numpy.array_equal(shlpr_mask, shlpr_mask.T):
            raise ValueError('shlpr_mask must be symmetric')

        self.mol = mol
        self.natm = mol.natm
        self.nbas = mol.nbas
        self.ao_loc = mol.ao_loc_nr()
        self.aoslice_by_atom = mol.aoslice_by_atom()
        shlpr_mask = numpy.asarray(shlpr_mask, order='C')

        pair_atoms = []
        shlpr = []
        shlpr_loc = [0]
        aopair_loc = [0]
        pair_fill = []
        pair_id = numpy.full((self.natm, self.natm), -1, dtype=numpy.int32)

        for A in range(self.natm):
            ish0, ish1 = self.aoslice_by_atom[A,:2]
            nao_A = self.aoslice_by_atom[A,3] - self.aoslice_by_atom[A,2]
            for B in range(A, self.natm):
                jsh0, jsh1 = self.aoslice_by_atom[B,:2]
                nao_B = self.aoslice_by_atom[B,3] - self.aoslice_by_atom[B,2]
                pair_shlpr = []

                if A == B:
                    for ish in range(ish0, ish1):
                        for jsh in range(ish, jsh1):
                            if shlpr_mask[ish,jsh]:
                                pair_shlpr.append((ish,jsh))
                    ndense = nao_A * (nao_A+1) // 2
                else:
                    for ish in range(ish0, ish1):
                        for jsh in range(jsh0, jsh1):
                            if shlpr_mask[ish,jsh]:
                                pair_shlpr.append((ish,jsh))
                    ndense = nao_A * nao_B

                if not pair_shlpr:
                    continue

                pair = len(pair_atoms)
                pair_atoms.append((A,B))
                pair_id[A,B] = pair_id[B,A] = pair

                pair_aopair0 = aopair_loc[-1]
                for ish, jsh in pair_shlpr:
                    di = self.ao_loc[ish+1] - self.ao_loc[ish]
                    dj = self.ao_loc[jsh+1] - self.ao_loc[jsh]
                    shlpr.append((ish,jsh))
                    if ish == jsh:
                        aopair_loc.append(aopair_loc[-1] + di*(di+1)//2)
                    else:
                        aopair_loc.append(aopair_loc[-1] + di*dj)

                shlpr_loc.append(len(shlpr))
                nkeep = aopair_loc[-1] - pair_aopair0
                pair_fill.append(nkeep / ndense)

        self.pair_atoms = numpy.asarray(
            pair_atoms, dtype=numpy.int32).reshape(-1,2)
        self.pair_id = pair_id
        self.shlpr = numpy.asarray(shlpr, dtype=numpy.int32).reshape(-1,2)
        self.shlpr_loc = numpy.asarray(shlpr_loc, dtype=numpy.int64)
        self.aopair_loc = numpy.asarray(aopair_loc, dtype=numpy.int64)
        self.pair_aopair_loc = self.aopair_loc[self.shlpr_loc]
        self.pair_fill = numpy.asarray(pair_fill, dtype=numpy.double)
        self.pair_kind = numpy.full(
            len(self.pair_atoms), self.SPARSE, dtype=numpy.uint8)
        self.naopair_by_pair = numpy.diff(self.pair_aopair_loc)
        self.npair = len(self.pair_atoms)
        self.nshlpr = len(self.shlpr)
        self.naopair = int(self.aopair_loc[-1])

    def get_pair_id(self, A, B):
        if not (0 <= A < self.natm and 0 <= B < self.natm):
            raise IndexError('atom index out of range')
        pair = self.pair_id[A,B]
        if pair < 0:
            raise KeyError('atom pair has no retained shell pair')
        return pair

    def get_shlpr_slice(self, A, B):
        pair = self.get_pair_id(A, B)
        return slice(*self.shlpr_loc[pair:pair+2])

    def get_aopair_slice(self, A, B):
        pair = self.get_pair_id(A, B)
        return slice(*self.pair_aopair_loc[pair:pair+2])


class PARI_COEFF:
    '''Sparse PARI coefficients with separately contiguous endpoint blocks.'''

    def __init__(self, aopair_layout, auxslice_by_atom, dtype=numpy.float64):
        self.aopair_layout = aopair_layout
        self.auxslice_by_atom = numpy.asarray(auxslice_by_atom).copy()
        if self.auxslice_by_atom.shape != (aopair_layout.natm, 4):
            raise ValueError('inconsistent auxiliary and AO slices')

        self.naux_by_atom = (
            self.auxslice_by_atom[:,3] - self.auxslice_by_atom[:,2])
        self._offsets = numpy.empty(
            (aopair_layout.npair, 3), dtype=numpy.int64)

        offset = 0
        for pair, (A,B) in enumerate(aopair_layout.pair_atoms):
            naopair = aopair_layout.naopair_by_pair[pair]
            self._offsets[pair,0] = offset
            offset += naopair * self.naux_by_atom[A]
            self._offsets[pair,1] = offset
            if A != B:
                offset += naopair * self.naux_by_atom[B]
            self._offsets[pair,2] = offset

        self._data = numpy.empty(offset, dtype=dtype)
        self.aux_loc = numpy.asarray(numpy.append(
            self.auxslice_by_atom[:,2],
            self.auxslice_by_atom[-1,3]), dtype=numpy.int64)
        self._build_d_layout()

    def _build_d_layout(self):
        layout = self.aopair_layout
        target_loc = numpy.empty(
            (layout.natm, layout.nbas+1), dtype=numpy.int64)
        source_shell = []
        coeff_offset = []
        edge_kind = []
        offset = 0

        for A in range(layout.natm):
            target_edges = [[] for i in range(layout.nbas)]
            naux = self.naux_by_atom[A]
            for pair, (B, C) in enumerate(layout.pair_atoms):
                if A == B:
                    coeff0 = self._offsets[pair,0]
                elif A == C:
                    coeff0 = self._offsets[pair,1]
                else:
                    continue

                shlpr0, shlpr1 = layout.shlpr_loc[pair:pair+2]
                aopair0 = layout.aopair_loc[shlpr0]
                for ijsh in range(shlpr0, shlpr1):
                    ish, jsh = layout.shlpr[ijsh]
                    row0 = layout.aopair_loc[ijsh] - aopair0
                    c0 = coeff0 + row0*naux
                    if ish == jsh:
                        target_edges[ish].append((ish,c0,2))
                    else:
                        target_edges[jsh].append((ish,c0,0))
                        target_edges[ish].append((jsh,c0,1))

            target_loc[A,0] = offset
            for ish, edges in enumerate(target_edges):
                for source, c0, kind in edges:
                    source_shell.append(source)
                    coeff_offset.append(c0)
                    edge_kind.append(kind)
                    offset += 1
                target_loc[A,ish+1] = offset

        self.d_target_loc = target_loc
        self.d_source_shell = numpy.asarray(
            source_shell, dtype=numpy.int32)
        self.d_coeff_offset = numpy.asarray(
            coeff_offset, dtype=numpy.int64)
        self.d_edge_kind = numpy.asarray(
            edge_kind, dtype=numpy.uint8)

    def left(self, A, B):
        pair = self.aopair_layout.get_pair_id(A, B)
        A, B = self.aopair_layout.pair_atoms[pair]
        naux = self.naux_by_atom[A]
        naopair = self.aopair_layout.naopair_by_pair[pair]
        p0, p1 = self._offsets[pair,:2]
        return self._data[p0:p1].reshape(naopair, naux)

    def right(self, A, B):
        pair = self.aopair_layout.get_pair_id(A, B)
        A, B = self.aopair_layout.pair_atoms[pair]
        if A == B:
            raise ValueError('diagonal atom pair has no right block')
        naux = self.naux_by_atom[B]
        naopair = self.aopair_layout.naopair_by_pair[pair]
        p0, p1 = self._offsets[pair,1:]
        return self._data[p0:p1].reshape(naopair, naux)

    def get_pair(self, A, B):
        pair = self.aopair_layout.get_pair_id(A, B)
        atom1, atom2 = self.aopair_layout.pair_atoms[pair]
        if atom1 == atom2:
            return (self.left(A, B),)
        return self.left(A, B), self.right(A, B)


class NBX_LAYOUT:
    '''Sparse auxiliary-function--AO-function pairs grouped by aux atom.'''

    def __init__(self, df_coeff):
        layout = df_coeff.aopair_layout
        target_ao_loc = numpy.full(
            (layout.natm, layout.nbas), -1, dtype=numpy.int32)
        ao_idx = []
        nao_by_aux_atom = numpy.empty(layout.natm, dtype=numpy.int32)

        for A in range(layout.natm):
            idx = []
            nao = 0
            target_loc = df_coeff.d_target_loc[A]
            for ish in range(layout.nbas):
                if target_loc[ish] == target_loc[ish+1]:
                    continue
                i0, i1 = layout.ao_loc[ish:ish+2]
                target_ao_loc[A,ish] = nao
                idx.extend(range(i0, i1))
                nao += i1 - i0
            ao_idx.append(numpy.asarray(idx, dtype=numpy.int32))
            nao_by_aux_atom[A] = nao

        self.target_ao_loc = target_ao_loc
        self.ao_idx = ao_idx
        self.nao_by_aux_atom = nao_by_aux_atom
        self.npair = int(numpy.dot(
            df_coeff.naux_by_atom, nao_by_aux_atom))


def get_shlpr_mask(mol, tol=1e-12):
    from pyscf.scf import _vhf
    opt = _vhf._VHFOpt(mol, 'int2e', 'CVHFnrs8_prescreen',
                       'CVHFnr_int2e_q_cond', 'CVHFnr_dm_cond')
    return opt.get_q_cond() > tol


def fill_aux_e2(mol, auxmol_or_auxbasis, intor='int3c2e', aosym='s1',
                comp=None, out=None, cintopt=None, shls_slice=None,
                shlpr_mask=None):
    '''3-center AO integrals (ij|L), evaluated with libpari.

    The shell indices for the auxiliary basis in ``shls_slice`` are relative
    to ``auxmol``, as in :func:`pyscf.df.incore.aux_e2`.

    ``shlpr_mask`` is an optional Boolean array of shape
    ``(mol.nbas, mol.nbas)``. False shell pairs are skipped and their dense
    output blocks are set to zero.
    '''
    if aosym != 's1':
        raise NotImplementedError('fill_aux_e2 only supports aosym=s1')

    if isinstance(auxmol_or_auxbasis, gto.MoleBase):
        auxmol = auxmol_or_auxbasis
    else:
        auxmol = df.addons.make_auxmol(mol, auxmol_or_auxbasis)

    if not mol.cart and auxmol.cart:
        raise NotImplementedError('Interface for int3c2e_ssc')
    elif mol.cart and not auxmol.cart:
        raise RuntimeError('Cartesian orbitals for mol and spherical orbitals '
                           'for auxmol not supported')

    if shls_slice is None:
        shls_slice = (0, mol.nbas, 0, mol.nbas,
                      mol.nbas, mol.nbas+auxmol.nbas)
    else:
        assert len(shls_slice) == 6
        assert shls_slice[1] <= mol.nbas
        assert shls_slice[3] <= mol.nbas
        assert shls_slice[5] <= auxmol.nbas
        shls_slice = list(shls_slice)
        shls_slice[4] += mol.nbas
        shls_slice[5] += mol.nbas

    intor = mol._add_suffix(intor)
    intor, comp = moleintor._get_intor_and_comp(intor, comp)
    if 'spinor' in intor:
        raise NotImplementedError('spinor integrals are not supported')

    atm, bas, env = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env,
        auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = moleintor.make_loc(bas, intor)

    if shlpr_mask is None:
        p_shlpr_mask = lib.c_null_ptr()
    else:
        shlpr_mask = numpy.asarray(shlpr_mask)
        if shlpr_mask.shape != (mol.nbas, mol.nbas):
            raise ValueError('shlpr_mask has incompatible shape')
        if shlpr_mask.dtype != numpy.bool_:
            raise TypeError('shlpr_mask must have boolean dtype')
        shlpr_mask = numpy.asarray(shlpr_mask, order='C')
        p_shlpr_mask = shlpr_mask.ctypes.data_as(ctypes.c_void_p)

    i0, i1, j0, j1, k0, k1 = shls_slice
    shape = (ao_loc[i1]-ao_loc[i0],
             ao_loc[j1]-ao_loc[j0],
             ao_loc[k1]-ao_loc[k0], comp)
    mat = numpy.ndarray(shape, numpy.double, out, order='F')

    if mat.size > 0:
        if cintopt is None:
            cintopt = moleintor.make_cintopt(
                atm, bas[:max(i1, j1)], env, intor)

        libpari.fill_aux_e2(
            getattr(moleintor.libcgto, intor),
            mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*6)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            p_shlpr_mask, ctypes.c_int(mol.nbas), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p))

    mat = numpy.rollaxis(mat, -1, 0)
    if comp == 1:
        mat = mat[0]
    return mat


def fill_aux_e2_sparse(mol, auxmol_or_auxbasis, aopair_layout, aux_atom,
                       atom_pair=None, intor='int3c2e', comp=None, out=None,
                       cintopt=None):
    '''Sparse 3-center AO integrals grouped by canonical AO atom pairs.

    For one auxiliary atom, the output has shape ``(naopair,naux)`` and is
    C-contiguous. If ``atom_pair`` is given, only that atom-pair block is
    returned. Otherwise, all blocks are concatenated in ``aopair_layout``.
    '''
    if not isinstance(aopair_layout, AOPAIR_LAYOUT):
        raise TypeError('aopair_layout must be an AOPAIR_LAYOUT')
    if (aopair_layout.natm != mol.natm or
        aopair_layout.nbas != mol.nbas or
        not numpy.array_equal(aopair_layout.ao_loc, mol.ao_loc_nr())):
        raise ValueError('aopair_layout is incompatible with mol')

    if isinstance(auxmol_or_auxbasis, gto.MoleBase):
        auxmol = auxmol_or_auxbasis
    else:
        auxmol = df.addons.make_auxmol(mol, auxmol_or_auxbasis)
    if auxmol.natm != mol.natm:
        raise ValueError('inconsistent auxiliary and AO atoms')

    if not mol.cart and auxmol.cart:
        raise NotImplementedError('Interface for int3c2e_ssc')
    elif mol.cart and not auxmol.cart:
        raise RuntimeError('Cartesian orbitals for mol and spherical orbitals '
                           'for auxmol not supported')

    if not 0 <= aux_atom < auxmol.natm:
        raise IndexError('auxiliary atom index out of range')

    aoslice = aopair_layout.aoslice_by_atom
    pair = None
    if atom_pair is None:
        shlpr0 = 0
        shlpr1 = aopair_layout.nshlpr
        i0, i1 = 0, mol.nbas
        j0, j1 = 0, mol.nbas
    else:
        if len(atom_pair) != 2:
            raise ValueError('atom_pair must contain two atom indices')
        pair = aopair_layout.get_pair_id(*atom_pair)
        A, B = aopair_layout.pair_atoms[pair]
        shlpr0, shlpr1 = aopair_layout.shlpr_loc[pair:pair+2]
        i0, i1 = aoslice[A,:2]
        j0, j1 = aoslice[B,:2]

    if pair is None:
        pair_kind = aopair_layout.pair_kind
    else:
        pair_kind = aopair_layout.pair_kind[pair:pair+1]
    if numpy.any(pair_kind != AOPAIR_LAYOUT.SPARSE):
        raise NotImplementedError('dense atom-pair blocks are not implemented')

    auxslice = auxmol.aoslice_by_atom()
    k0, k1 = auxslice[aux_atom,:2] + mol.nbas
    shls_slice = (i0, i1, j0, j1, k0, k1)

    intor = mol._add_suffix(intor)
    intor, comp = moleintor._get_intor_and_comp(intor, comp)
    if 'spinor' in intor:
        raise NotImplementedError('spinor integrals are not supported')

    atm, bas, env = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env,
        auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = moleintor.make_loc(bas, intor)

    nrow = (aopair_layout.aopair_loc[shlpr1] -
            aopair_layout.aopair_loc[shlpr0])
    naux = ao_loc[k1] - ao_loc[k0]
    mat = numpy.ndarray((comp,nrow,naux), numpy.double, out, order='C')

    if mat.size > 0:
        if cintopt is None:
            cintopt = moleintor.make_cintopt(
                atm, bas[:max(i1, j1)], env, intor)

        libpari.fill_aux_e2_sparse(
            getattr(moleintor.libcgto, intor),
            mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            ctypes.c_int(shlpr0), ctypes.c_int(shlpr1),
            aopair_layout.shlpr.ctypes.data_as(ctypes.c_void_p),
            aopair_layout.aopair_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*6)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p))

    if comp == 1:
        mat = mat[0]
    return mat


def fill_j2c(auxmol, aux_atom, out=None):
    '''Two-center metric panel ``(all auxiliary AOs|auxiliary AOs on A)``.'''
    if not 0 <= aux_atom < auxmol.natm:
        raise IndexError('auxiliary atom index out of range')
    auxslice = auxmol.aoslice_by_atom()
    sh0, sh1 = auxslice[aux_atom,:2]
    mat = auxmol.intor(
        'int2c2e', hermi=0, shls_slice=(sh0,sh1,0,auxmol.nbas),
        out=out)
    return mat.T


def _half_transform(mo_coeff, df_coeff, aux_atom,
                    out_ao_loc, nao_out, out=None):
    layout = df_coeff.aopair_layout
    if not 0 <= aux_atom < layout.natm:
        raise IndexError('auxiliary atom index out of range')
    mo_coeff = numpy.asarray(
        mo_coeff, dtype=numpy.double, order='C')
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != layout.ao_loc[-1]:
        raise ValueError('mo_coeff has incompatible shape')

    nmo = mo_coeff.shape[1]
    naux = df_coeff.naux_by_atom[aux_atom]
    mat = numpy.ndarray((nmo,naux,nao_out), numpy.double, out, order='C')
    libpari.PARIhalf_transform(
        mat.ctypes.data_as(ctypes.c_void_p),
        mo_coeff.ctypes.data_as(ctypes.c_void_p),
        df_coeff._data.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nmo), ctypes.c_int(naux), ctypes.c_int(nao_out),
        ctypes.c_int(layout.nbas),
        layout.ao_loc.ctypes.data_as(ctypes.c_void_p),
        out_ao_loc.ctypes.data_as(ctypes.c_void_p),
        df_coeff.d_target_loc[aux_atom].ctypes.data_as(ctypes.c_void_p),
        df_coeff.d_source_shell.ctypes.data_as(ctypes.c_void_p),
        df_coeff.d_coeff_offset.ctypes.data_as(ctypes.c_void_p),
        df_coeff.d_edge_kind.ctypes.data_as(ctypes.c_void_p))
    return mat


class PARI(lib.StreamObject):
    '''Pair-atomic resolution-of-identity object.'''

    _keys = {
        'mol', 'auxmol', 'schwarz_tol', 'shlpr_mask',
        'aopair_layout', 'nbx_layout', 'df_coeff',
    }

    def __init__(self, mol, auxbasis=None, schwarz_tol=1e-12):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self._auxbasis = auxbasis
        self.schwarz_tol = schwarz_tol

        self.auxmol = None
        self.shlpr_mask = None
        self.aopair_layout = None
        self.nbx_layout = None
        self.df_coeff = None

        self._j_df = df.DF(mol, auxbasis)

    @property
    def auxbasis(self):
        return self._auxbasis

    @auxbasis.setter
    def auxbasis(self, x):
        if self._auxbasis != x:
            self._auxbasis = x
            self.reset()

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        if self.auxmol is None:
            log.info('auxbasis = %s', self.auxbasis)
        else:
            log.info('auxbasis = auxmol.basis = %s', self.auxmol.basis)
        log.info('schwarz_tol = %g', self.schwarz_tol)
        log.info('max_memory = %s', self.max_memory)
        return self

    def dump_sparsity(self, verbose=None):
        log = logger.new_logger(self, verbose)
        mol = self.mol
        layout = self.aopair_layout
        auxslice = self.auxmol.aoslice_by_atom()
        aoslice = layout.aoslice_by_atom

        natmpair = mol.natm * (mol.natm+1) // 2
        nshlpair = mol.nbas * (mol.nbas+1) // 2
        nao = mol.nao_nr()
        naopair = nao * (nao+1) // 2
        log.info('PARI important atom pairs = %d / %d (%.1f%%)',
                 layout.npair, natmpair, layout.npair/natmpair*100)
        log.info('PARI important shell pairs = %d / %d (%.1f%%)',
                 layout.nshlpr, nshlpair, layout.nshlpr/nshlpair*100)
        log.info('PARI important AO pairs = %d / %d (%.1f%%)',
                 layout.naopair, naopair, layout.naopair/naopair*100)

        naux_by_atom = auxslice[:,3] - auxslice[:,2]
        naux = auxslice[-1,3]
        nao_by_atom = aoslice[:,3] - aoslice[:,2]
        nbx_dense = naux * nao
        log.info('PARI NBX auxiliary-AO pairs = %d / %d (%.1f%%)',
                 self.nbx_layout.npair, nbx_dense,
                 self.nbx_layout.npair/nbx_dense*100)
        dense_size = 0
        for A in range(mol.natm):
            dense_size += (nao_by_atom[A] * (nao_by_atom[A]+1) // 2 *
                           naux_by_atom[A])
            for B in range(A+1, mol.natm):
                dense_size += (nao_by_atom[A] * nao_by_atom[B] *
                               (naux_by_atom[A] + naux_by_atom[B]))

        itemsize = self.df_coeff._data.itemsize
        coeff_mem = self.df_coeff._data.nbytes
        dense_mem = dense_size * itemsize
        log.info('PARI fitting coefficients = %.2f MB / %.2f MB dense '
                 '(%.1f%%)', coeff_mem/1e6, dense_mem/1e6,
                 coeff_mem/dense_mem*100)

        nocc = mol.nelectron // 2
        max_naux = naux_by_atom.max(initial=0)
        max_nactive = self.nbx_layout.nao_by_aux_atom.max(initial=0)
        d_buf_mem = numpy.max(
            nocc * naux_by_atom *
            self.nbx_layout.nao_by_aux_atom, initial=0) * itemsize
        h_buf_mem = nocc * max_naux * nao * itemsize
        g_buf_mem = max_naux * nao**2 * itemsize
        l_buf_mem = max_nactive * nao * itemsize
        buf_mem = d_buf_mem + h_buf_mem + g_buf_mem + l_buf_mem
        dense_buf_mem = max_naux * (
            2*nocc*nao + nao**2) * itemsize
        metric_mem = naux**2 * itemsize
        metric_panel_mem = naux * max_naux * itemsize
        max_shell = numpy.diff(layout.ao_loc).max(initial=0)
        d_thread_mem = (lib.num_threads() * max_shell * max_naux *
                        (nocc + max_shell) * itemsize)
        max_pair = layout.naopair_by_pair.max(initial=0)
        e_thread_mem = (lib.num_threads() * max_pair *
                        max_naux * itemsize)
        thread_mem = max(d_thread_mem, e_thread_mem)
        matrix_mem = 2 * nao**2 * itemsize
        peak_mem = (coeff_mem + metric_panel_mem + buf_mem +
                    thread_mem + matrix_mem)
        log.info('Estimated PARI K peak memory = %.2f MB (nocc = %d)',
                 peak_mem/1e6, nocc)
        log.info('  coefficients %.2f MB, j2c panel %.2f MB, '
                 'NBX-D/H/dense-G/L-row buffers %.2f MB, '
                 'L/K matrices %.2f MB',
                 coeff_mem/1e6, metric_panel_mem/1e6, buf_mem/1e6,
                 matrix_mem/1e6)
        log.info('  C-kernel thread buffers %.2f MB (%d threads)',
                 thread_mem/1e6, lib.num_threads())
        log.info('  excluding the DF-J cache and caller-owned dm/mo_coeff')
        log.info('  dense D/H/G buffers require %.2f MB',
                 dense_buf_mem/1e6)
        log.info('  full fitting j2c requires %.2f MB and is not retained',
                 metric_mem/1e6)
        return self

    def _build_auxmol(self):
        if self.auxmol is None:
            if self._j_df.auxmol is not None:
                self.auxmol = self._j_df.auxmol
            else:
                self.auxmol = df.addons.make_auxmol(
                    self.mol, self.auxbasis)
        if self.auxmol.natm != self.mol.natm:
            raise ValueError('inconsistent auxiliary and AO atoms')
        return self.auxmol

    def _build_aopair_layout(self):
        if self.aopair_layout is None:
            self.shlpr_mask = get_shlpr_mask(
                self.mol, self.schwarz_tol)
            self.aopair_layout = AOPAIR_LAYOUT(
                self.mol, self.shlpr_mask)
        return self.aopair_layout

    def fill_aux_e2(self, *args, **kwargs):
        self._build_auxmol()
        self._build_aopair_layout()
        kwargs.setdefault('shlpr_mask', self.shlpr_mask)
        return fill_aux_e2(self.mol, self.auxmol, *args, **kwargs)

    def fill_aux_e2_sparse(self, aux_atom, atom_pair=None, **kwargs):
        self._build_auxmol()
        self._build_aopair_layout()
        return fill_aux_e2_sparse(
            self.mol, self.auxmol, self.aopair_layout, aux_atom,
            atom_pair=atom_pair, **kwargs)

    def fill_j2c(self, aux_atom, out=None):
        self._build_auxmol()
        return fill_j2c(self.auxmol, aux_atom, out)

    def half_transform(self, mo_coeff, aux_atom, compact=True, out=None):
        '''Half-transform PARI coefficients for one auxiliary atom.

        The compact output contains only the NBX-active AO functions.
        '''
        if self.df_coeff is None:
            self.build()
        if not 0 <= aux_atom < self.mol.natm:
            raise IndexError('auxiliary atom index out of range')
        if compact:
            nao = int(self.nbx_layout.nao_by_aux_atom[aux_atom])
            out_ao_loc = self.nbx_layout.target_ao_loc[aux_atom]
        else:
            nao = self.aopair_layout.ao_loc[-1]
            out_ao_loc = self.aopair_layout.ao_loc
        return _half_transform(
            mo_coeff, self.df_coeff, aux_atom, out_ao_loc, nao, out)

    def fitting(self):
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(self)
        tspans = numpy.zeros((4,2))
        mol = self.mol

        tick = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        auxmol = self._build_auxmol()
        layout = self._build_aopair_layout()
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[0] += tock - tick

        tick = tock
        j2c = df.incore.fill_2c2e(mol, auxmol)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[1] += tock - tick

        auxslice = auxmol.aoslice_by_atom()
        df_coeff = PARI_COEFF(layout, auxslice, j2c.dtype)
        for A, B in layout.pair_atoms:
            tick = numpy.asarray(
                (logger.process_clock(), logger.perf_counter()))
            j3c_A = self.fill_aux_e2_sparse(A, (A, B))
            j3c_B = None
            if A != B:
                j3c_B = self.fill_aux_e2_sparse(B, (A, B))
            tock = numpy.asarray(
                (logger.process_clock(), logger.perf_counter()))
            tspans[2] += tock - tick

            tick = tock
            if A == B:
                auxidx = slice(*auxslice[A,2:])
                coeff = scipy.linalg.solve(
                    j2c[auxidx,auxidx], j3c_A.T).T
                df_coeff.left(A, B)[:] = coeff
            else:
                auxidx_A = numpy.arange(*auxslice[A,2:])
                auxidx_B = numpy.arange(*auxslice[B,2:])
                auxidx = numpy.hstack((auxidx_A, auxidx_B))
                j3c = numpy.hstack((j3c_A, j3c_B))
                coeff = scipy.linalg.solve(
                    j2c[numpy.ix_(auxidx,auxidx)], j3c.T).T
                naux_A = len(auxidx_A)
                df_coeff.left(A, B)[:] = coeff[:,:naux_A]
                df_coeff.right(A, B)[:] = coeff[:,naux_A:]
            tock = numpy.asarray(
                (logger.process_clock(), logger.perf_counter()))
            tspans[3] += tock - tick

        self.df_coeff = df_coeff
        self.nbx_layout = NBX_LAYOUT(df_coeff)
        self.dump_sparsity()
        tnames = ('setup', 'j2c', 'j3c', 'solve')
        for name, tspan in zip(tnames, tspans):
            cpu0 = logger.process_clock() - tspan[0]
            wall0 = logger.perf_counter() - tspan[1]
            log.timer('PARI fitting ' + name, cpu0, wall0)
        log.timer('PARI fitting', *t0)
        return self

    def build(self):
        self.check_sanity()
        self.dump_flags()
        return self.fitting()

    def kernel(self):
        return self.build()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
            self.stdout = mol.stdout
            self.verbose = mol.verbose
            self.max_memory = mol.max_memory
        self.auxmol = None
        self.shlpr_mask = None
        self.aopair_layout = None
        self.nbx_layout = None
        self.df_coeff = None
        self._j_df.reset(self.mol)
        self._j_df.auxbasis = self.auxbasis
        return self

    def get_j(self, dm, hermi=1, direct_scf_tol=1e-13, omega=None):
        from pyscf.pari import pari_jk
        return pari_jk.get_j(
            self, dm, hermi, direct_scf_tol, omega)

    def get_k(self, dm, hermi=1, mo_coeff=None, mo_occ=None, omega=None):
        from pyscf.pari import pari_jk
        return pari_jk.get_k(
            self, dm, hermi, mo_coeff, mo_occ, omega)

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, mo_coeff=None, mo_occ=None,
               omega=None):
        from pyscf.pari import pari_jk
        return pari_jk.get_jk(
            self, dm, hermi, with_j, with_k, direct_scf_tol,
            mo_coeff, mo_occ, omega)

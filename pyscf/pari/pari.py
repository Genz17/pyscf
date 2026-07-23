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

from pyscf import df
from pyscf import gto
from pyscf import lib
from pyscf.gto import moleintor


libpari = lib.load_library('libpari')


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

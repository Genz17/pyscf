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

from pyscf import gto
from pyscf import lib
from pyscf.df import df_jk
from pyscf.gto import moleintor
from pyscf.lib import logger
from pyscf.pari import pari as pari_module


def pari(mf, auxbasis=None, with_pari=None, schwarz_tol=1e-12):
    '''Apply PARI J/K builders to an RHF object.

    Args:
        mf (pyscf.scf.hf.RHF):
            Mean-field object to decorate.
        auxbasis (str or dict):
            Auxiliary-basis specification. The PySCF default JK-fitting
            basis is used when omitted.
        with_pari (pyscf.pari.PARI):
            Optional prebuilt PARI object.
        schwarz_tol (float):
            AO shell-pair Schwarz threshold. Default is 1e-12.

    Returns:
        pyscf.scf.hf.RHF:
            RHF object whose J/K builder is backed by ``with_pari``.
    '''
    from pyscf import scf

    if (not isinstance(mf, scf.hf.RHF) or mf.istype('ROHF') or
        isinstance(mf, scf.hf.KohnShamDFT)):
        raise NotImplementedError('PARI SCF only supports RHF')
    if isinstance(mf, df_jk._DFHF):
        raise RuntimeError('PARI cannot be combined with density_fit')

    if with_pari is None:
        with_pari = pari_module.PARI(
            mf.mol, auxbasis=auxbasis, schwarz_tol=schwarz_tol)
        with_pari.max_memory = mf.max_memory
        with_pari.stdout = mf.stdout
        with_pari.verbose = mf.verbose
    elif not isinstance(with_pari, pari_module.PARI):
        raise TypeError('with_pari must be a PARI object')
    elif with_pari.mol is not mf.mol:
        raise ValueError('with_pari and mf must use the same mol')

    if isinstance(mf, _PARIHF):
        mf = mf.copy()
        mf.with_pari = with_pari
        mf.direct_scf = False
        return mf

    parimf = _PARIHF(mf, with_pari)
    return lib.set_class(parimf, (_PARIHF, mf.__class__))


class _PARIHF:
    '''RHF mixin using global DF for J and PARI for K.'''

    __name_mixin__ = 'PARI'

    _keys = {'with_pari'}

    def __init__(self, mf, with_pari=None):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.with_pari = with_pari
        self.direct_scf = False

    def undo_pari(self):
        '''Remove the PARI mixin and return the underlying SCF object.'''
        obj = lib.view(self, lib.drop_class(self.__class__, _PARIHF))
        del obj.with_pari
        return obj

    def reset(self, mol=None):
        '''Reset the SCF and associated PARI objects.'''
        if self.with_pari:
            self.with_pari.reset(mol)
        return super().reset(mol)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        '''Build Coulomb and exchange matrices through ``with_pari``.'''
        assert (with_j or with_k)
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if not self.with_pari:
            return super().get_jk(mol, dm, hermi, with_j, with_k, omega)
        if mol is not self.mol:
            raise ValueError('PARI does not support a different mol in get_jk')

        return self.with_pari.get_jk(
            dm, hermi, with_j, with_k, self.direct_scf_tol,
            omega=omega)

    def density_fit(self, *args, **kwargs):
        '''Reject simultaneous PARI and conventional DF SCF mixins.'''
        raise RuntimeError('PARI cannot be combined with density_fit')

    def nuc_grad_method(self):
        '''Return a PARI gradient object when gradient support is available.'''
        raise NotImplementedError('PARI nuclear gradients are not supported')
    Gradients = nuc_grad_method

    def Hessian(self):
        '''Return a PARI Hessian object when Hessian support is available.'''
        raise NotImplementedError('PARI Hessians are not supported')


def _fill_g(mypari, aux_atom, j2c, intor='int3c2e',
            out=None, cintopt=None):
    '''Build one metric-corrected sparse G panel.

    Args:
        mypari (pyscf.pari.PARI):
            Initialized PARI object.
        aux_atom (int):
            Atom whose auxiliary functions form the panel.
        j2c (numpy.ndarray):
            Metric panel with shape ``(naux, naux_A)``.
        intor (str):
            Three-center integral name. Default is ``'int3c2e'``.
        out (numpy.ndarray):
            Optional output buffer.
        cintopt:
            Optional libcint optimizer shared across auxiliary atoms.

    Returns:
        numpy.ndarray:
            C-contiguous G panel with shape ``(naopair, naux_A)``.
    '''
    return _fill_g_drv(
        mypari, aux_atom, j2c, intor, out, cintopt)


def _fill_gj(mypari, aux_atom, j2c, rho, vj, intor='int3c2e',
             out=None, cintopt=None):
    '''Build one G panel and accumulate its raw integrals into J.

    Args:
        mypari (pyscf.pari.PARI):
            Initialized PARI object.
        aux_atom (int):
            Atom whose auxiliary functions form the panel.
        j2c (numpy.ndarray):
            Metric panel with shape ``(naux, naux_A)``.
        rho (numpy.ndarray):
            Fitted Coulomb density for the current auxiliary atom.
        vj (numpy.ndarray):
            C-contiguous AO Coulomb matrix updated in place.
        intor (str):
            Three-center integral name. Default is ``'int3c2e'``.
        out (numpy.ndarray):
            Optional output buffer.
        cintopt:
            Optional libcint optimizer shared across auxiliary atoms.

    Returns:
        numpy.ndarray:
            C-contiguous G panel with shape ``(naopair, naux_A)``.
    '''
    return _fill_g_drv(
        mypari, aux_atom, j2c, intor, out, cintopt, rho, vj)


def _fill_g_drv(mypari, aux_atom, j2c, intor='int3c2e',
                out=None, cintopt=None, rho=None, vj=None):
    '''Common driver for metric-corrected G and fused J/G panels.'''
    mol = mypari.mol
    auxmol = mypari.auxmol
    layout = mypari.aopair_layout
    df_coeff = mypari.df_coeff
    if numpy.any(layout.pair_kind != pari_module.AOPAIR_LAYOUT.SPARSE):
        raise NotImplementedError('dense atom-pair blocks are not implemented')
    if not 0 <= aux_atom < auxmol.natm:
        raise IndexError('auxiliary atom index out of range')

    if not mol.cart and auxmol.cart:
        raise NotImplementedError('Interface for int3c2e_ssc')
    elif mol.cart and not auxmol.cart:
        raise RuntimeError('Cartesian orbitals for mol and spherical orbitals '
                           'for auxmol not supported')

    auxslice = auxmol.aoslice_by_atom()
    k0, k1 = auxslice[aux_atom,:2] + mol.nbas
    shls_slice = (0, mol.nbas, 0, mol.nbas, k0, k1)

    intor = mol._add_suffix(intor)
    intor, comp = moleintor._get_intor_and_comp(intor, None)
    if comp != 1:
        raise NotImplementedError('fill_g only supports one component')
    if 'spinor' in intor:
        raise NotImplementedError('spinor integrals are not supported')

    atm, bas, env = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env,
        auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = moleintor.make_loc(bas, intor)
    naux = ao_loc[k1] - ao_loc[k0]
    mat = numpy.ndarray(
        (layout.naopair,naux), numpy.double, out, order='C')

    j2c = numpy.asarray(j2c, dtype=numpy.double, order='C')
    if (j2c.ndim != 2 or j2c.shape[0] != df_coeff.aux_loc[-1] or
        j2c.shape[1] != naux):
        raise ValueError('j2c panel has incompatible shape')

    if (rho is None) != (vj is None):
        raise ValueError('rho and vj must be provided together')
    if rho is not None:
        rho = numpy.asarray(rho, dtype=numpy.double, order='C')
        vj = numpy.asarray(vj)
        if rho.shape != (naux,):
            raise ValueError('rho panel has incompatible shape')
        if (vj.shape != (mol.nao_nr(), mol.nao_nr()) or
            vj.dtype != numpy.double or not vj.flags.c_contiguous):
            raise ValueError('vj must be a C-contiguous double array')

    if mat.size > 0:
        if cintopt is None:
            cintopt = moleintor.make_cintopt(
                atm, bas[:mol.nbas], env, intor)

        if rho is None:
            fill = pari_module.libpari.PARIfill_g
            jargs = ()
        else:
            fill = pari_module.libpari.PARIfill_gj
            jargs = (
                vj.ctypes.data_as(ctypes.c_void_p),
                rho.ctypes.data_as(ctypes.c_void_p),
            )
        fill(
            getattr(moleintor.libcgto, intor),
            mat.ctypes.data_as(ctypes.c_void_p),
            *jargs,
            df_coeff._data.ctypes.data_as(ctypes.c_void_p),
            j2c.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(layout.npair),
            layout.pair_atoms.ctypes.data_as(ctypes.c_void_p),
            layout.pair_aopair_loc.ctypes.data_as(ctypes.c_void_p),
            layout.shlpr_loc.ctypes.data_as(ctypes.c_void_p),
            layout.shlpr.ctypes.data_as(ctypes.c_void_p),
            layout.aopair_loc.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            df_coeff._offsets.ctypes.data_as(ctypes.c_void_p),
            df_coeff.aux_loc.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*6)(*shls_slice), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p))
    return mat


def _build_l_nbx(L, D, H, ao_idx, out=None):
    '''Contract NBX-compressed D with dense H and accumulate into L.

    .. math::

        L_{\\mu\\nu} \\mathrel{+}=
        \\sum_{iP} D_{iP\\mu} H_{iP\\nu}.

    Args:
        L (numpy.ndarray):
            C-contiguous AO matrix updated in place.
        D (numpy.ndarray):
            NBX-compressed D tensor with shape
            ``(nmo, naux_A, nao_active)``.
        H (numpy.ndarray):
            H tensor with shape ``(nmo, naux_A, nao)``.
        ao_idx (numpy.ndarray):
            Global AO indices corresponding to the active D rows.
        out (numpy.ndarray):
            Optional contraction buffer.

    Returns:
        numpy.ndarray:
            The updated ``L`` matrix.
    '''
    L = numpy.asarray(L)
    D = numpy.asarray(D)
    H = numpy.asarray(H)
    ao_idx = numpy.asarray(ao_idx, dtype=numpy.int32, order='C')
    if (L.ndim != 2 or L.shape[0] != L.shape[1] or
        L.dtype != numpy.double or not L.flags.c_contiguous):
        raise ValueError('L must be a C-contiguous double array')
    if (D.ndim != 3 or H.ndim != 3 or
        D.shape[:2] != H.shape[:2] or H.shape[2] != L.shape[1] or
        D.shape[2] != len(ao_idx) or
        D.dtype != numpy.double or H.dtype != numpy.double or
        not D.flags.c_contiguous or not H.flags.c_contiguous):
        raise ValueError('D and H have incompatible shapes')

    buf = numpy.ndarray(
        (len(ao_idx),L.shape[1]), numpy.double, out, order='C')
    if buf.size > 0:
        ndp = D.shape[0] * D.shape[1]
        lib.dot(D.reshape(ndp,len(ao_idx)).T,
                H.reshape(ndp,L.shape[1]), c=buf)
        pari_module.libpari.PARIscatter_l_nbx(
            L.ctypes.data_as(ctypes.c_void_p),
            buf.ctypes.data_as(ctypes.c_void_p),
            ao_idx.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(L.shape[1]), ctypes.c_int(len(ao_idx)))
    return L


def get_j(mypari, dm, hermi=1, direct_scf_tol=1e-13, omega=None):
    '''Build the Coulomb matrix with global density fitting.

    Args:
        mypari (pyscf.pari.PARI):
            PARI object providing the molecule and auxiliary basis.
        dm (numpy.ndarray):
            AO density matrix.
        hermi (int):
            Hermiticity flag. Default is one.
        direct_scf_tol (float):
            Integral screening threshold. Default is 1e-13.
        omega (float):
            Range-separation parameter. Range separation is currently
            unsupported.

    Returns:
        numpy.ndarray:
            Coulomb matrix in the AO basis.
    '''
    if omega is not None:
        raise NotImplementedError('range-separated Coulomb is not supported')

    mypari._build_auxmol()
    mydf = mypari._j_df
    mydf.max_memory = mypari.max_memory
    mydf.stdout = mypari.stdout
    mydf.verbose = mypari.verbose
    vj = df_jk.get_j(mydf, dm, hermi, direct_scf_tol)
    return vj


def _get_jrho(mypari, dm, direct_scf_tol=1e-13):
    '''Build the fitted auxiliary density for DF-J.

    .. math::

        \\rho_P = \\sum_Q (P|Q)^{-1}
        \\sum_{\\mu\\nu} (Q|\\mu\\nu)D_{\\nu\\mu}.

    Args:
        mypari (pyscf.pari.PARI):
            PARI object providing the molecule and auxiliary basis.
        dm (numpy.ndarray):
            AO density matrix.
        direct_scf_tol (float):
            Integral screening threshold. Default is 1e-13.

    Returns:
        numpy.ndarray:
            C-contiguous fitted density with shape ``(naux,)``.
    '''
    from pyscf.scf import _vhf
    from pyscf.scf import jk

    log = logger.new_logger(mypari)
    mypari._build_auxmol()
    mydf = mypari._j_df
    mydf.max_memory = mypari.max_memory
    mydf.stdout = mypari.stdout
    mydf.verbose = mypari.verbose

    if mydf._vjopt is None:
        mol = mypari.mol
        auxmol = mypari.auxmol
        mydf.auxmol = auxmol
        opt = _vhf._VHFOpt(
            mol, 'int3c2e', 'CVHFnr3c2e_schwarz_cond',
            dmcondname='CVHFnr_dm_cond',
            direct_scf_tol=direct_scf_tol)
        opt.init_cvhf_direct(
            mol, 'int2e', 'CVHFnr_int2e_q_cond')

        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = numpy.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [
            j2c_diag[i0:i1].max()
            for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]
        opt.q_cond = numpy.hstack((opt.q_cond.ravel(), aux_q_cond))

        try:
            opt.j2c = scipy.linalg.cho_factor(j2c, lower=True)
            opt.j2c_type = 'cd'
        except scipy.linalg.LinAlgError:
            opt.j2c = j2c
            opt.j2c_type = 'regular'

        bas_placeholder = numpy.array(
            [0, 0, 1, 1, 0, 0, 0, 0], dtype=numpy.int32)
        fakemol = mol + auxmol
        fakemol._bas = numpy.vstack((fakemol._bas, bas_placeholder))
        opt.fakemol = fakemol
        mydf._vjopt = opt

    opt = mydf._vjopt
    mol = mypari.mol
    auxmol = mydf.auxmol
    dm = numpy.asarray(dm, dtype=numpy.double, order='C')
    if dm.shape != (mol.nao_nr(), mol.nao_nr()):
        raise NotImplementedError('fused PARI J/K supports one density matrix')

    nbas = mol.nbas
    nbas1 = nbas + auxmol.nbas
    shls_slice = (
        0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)
    t0 = (logger.process_clock(), logger.perf_counter())
    with lib.temporary_env(
            opt, prescreen='CVHFnr3c2e_vj_pass1_prescreen'):
        jaux = jk.get_jk(
            opt.fakemol, dm[numpy.newaxis],
            ['ijkl,ji->kl'], 'int3c2e', aosym='s2ij',
            hermi=0, shls_slice=shls_slice, vhfopt=opt)
    jaux = numpy.asarray(jaux)[:,:,0]
    log.timer('PARI JK J pass 1', *t0)

    t0 = (logger.process_clock(), logger.perf_counter())
    if opt.j2c_type == 'cd':
        rho = scipy.linalg.cho_solve(opt.j2c, jaux.T)
    else:
        rho = scipy.linalg.solve(opt.j2c, jaux.T)
    log.timer('PARI JK J solve', *t0)
    return numpy.asarray(rho[:,0], order='C')


def get_k(mypari, dm, hermi=1, mo_coeff=None, mo_occ=None, omega=None,
          s1e=None):
    '''Build the exchange matrix with PARI.

    Args:
        mypari (pyscf.pari.PARI):
            Initialized PARI object.
        dm (numpy.ndarray):
            AO density matrix.
        hermi (int):
            Hermiticity flag. Only one is supported.
        mo_coeff (numpy.ndarray):
            Optional AO-to-MO coefficients.
        mo_occ (numpy.ndarray):
            Optional MO occupations.
        omega (float):
            Range-separation parameter. Range separation is currently
            unsupported.
        s1e (numpy.ndarray):
            AO overlap matrix used when ``dm`` must be factorized.

    Returns:
        numpy.ndarray:
            Exchange matrix in the AO basis.
    '''
    return _get_k(
        mypari, dm, hermi, mo_coeff, mo_occ, omega, s1e)


def _get_k(mypari, dm, hermi=1, mo_coeff=None, mo_occ=None, omega=None,
           s1e=None, rho=None):
    '''Build PARI K and optionally accumulate fused DF-J pass 2.

    For the currently supported real-orbital case, the main intermediates
    are

    .. math::

        D_{iP\\mu} &= \\sum_\\sigma M_{\\sigma i}d_{P\\sigma\\mu}, \\\\
        G_{P\\lambda\\nu} &=
            (P|\\lambda\\nu)
            - \\frac{1}{2}\\sum_Q(P|Q)d_{Q\\lambda\\nu}, \\\\
        H_{iP\\nu} &= \\sum_\\lambda M_{\\lambda i}
            G_{P\\lambda\\nu}, \\\\
        L_{\\mu\\nu} &= \\sum_{iP}D_{iP\\mu}H_{iP\\nu}, \\\\
        K_{\\mu\\nu} &= L_{\\mu\\nu} + L_{\\nu\\mu}.

    These steps correspond to the D, E/G, H, and L intermediates in
    Table 1 of the Head-Gordon PARI-K paper. The implementation processes
    one auxiliary atom at a time. D is NBX-compressed, G retains the
    packed AO shell-pair layout, H is formed by a sparse half
    transformation, and L is accumulated into the dense AO matrix.
    '''
    # G retains the sparse AO-pair layout. The same target/source
    # half-transform used for D builds H without scattering G to dense.
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
    if s1e is None and (mo_coeff is None or mo_occ is None):
        s1e = mypari.mol.intor_symmetric('int1e_ovlp')
    mo_coeff = _factor_dm(dm, s1e, mo_coeff, mo_occ)
    if numpy.iscomplexobj(mo_coeff):
        raise NotImplementedError('complex orbitals are not supported')

    if mypari.df_coeff is None:
        mypari.build()

    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mypari)
    mol = mypari.mol
    auxmol = mypari.auxmol
    layout = mypari.aopair_layout
    nbx_layout = mypari.nbx_layout
    auxslice = auxmol.aoslice_by_atom()
    nocc = mo_coeff.shape[1]
    naux = auxmol.nao_nr()
    atm, bas, env = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env,
        auxmol._atm, auxmol._bas, auxmol._env)
    intor = mol._add_suffix('int3c2e')
    cintopt = moleintor.make_cintopt(
        atm, bas[:mol.nbas], env, intor)

    tnames = ('Dmat', 'j2c', 'Gmat', 'Hmat', 'Lmat')
    tspans = numpy.zeros((5,2))
    dtype = numpy.result_type(mo_coeff, numpy.double)
    naux_by_atom = auxslice[:,3] - auxslice[:,2]
    max_naux = numpy.max(naux_by_atom)
    max_nactive = numpy.max(nbx_layout.nao_by_aux_atom)
    max_d_size = numpy.max(
        nocc * naux_by_atom * nbx_layout.nao_by_aux_atom)
    mo_coeff = numpy.asarray(mo_coeff, dtype=dtype, order='C')
    Dbuf = numpy.empty(max_d_size, dtype=dtype)
    Hbuf = numpy.empty(nocc*max_naux*nao, dtype=dtype)
    Gbuf = numpy.empty(layout.naopair*max_naux, dtype=dtype)
    j2cbuf = numpy.empty(naux*max_naux, dtype=dtype)
    Lbuf = numpy.empty(max_nactive*nao, dtype=dtype)
    Lmat = numpy.zeros((nao, nao), dtype=dtype)
    if rho is not None:
        rho = numpy.asarray(rho, dtype=numpy.double, order='C')
        if rho.shape != (naux,):
            raise ValueError('rho has incompatible shape')
        vj = numpy.zeros((nao, nao), dtype=numpy.double)
    for A in range(mol.natm):
        aux0, aux1 = auxslice[A,2:]
        naux_A = auxslice[A,3] - auxslice[A,2]
        nactive = nbx_layout.nao_by_aux_atom[A]

        tick = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        out = Dbuf[:nocc*naux_A*nactive]
        Dmat = mypari.half_transform(mo_coeff, A, out=out)
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
        out = Gbuf[:layout.naopair*naux_A]
        if rho is None:
            Gmat = _fill_g(
                mypari, A, j2c, out=out, cintopt=cintopt)
        else:
            Gmat = _fill_gj(
                mypari, A, j2c, rho[aux0:aux1], vj,
                out=out, cintopt=cintopt)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[2] += tock - tick

        tick = tock
        out = Hbuf[:nocc*naux_A*nao]
        Hmat = pari_module._half_transform(
            mo_coeff, Gmat, naux_A, layout, layout.ao_loc, nao,
            layout.target_loc, layout.source_shell,
            layout.aopair_offset, layout.edge_kind,
            offset_scale=naux_A, out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[3] += tock - tick

        tick = tock
        out = Lbuf[:nactive*nao]
        _build_l_nbx(
            Lmat, Dmat, Hmat, nbx_layout.ao_idx[A], out=out)
        tock = numpy.asarray(
            (logger.process_clock(), logger.perf_counter()))
        tspans[4] += tock - tick

    vk = Lmat + Lmat.T
    prefix = 'PARI K' if rho is None else 'PARI JK'
    for name, tspan in zip(tnames, tspans):
        cpu0 = logger.process_clock() - tspan[0]
        wall0 = logger.perf_counter() - tspan[1]
        log.timer(prefix + ' ' + name, cpu0, wall0)

    pair_factor = 2 - (layout.pair_atoms[:,0] ==
                       layout.pair_atoms[:,1])
    log.debug(prefix + ' calls: D %d sparse dgemm; '
              'G %d fused pair jobs/%d metric products; '
              'H %d sparse dgemm; L %d NBX dgemm/scatter',
              numpy.count_nonzero(numpy.diff(
                  mypari.df_coeff.d_target_loc, axis=1)),
              mol.natm*layout.npair, mol.natm*pair_factor.sum(),
              mol.natm*numpy.count_nonzero(numpy.diff(layout.target_loc)),
              mol.natm)
    if rho is None:
        log.timer('PARI K', *t0)
        return vk
    log.timer('PARI JK K and J pass 2', *t0)
    return vj, vk


def get_k_slow(mypari, dm, hermi=1, mo_coeff=None, mo_occ=None, omega=None,
               s1e=None):
    '''Build PARI K with the reference dense-G Python implementation.

    Args:
        mypari (pyscf.pari.PARI):
            Initialized PARI object.
        dm (numpy.ndarray):
            AO density matrix.
        hermi (int):
            Hermiticity flag. Only one is supported.
        mo_coeff (numpy.ndarray):
            Optional AO-to-MO coefficients.
        mo_occ (numpy.ndarray):
            Optional MO occupations.
        omega (float):
            Range-separation parameter. Range separation is currently
            unsupported.
        s1e (numpy.ndarray):
            AO overlap matrix used when ``dm`` must be factorized.

    Returns:
        numpy.ndarray:
            Exchange matrix in the AO basis.
    '''
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
    if s1e is None and (mo_coeff is None or mo_occ is None):
        s1e = mypari.mol.intor_symmetric('int1e_ovlp')
    mo_coeff = _factor_dm(dm, s1e, mo_coeff, mo_occ)
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

    tnames = ('Dmat', 'j2c', 'Gmat', 'Emat',
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
           direct_scf_tol=1e-13, mo_coeff=None, mo_occ=None, omega=None,
           s1e=None):
    '''Build Coulomb and exchange matrices.

    J pass 2 is fused with the PARI K integral pass when both matrices are
    requested.

    Args:
        mypari (pyscf.pari.PARI):
            Initialized PARI object.
        dm (numpy.ndarray):
            AO density matrix.
        hermi (int):
            Hermiticity flag. Only one is supported when K is requested.
        with_j (bool):
            Build the Coulomb matrix. Default is True.
        with_k (bool):
            Build the exchange matrix. Default is True.
        direct_scf_tol (float):
            DF-J integral screening threshold. Default is 1e-13.
        mo_coeff (numpy.ndarray):
            Optional AO-to-MO coefficients.
        mo_occ (numpy.ndarray):
            Optional MO occupations.
        omega (float):
            Range-separation parameter. Range separation is currently
            unsupported.
        s1e (numpy.ndarray):
            AO overlap matrix used when ``dm`` must be factorized.

    Returns:
        tuple of numpy.ndarray:
            Coulomb and exchange matrices. A matrix is None when its
            corresponding build flag is False.
    '''
    assert (with_j or with_k)
    if with_j and with_k:
        if omega is not None:
            raise NotImplementedError(
                'range-separated J/K is not supported')
        if hermi != 1:
            raise NotImplementedError('PARI K only supports hermi=1')
        t0 = (logger.process_clock(), logger.perf_counter())
        rho = _get_jrho(mypari, dm, direct_scf_tol)
        vj, vk = _get_k(
            mypari, dm, hermi, mo_coeff, mo_occ, omega, s1e, rho)
        logger.timer(mypari, 'PARI JK', *t0)
        return vj, vk
    elif with_j:
        return get_j(
            mypari, dm, hermi, direct_scf_tol, omega), None
    else:
        return None, get_k(
            mypari, dm, hermi, mo_coeff, mo_occ, omega, s1e)


def _factor_dm(dm, s1e, mo_coeff=None, mo_occ=None):
    '''Factor a positive-semidefinite AO density matrix.

    The returned factor C satisfies ``dm = C @ C.T``. Tagged or explicitly
    supplied orbitals are used when available; otherwise, a generalized
    eigendecomposition in the AO metric is performed.

    Args:
        dm (numpy.ndarray):
            AO density matrix.
        s1e (numpy.ndarray):
            AO overlap matrix. It may be None when orbital information is
            supplied.
        mo_coeff (numpy.ndarray):
            Optional AO-to-MO coefficients.
        mo_occ (numpy.ndarray):
            Optional nonnegative MO occupations.

    Returns:
        numpy.ndarray:
            Density factor with shape ``(nao, rank)``.
    '''
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

    s1e = numpy.asarray(s1e)
    if s1e.shape != dm.shape:
        raise ValueError('s1e has incompatible shape')
    sdm = lib.dot(s1e, lib.dot(dm, s1e))
    sdm = (sdm + sdm.T) * .5
    occ, coeff = scipy.linalg.eigh(sdm, s1e)
    if occ[0] < -1e-10:
        raise ValueError('density matrix is not positive semidefinite')
    mask = occ > 1e-12
    return coeff[:,mask] * numpy.sqrt(occ[mask])


def _unpack_aopair(out, packed, layout, pair):
    '''Scatter one packed canonical atom-pair block into a dense tensor.'''
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

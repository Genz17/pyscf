#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: 
#         Gengzhi Yang <genzyang17@gmail.com>
#         Hong-Zhou Ye <hzyechem@gmail.com>
#


import numpy as np

from .aft import AFTDF
from .stc_helper import get_coulG
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df import aft_jk, ft_ao
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import (
    is_zero, group_by_conj_pairs, kk_adapted_iter,
)
from pyscf.pbc.df.df_jk import (
    _format_dms, _ewald_exxdiv_for_G0,
)
from pyscf.pbc.df.aft_jk import (
    _update_vk_, _update_vk1_,
    _update_vk_dmf, _update_vk1_dmf,
    _mo_k2gamma, _gen_ft_kernel_fake_gamma,
    _update_vk_fake_gamma,
)
from pyscf.pbc.df.aft import _check_kpts


class AFTDF_STC(AFTDF):

    omega_dot_Rc = 4.
    Rc_type = 'ws'  # inradius of WS; alternative is 'sph'

    def dump_flags(self, verbose=None):
        AFTDF.dump_flags(self, verbose)

        log = logger.new_logger(self, verbose)
        if log.verbose < logger.INFO:
            return self
        log.info('omega_dot_Rc= %.15g', self.omega_dot_Rc)
        log.info('Rc_type= %s', self.Rc_type)
        return self

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        kpts, is_single_kpt = _check_kpts(self, kpts)

        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                vj, vk = rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k = False,
                                     omega=None, exxdiv=exxdiv)
                if with_k:
                    vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv, omega = omega)
                return vj, vk

        # if is_single_kpt:
        #     vj, vk = aft_jk.get_jk(self, dm, hermi, kpts[0], kpts_band,
        #                            with_j, with_k, exxdiv)
        # else:

        if True:
            vj = vk = None
            if with_k:
                vk = get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = aft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    def weighted_coulG(mydf, kpt=np.zeros(3), exx=False, mesh=None, omega_stc = None, omega=None):
        '''Weighted regular Coulomb kernel'''

        if not isinstance(exx, str) or exx.lower() not in ('vcut_ws', 'vcut_sph'):
            return super().weighted_coulG(kpt, exx, mesh, omega=omega)

        cell = mydf.cell
        if mesh is None:
            mesh = mydf.mesh
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        coulG = get_coulG(cell, kpt, exx, mydf, mesh, Gv, omega_stc=omega_stc, omega = omega)
        coulG *= kws
        return coulG


def ws_inradius(a, kmesh):
    """
    Wigner-Seitz inradius of the BvK superlattice.

    Parameters
    ----------
    a : (3, 3) array_like
        Primitive lattice vectors stored by rows.
    kmesh : (3,) array_like of int
        k-point mesh, e.g. (3, 3, 1).

    Returns
    -------
    Rin : float
        Inradius of the BvK Wigner-Seitz cell, in the same
        length unit as `a`.
    """
    from itertools import product

    a = np.asarray(a, dtype=float)
    kmesh = np.asarray(kmesh, dtype=int)

    # BvK lattice vectors, stored by rows
    A = kmesh[:, None] * a

    # Metric in lattice-coordinate space:
    # |m @ A|^2 = m @ G @ m
    G = A @ A.T

    # The shortest lattice vector cannot be longer than
    # the shortest generating vector.
    best2 = np.min(np.diag(G))

    # If lambda_min is the smallest eigenvalue of G,
    # m @ G @ m >= lambda_min * |m|^2.
    # Therefore any vector shorter than our current upper
    # bound must satisfy |m| <= sqrt(best2/lambda_min).
    lam_min = np.linalg.eigvalsh(G)[0]
    mmax = int(np.ceil(np.sqrt(best2 / lam_min)))

    for m in product(range(-mmax, mmax + 1), repeat=3):
        if m == (0, 0, 0):
            continue

        m = np.asarray(m)
        r2 = m @ G @ m

        if r2 < best2:
            best2 = r2

    return 0.5 * np.sqrt(best2)


def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None, omega = None):
    if kpts_band is not None:
        return get_k_for_bands(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)

    cpu0 = cpu1 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mydf)
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    mo_coeff = getattr(dm_kpts, 'mo_coeff', None)
    mo_occ = getattr(dm_kpts, 'mo_occ', None)
    dm_kpts = np.asarray(dm_kpts)

    dms = _format_dms(dm_kpts, kpts)
    n_dm, nkpts, nao = dms.shape[:3]
    vkR = np.zeros((n_dm,nkpts,nao,nao))
    vkI = np.zeros((n_dm,nkpts,nao,nao))
    vk = [vkR, vkI]
    weight = 1. / nkpts

    if mydf.Rc_type.lower() == 'sph':
        Rc = (3*nkpts*cell.vol/(4*np.pi))**(1./3)
    elif mydf.Rc_type.lower() == 'ws':
        from pyscf.pbc.lo.base import get_kmesh
        kmesh = get_kmesh(cell, kpts)
        log.warn('Using kmesh= %s to calculate WS-inradius Rc', kmesh)
        Rc = ws_inradius(cell.lattice_vectors(), kmesh)
    else:
        raise NotImplementedError

    aosym = 's1'
    kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
    rcut = ft_ao.estimate_rcut(cell)
    supmol = ft_ao.ExtendedMole.from_cell(cell, kmesh, rcut.max())
    supmol = supmol.strip_basis(rcut)

    t_rev_pairs = group_by_conj_pairs(cell, kpts, return_kpts_pairs=False)
    try:
        t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
    except TypeError:
        t_rev_pairs = [[k, k] if k_conj is None else [k, k_conj]
                       for k, k_conj in t_rev_pairs]
        t_rev_pairs = np.asarray(t_rev_pairs, dtype=np.int32, order='F')
    log.debug1('Num time-reversal pairs %d', len(t_rev_pairs))

    time_reversal_symmetry = mydf.time_reversal_symmetry
    if time_reversal_symmetry:
        for k, k_conj in t_rev_pairs:
            if k != k_conj and abs(dms[:,k_conj] - dms[:,k].conj()).max() > 1e-6:
                time_reversal_symmetry = False
                log.debug2('Disable time_reversal_symmetry')
                break

    if time_reversal_symmetry:
        k_to_compute = np.zeros(nkpts, dtype=np.int8)
        k_to_compute[t_rev_pairs[:,0]] = 1
    else:
        k_to_compute = np.ones(nkpts, dtype=np.int8)

    contract_mo_early = False
    if mo_coeff is None:
        dmsR = np.asarray(dms.real, order='C')
        dmsI = np.asarray(dms.imag, order='C')
        dm = [dmsR, dmsI]
        dm_factor = None

        if np.count_nonzero(k_to_compute) >= 2 * lib.num_threads():
            update_vk = _update_vk1_
        else:
            update_vk = _update_vk_
        log.debug2('set update_vk to %s', update_vk)
    else:
        # dm ~= dm_factor * dm_factor.T
        n_dm, nkpts, nao = dms.shape[:3]
        # mo_coeff, mo_occ are not a list of aligned array if
        # remove_lin_dep was applied to scf object
        if dm_kpts.ndim == 4:  # KUHF
            nocc = max(max(np.count_nonzero(x > 0) for x in z) for z in mo_occ)
            dm_factor = [[x[:,:nocc] for x in mo] for mo in mo_coeff]
            occs = [[x[:nocc] for x in z] for z in mo_occ]
        else:  # KRHF
            nocc = max(np.count_nonzero(x > 0) for x in mo_occ)
            dm_factor = [[mo[:,:nocc] for mo in mo_coeff]]
            occs = [[x[:nocc] for x in mo_occ]]
        dm_factor = np.array(dm_factor, dtype=np.complex128, order='C')
        dm_factor *= np.sqrt(np.array(occs, dtype=np.double))[:,:,None]

        bvk_ncells, rs_nbas, nimgs = supmol.bas_mask.shape
        s_nao = supmol.nao
        contract_mo_early = (time_reversal_symmetry and
                             bvk_ncells*nao*4 > s_nao*nocc*n_dm)
        log.debug2('time_reversal_symmetry = %s bvk_ncells = %d '
                   's_nao = %d nocc = %d n_dm = %d',
                   time_reversal_symmetry, bvk_ncells, s_nao, nocc, n_dm)
        log.debug2('Use algorithm contract_mo_early = %s', contract_mo_early)
        if contract_mo_early:
            s_nao = supmol.nao
            moR, moI = _mo_k2gamma(supmol, dm_factor, kpts, t_rev_pairs)
            if abs(moI).max() < 1e-5:
                dm = [moR, None]
                dm_factor = moR = moI = None
                ft_kern = _gen_ft_kernel_fake_gamma(cell, supmol, aosym)
                update_vk = _update_vk_fake_gamma
            else:
                moR = moI = None
                contract_mo_early = False

        if not contract_mo_early:
            dm = [np.asarray(dm_factor.real, order='C'),
                  np.asarray(dm_factor.imag, order='C')]
            dm_factor = None
            if np.count_nonzero(k_to_compute) >= 2 * lib.num_threads():
                update_vk = _update_vk1_dmf
            else:
                update_vk = _update_vk_dmf
        log.debug2('set update_vk to %s with dm_factor', update_vk)

    if not contract_mo_early:
        ft_kern = supmol.gen_ft_kernel(aosym, return_complex=False,
                                       kpts=kpts, verbose=log)

    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    Gv = np.asarray(Gv, order='F')
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))
    log.debug1('max_memory = %d MB (%d in use)', max_memory+mem_now, mem_now)

    if contract_mo_early:
        Gblksize = max(24, int((max_memory*1e6/16-nkpts*nao**2*3)/
                               (nao*s_nao+nao*nkpts*nocc))//8*8)
        Gblksize = min(Gblksize, ngrids, 200000)
        log.debug1('Gblksize = %d', Gblksize)
        buf = np.empty(Gblksize*s_nao*nao*2)
    else:
        Gblksize = max(24, int(max_memory*1e6/16/nao**2/(nkpts+3))//8*8)
        Gblksize = min(Gblksize, ngrids, 200000)
        log.debug1('Gblksize = %d', Gblksize)
        buf = np.empty(nkpts*Gblksize*nao**2*2)

    omega_stc = mydf.omega_dot_Rc / Rc
    log.warn('omega_stc = %.10f', omega_stc)

    for group_id, (kpt, ki_idx, kj_idx, self_conj) \
            in enumerate(kk_adapted_iter(cell, kpts)):
        vkcoulG = mydf.weighted_coulG(kpt, exxdiv, mesh, omega_stc = omega_stc, omega = omega)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            log.debug3('update_vk [%s:%s]', p0, p1)
            Gpq = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, kpt, out=buf)
            update_vk(vk, Gpq, dm, vkcoulG[p0:p1] * weight, ki_idx, kj_idx,
                      not self_conj, k_to_compute, t_rev_pairs)
            Gpq = None
        cpu1 = log.timer_debug1(f'get_k_kpts group {group_id}', *cpu1)

    if is_zero(kpts) and not np.iscomplexobj(dm_kpts):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j

    # Add ewald_exxdiv contribution because G=0 was not included in the
    # non-uniform grids
    if exxdiv == 'ewald' and cell.low_dim_ft_type == 'inf_vacuum':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts)

    if time_reversal_symmetry:
        for k, k_conj in t_rev_pairs:
            if k != k_conj:
                vk_kpts[:,k_conj] = vk_kpts[:,k].conj()
    log.timer_debug1('get_k_kpts', *cpu0)
    return vk_kpts.reshape(dm_kpts.shape)


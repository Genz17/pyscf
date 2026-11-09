#!/usr/bin/env python

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df import aft, df, df_jk, ft_ao, incore, rsdf_builder, rsdf_helper
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.lib.kpts_helper import is_zero, member
import ctypes

libpbc = lib.load_library('libpbc')


def _pair_list(cell, kmesh, threshold):
    '''Screen shell products over BvK images, retaining entire AO shell pairs.'''
    nk = int(np.prod(kmesh))
    if threshold == 0:
        mask = np.ones((nk, cell.nbas, cell.nbas), dtype=bool)
    else:
        rs_cell = ft_ao._RangeSeparatedCell.from_cell(cell)
        rcut = ft_ao.estimate_rcut(rs_cell, threshold)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut.max())
        supmol = supmol.strip_basis(rcut)
        mask = supmol.get_ovlp_mask(threshold)
        mask = lib.condense('np.any', mask, rs_cell.sh_loc, supmol.sh_loc)
        mask = mask.reshape(cell.nbas, nk, cell.nbas).transpose(1, 0, 2)
        # Include the translated reverse of every retained pair.
        negative = k2gamma.double_translation_indices(kmesh)[:, 0]
        mask = mask | (mask[negative].transpose(0, 2, 1))

    ao_loc = cell.ao_loc_nr()
    pairs, shells = [], []
    for m, ish, jsh in np.argwhere(mask):
        p0 = len(pairs)
        pairs.extend((i, j, m)
                     for i in range(ao_loc[ish], ao_loc[ish+1])
                     for j in range(ao_loc[jsh], ao_loc[jsh+1]))
        shells.append((ish, jsh + m*cell.nbas, p0, len(pairs))) # (ish at 0, jsh at m, p0, p1), covered AO of paris[p0:p1]
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 3), shells


def _int3c_kernel(cell, auxcell, eta):
    '''Periodic SR integrals, including both AO image sums about each auxiliary.'''

    builder = incore.Int3cBuilder(cell, auxcell)
    builder.rs_cell = ft_ao._RangeSeparatedCell.from_cell(cell)
    builder.bvk_kmesh = np.ones(3, dtype=int)
    rcut = rsdf_builder.estimate_rcut(builder.rs_cell, auxcell, eta)
    builder.supmol = ft_ao.ExtendedMole.from_cell(builder.rs_cell, builder.bvk_kmesh, rcut.max()).strip_basis(rcut)

    builder.supmol.omega = -eta
    return builder.gen_int3c_kernel(aosym='s1', return_complex=True)


def _ft_kernel(cell):
    rs_cell = ft_ao._RangeSeparatedCell.from_cell(cell)
    rcut = ft_ao.estimate_rcut(rs_cell)
    supmol = ft_ao.ExtendedMole.from_cell(rs_cell, [1, 1, 1], rcut.max())
    supmol = supmol.strip_basis(rcut)
    return supmol.gen_ft_kernel(aosym='s1', return_complex=True, kpts=np.zeros((1, 3)))


def _metric(mydf, scell, eta):
    '''Full metric and weighted auxiliary FTs, in canonical FFT momentum order.'''

    cell, auxcell = mydf.cell, mydf.auxcell
    kmesh = mydf.kmesh
    qpts = cell.make_kpts(kmesh, wrap_around=True, with_gamma_point=True)

    naux = auxcell.nao_nr()
    precision = min(cell.precision*.01, 1e-12)
    charge = ft_ao.ft_ao(auxcell, np.zeros((1, 3)))[0].real # Fourier transformation of the AOs at G = 0
    if cell.omega <= 0:
        metric = np.asarray(rsdf_helper.intor_j2c(auxcell, eta, precision=precision, kpts=qpts), dtype=complex)

        # SR has a finite zero mode pi/eta**2. Remove it as well as the LR one.
        metric[0] -= np.pi / (eta**2 * cell.vol) * np.outer(charge, charge)
    else:
        metric = np.zeros((len(qpts), naux, naux), dtype=complex)

    reciprocal = [None] * len(qpts)
    if cell.omega >= 0:
        mesh = mydf.mesh
        if mesh is None:
            mesh = rsdf_helper.estimate_mesh_for_omega(auxcell, eta, precision=precision)[1]

        # An odd supercell grid includes both G and -G, also at Nyquist q.
        mesh = np.asarray(mesh, dtype=int) * kmesh
        mesh = mesh // 2 * 2 + 1
        Gv = scell.get_Gv(mesh)
        indices = np.rint(Gv.dot(scell.lattice_vectors().T)/(2*np.pi)).astype(int)
        qindex = np.ravel_multi_index((indices % kmesh).T, kmesh)
        for q in range(len(qpts)):
            gv = Gv[qindex == q]
            g2 = np.einsum('gi,gi->g', gv, gv)
            weight = np.zeros(len(gv))
            nz = g2 > 1e-16
            # The auxiliary-cell Fourier sum cancels Nk in the supercell volume.
            weight[nz] = 4*np.pi/g2[nz] * np.exp(-g2[nz]/(4*eta**2)) / cell.vol
            auxG = ft_ao.ft_ao(auxcell, gv)
            weighted = auxG.conj().T * weight
            metric[q] += weighted.dot(auxG)
            reciprocal[q] = (gv, weighted)

    cholesky = []
    for q, v in enumerate(metric):
        try:
            cholesky.append(scipy.linalg.cholesky(v, lower=True))
        except np.linalg.LinAlgError as err:
            raise np.linalg.LinAlgError(
                'Global DF metric is not positive definite at q=%s; '
                'use a linearly independent auxiliary basis' % qpts[q]) from err
    return cholesky, reciprocal, charge


def _make_cderi(mydf, shells):
    cell, auxcell = mydf.cell, mydf.auxcell
    nk = len(mydf.kpts)
    naux = auxcell.nao_nr()

    eta = abs(cell.omega) if cell.omega else mydf.eta
    
    scell = tools.super_cell(cell, mydf.kmesh)
    sauxcell = tools.super_cell(auxcell, mydf.kmesh)

    chol, reciprocal, charge = _metric(mydf, scell, eta)
    int3c = _int3c_kernel(scell, sauxcell, eta) if cell.omega <= 0 else None
    ft = _ft_kernel(scell)
    factors = np.empty((nk, naux, len(mydf.pairs)), dtype=complex)
    axes = (0, 1, 2)
    for ish, jsh, p0, p1 in shells:
        shls = (ish, ish+1, jsh, jsh+1)
        shape = tuple(mydf.kmesh) + (naux, p1-p0)
        if int3c is None:
            a = np.zeros((nk, naux, p1-p0), dtype=complex)
        else:
            # The integral kernel returns [pair, auxiliary cell * auxiliary AO].
            a = int3c(shls + (0, sauxcell.nbas))[0].T.copy()
            a = np.fft.fftn(a.reshape(shape), axes=axes).reshape(nk, naux, -1)
            overlap = ft(np.zeros((1, 3)), shls_slice=shls).reshape(-1).real
            a[0] -= (np.pi / (eta**2 * cell.vol) * charge[:, None] * overlap)

        for q in range(nk):
            if reciprocal[q] is not None:
                gv, weighted = reciprocal[q]
                # Only a retained shell pair is transformed; its BvK images
                # are summed analytically by the periodic AO-pair FT kernel.
                rho = ft(gv, shls_slice=shls).reshape(len(gv), p1-p0)
                a[q] += weighted.dot(rho)
            a[q] = scipy.linalg.solve_triangular(chol[q], a[q], lower=True)
        factors[:, :, p0:p1] = np.fft.ifftn(a.reshape(shape), axes=axes).reshape(nk, naux, p1-p0)

    return np.asarray(np.real_if_close(factors), order='C')


def _get_j(mydf, dm):
    '''Coulomb contraction in cell space; dm[d,i,j] uses exp(-ik.d).'''
    mu, nu, m = mydf.pairs.T
    negative = mydf._diff[:, 0]
    b = mydf._cderi.sum(axis=0)  # The charge density has q=0.
    rho = np.einsum('ap,p->a', b, dm[negative[m], nu, mu])
    vj = np.zeros_like(dm)
    vj[m, mu, nu] = np.einsum('ap,a->p', b.conj(), rho)
    return vj

def _get_k(mydf, dm):
    """Exchange contraction through the C kernel; dm is in cell space."""

    # Match the C kernel's element types and memory layout.
    b = np.ascontiguousarray(mydf._cderi, dtype=np.complex128)
    dm = np.ascontiguousarray(dm, dtype=np.complex128)
    pairs = np.ascontiguousarray(mydf.pairs, dtype=np.intc)
    diff = np.ascontiguousarray(mydf._diff, dtype=np.intc)

    nk, naux, npair = b.shape
    nao = dm.shape[1]
    vk = np.empty((nk, nao, nao), dtype=np.complex128)

    if nk == 0 or nao == 0 or naux == 0 or npair == 0:
        vk.fill(0)
        return vk

    blockdim = int(mydf.blockdim)
    if blockdim < 1:
        raise ValueError('blockdim must be positive')

    nthreads = max(1, lib.num_threads())
    memory = max(1.0, mydf.max_memory - lib.current_memory()[0]) * 1e6

    # Actual scratch uses 16 bytes per complex element.
    # Use 32 here to leave some memory headroom.
    bytes_per_aux = 32 * nthreads * nk * nao
    blksize = max(1,min(nk * naux, blockdim, int(memory / bytes_per_aux)))

    libpbc._get_k(
        vk.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        dm.ctypes.data_as(ctypes.c_void_p),
        pairs.ctypes.data_as(ctypes.c_void_p),
        diff.ctypes.data_as(ctypes.c_void_p),
        nk,
        naux,
        nao,
        npair,
        blksize,
    )

    return vk


class GDF(df.GDF):
    '''Global DF with in-memory, translation-covariant factors.
        Assumes one density matrix per k-point (restricted HF).
        ``kmesh`` gives the three mesh dimensions (default 1, 1, 1).
        The unshifted k-points are generated with ``cell.make_kpts``.
        ``eta`` is the numerical Coulomb splitting parameter (default 0.4).
        ``pair_threshold`` controls the periodic shell-overlap estimate; zero
        retains all pairs. ``mesh`` sets the primitive-cell LR reciprocal cutoff.
        ``blockdim`` bounds the auxiliary batch size in the exchange build.
    '''
    _keys = {'pair_threshold', 'kmesh', 'pairs'}

    def __init__(self, cell, kmesh=(1, 1, 1)):
        self.kmesh = np.array(kmesh, dtype=int)
        if (self.kmesh.shape != (3,) or np.any(self.kmesh <= 0) or
            not np.array_equal(self.kmesh, kmesh)):
            raise ValueError('kmesh must contain three positive integers')
        df.GDF.__init__(self, cell, cell.make_kpts(self.kmesh, with_gamma_point=True))
        self.eta = .4
        self.pair_threshold = cell.precision
        self.pairs = None
        self._diff = None
        self._translations = None

    def reset(self, cell=None):
        df.GDF.reset(self, cell)
        self.kpts = self.cell.make_kpts(self.kmesh, with_gamma_point=True)
        self.pairs = self._diff = self._translations = None # diff[s,t] = t - s
        return self

    def has_kpts(self, kpts):
        if kpts is None:
            return True
        return all(len(member(k, self.kpts)) > 0
                   for k in np.asarray(kpts).reshape(-1, 3))

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n******** %s ********', self.__class__)
        log.info('auxbasis = %s', self.auxbasis)
        log.info('eta = %s, mesh = %s', self.eta, self.mesh)
        log.info('pair_threshold = %g', self.pair_threshold)
        log.info('kmesh = %s, len(kpts) = %d', self.kmesh, len(self.kpts))
        return self

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        if self.cell.dimension != 3:
            raise NotImplementedError('Global DF currently requires dimension=3')
        if self.pair_threshold < 0 or self.eta <= 0:
            raise ValueError('pair_threshold must be nonnegative and eta positive')
        self.kpts = self.cell.make_kpts(self.kmesh, with_gamma_point=True)
        if not self.has_kpts(kpts_band) or not self.has_kpts(self.kpts_band):
            raise NotImplementedError('Band k-points must belong to the fitted mesh')

        self._diff = k2gamma.double_translation_indices(self.kmesh)
        self._translations = k2gamma.translation_vectors_for_kmesh(self.cell, self.kmesh)

        self.dump_flags()

        self._cderi = None
        self.auxcell = df.make_auxcell(self.cell, self.auxbasis)

        if with_j3c:
            cpu0 = logger.process_clock(), logger.perf_counter()
            self.pairs, shells = _pair_list(self.cell, self.kmesh, self.pair_threshold)

            logger.info(self, 'Retain %d / %d AO pairs', len(self.pairs), len(self.kpts)*self.cell.nao_nr()**2)

            self._cderi = _make_cderi(self, shells)
            logger.timer(self, 'global DF build', *cpu0)
        return self

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None and omega != 0:
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     exxdiv=exxdiv)
        if exxdiv not in (None, 'ewald'):
            raise NotImplementedError('Global DF supports exxdiv=None or "ewald"')
        if self._cderi is None:
            self.build()
        kpts, single_kpt = aft._check_kpts(self, kpts)
        if (len(kpts) != len(self.kpts) or
            any(len(member(k, kpts)) != 1 for k in self.kpts)):
            raise ValueError('Density k-points must cover the fitted mesh')
        if kpts_band is not None and not self.has_kpts(kpts_band):
            raise NotImplementedError('Band k-points must belong to the fitted mesh')

        dm = np.asarray(dm)
        nao = dm.shape[-1]
        dms = dm.reshape(len(kpts), nao, nao)
        phase = np.exp(1j*kpts.dot(self._translations.T))
        dm_r = np.einsum('kd,kij->dij', phase.conj(), dms) / len(kpts)
        # Match the single-k-point DF API when the band is the same point.
        single_kpt = single_kpt and (kpts_band is None or is_zero(kpts_band-kpts[0]))
        bands = kpts if single_kpt else df_jk._format_kpts_band(kpts_band, kpts)
        phase_out = np.exp(1j*bands.dot(self._translations.T))

        def to_kspace(v):
            v = np.einsum('kd,dij->kij', phase_out, v)
            if hermi == 1:
                v = (v + v.conj().transpose(0, 2, 1)) * .5
            if is_zero(kpts) and is_zero(bands) and not np.iscomplexobj(dm):
                v = v.real
            return v

        vj = vk = None
        if with_j:
            vj = to_kspace(_get_j(self, dm_r))
            vj = (vj.reshape(dm.shape) if single_kpt else
                  df_jk._format_jks(vj[None], dm, kpts_band, kpts))
        if with_k:
            vk = to_kspace(_get_k(self, dm_r))
            if exxdiv == 'ewald':
                df_jk._ewald_exxdiv_for_G0(self.cell, kpts, dms[None], vk[None], bands)
            vk = (vk.reshape(dm.shape) if single_kpt else
                  df_jk._format_jks(vk[None], dm, kpts_band, kpts))
        return vj, vk

    # Use the AFT context manager, which supports both signs of omega.
    range_coulomb = aft.AFTDFMixin.range_coulomb

    def get_naoaux(self):
        if self._cderi is None:
            self.build()
        return self._cderi.shape[1]

    def cderi_array(self, label=None):
        '''Return B[t,Q,p]; unlike conventional GDF this is an in-memory array.'''
        if self._cderi is None:
            self.build()
        return self._cderi

    def sr_loop(self, kpti_kptj=np.zeros((2, 3)), max_memory=2000,
                compact=True, blksize=None, aux_slice=None):
        '''Expose conventional momentum factors for the existing AO2MO APIs.'''
        if not self.has_kpts(kpti_kptj):
            raise NotImplementedError('k-points must belong to the fitted mesh')
        if self._cderi is None:
            self.build()
        ki, kj = np.asarray(kpti_kptj)
        m = self.pairs[:, 2]
        phase_q = np.exp(-1j*self._translations.dot(kj-ki))
        phase_j = np.exp(1j*self._translations.dot(kj))[m]
        nao = self.cell.nao_nr()
        if aux_slice is None:
            aux_slice = (0, self.get_naoaux())
        if blksize is None:
            blksize = self.blockdim
        for a0, a1 in lib.prange(*aux_slice, blksize):
            bp = np.einsum('t,tap,p->ap', phase_q,
                           self._cderi[:, a0:a1], phase_j)
            b = np.zeros((a1-a0, nao, nao), dtype=complex)
            for p, (i, j, _) in enumerate(self.pairs):
                b[:, i, j] += bp[:, p]
            if compact and is_zero(ki-kj):
                b = lib.pack_tril(b)
            else:
                b = b.reshape(a1-a0, nao**2)
            yield np.asarray(b.real, order='C'), np.asarray(b.imag, order='C'), 1


DF = GDF

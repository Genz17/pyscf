#!/usr/bin/env python

r'''Literal real-space PARI equations for periodic short-range exchange.

For P[k] = M[k].conj() @ M[k].T, this module explicitly constructs

    D[mu,Q_U,i,k] = sum_(lambda,T1) C[mu_0,lambda_T1,Q_U]
                                    M[lambda,i,k] exp(-i k.T1)
    H[nu_T,Q_U,i,k] = sum_(sigma,T2) M[sigma,i,k] exp(-i k.T2)
        [(Q_U|sigma_T2 nu_T) - 1/2 sum_R C[sigma_T2,nu_T,R] (Q_U|R)]
    L[0,T] = sum_(k,Q_U,i) D[mu,Q_U,i,k].conj() H[nu_T,Q_U,i,k] / Nk
    K[0,T] = L[0,T] + L[0,-T].conj().T
    K[kout] = sum_T exp(i kout.T) K[0,T].

Every integral uses erfc(omega*r)/r, including its finite G=0 contribution.
Auxiliary images remain distinct.  Translation covariance reuses H blocks:
H[nu_T,Q_U,k] = exp(-i k.U) H[nu_(T-U),Q_0,k].

Use get_k_from_m(M), get_k_from_mo(mo_coeff, mo_occ), or get_k(dm).
The density route requires Hermitian PSD matrices.  K excludes the exchange
minus sign and RHF factor of 1/2.  Independent pair_rcut, integral_rcut, and
metric_rcut control finite lattice sums; None estimates each radius.
See pari.md for the derivation, input conventions, and reference-code limits.
'''

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from pyscf import gto, lib
from pyscf.df import incore as mol_incore
from pyscf.lib import logger
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import incore, rsdf_builder
from pyscf.pbc.tools import pbc as pbctools


def _as_kpts(kpts):
    return np.zeros((1, 3)) if kpts is None else np.asarray(kpts, dtype=float).reshape(-1, 3)


def _kpoint_blocks(values, ndim):
    '''Accept one array or a list of arrays with different occupied ranks.'''
    if len(values) == 0 or np.ndim(values[0]) == ndim - 1:
        values = (values,)
    return tuple(np.asarray(x) for x in values)


def _atom_mol(cell, atom, shift):
    '''Extract an atom without renormalizing or reordering its basis.'''
    mol = gto.Mole()
    mol.cart = cell.cart
    mol.stdout, mol.verbose = cell.stdout, cell.verbose
    mol.max_memory = cell.max_memory
    mol._atm = cell._atm[atom:atom+1].copy()
    mol._bas = cell._bas[cell._bas[:, gto.ATOM_OF] == atom].copy()
    mol._bas[:, gto.ATOM_OF] = 0
    mol._env = cell._env.copy()
    ptr = mol._atm[0, gto.PTR_COORD]
    mol._env[ptr:ptr+3] += shift
    mol._atom = [(cell.atom_symbol(atom), mol._env[ptr:ptr+3].copy())]
    mol._built = True
    return mol


def _fit(metric, rhs, threshold):
    '''Solve a local fit in a diagonally normalized, rank-revealing metric.'''
    scale = np.sqrt(metric.diagonal())
    normalized = metric / scale[:, None] / scale[None, :]
    w, u = linalg.eigh(normalized)
    if w[0] < -max(threshold, 1e-12) * w[-1]:
        raise ValueError('The short-range auxiliary metric is not positive semidefinite')
    keep = w > threshold * w[-1]
    # rhs is (nao_A, nao_B, naux); do not assume contiguous AO shell sizes.
    shape = rhs.shape
    b = rhs.reshape(-1, shape[-1]).T / scale[:, None]
    u = u[:, keep]
    projected = np.einsum('qi,qj->ij', u, b) / w[keep, None]
    c = np.einsum('qi,ij->qj', u, projected) / scale[:, None]
    return c.T.reshape(shape)


@dataclass
class _Pair:
    a: int
    b: int
    translation: np.ndarray
    ca: np.ndarray
    cb: object  # None for a pair on one atom in one cell (a set union).
    image: tuple


def _images(cell, radius):
    translations = cell.get_lattice_Ls(rcut=radius, dimension=cell.dimension)
    indices = np.rint(translations @ np.linalg.inv(cell.lattice_vectors())).astype(int)
    indices = {tuple(x) for x in indices}
    indices |= {tuple(-np.asarray(x)) for x in indices}
    indices.add((0, 0, 0))
    lattice = cell.lattice_vectors()
    indices = sorted(indices, key=lambda x: (np.linalg.norm(np.dot(x, lattice)), x))
    return indices, np.asarray(indices).dot(lattice)


@dataclass
class PARIIntermediates:
    '''Literal D and H intermediates for one set of orbital factors M.

    d[U][k][mu,Q,i] is D_i^{mu_0 Q_U,k}.  h0[T][k][nu,Q,i] is
    H_i^{nu_T Q_0,k}.  Q runs over the whole reference-cell auxiliary basis;
    U and T are integer lattice triples, not Cartesian vectors.  h(T,U)
    returns the actual H_i^{nu_T Q_U,k} without identifying different images.
    '''
    kpts: np.ndarray
    lattice: np.ndarray
    m: tuple
    naux: int
    d: dict
    h0: dict
    token: object

    def h(self, translation, aux_image=(0, 0, 0)):
        u = np.asarray(aux_image)
        relative = tuple(np.asarray(translation) - u)
        value = self.h0.get(relative)
        if value is None:
            return tuple(np.zeros((x.shape[0], self.naux, x.shape[1]), complex)
                         for x in self.m)
        phases = np.exp(-1j * self.kpts.dot(np.dot(u, self.lattice)))
        return tuple(phase * block for phase, block in zip(phases, value))


class PARI(lib.StreamObject):
    r'''Robust short-range periodic PARI exchange.

    Args:
        cell: Built periodic Cell.
        kpts: Full, equally weighted list of k-points, in inverse Bohr.
        auxbasis: Auxiliary basis accepted by ``incore.make_auxcell``.
        omega: Positive erfc range parameter.  If omitted, infer it only from
            a negative ``cell.omega`` (PySCF's short-range convention).

    Attributes:
        precision: Integral lattice-sum precision; defaults to cell.precision.
        pair_rcut: AO-pair separation cutoff in Bohr; None estimates a cutoff.
        integral_rcut: Independent lattice radius for bare three-center
            integrals; None estimates a short-range three-center cutoff.
        metric_rcut: Independent radius for auxiliary metric lattice sums.
        linear_dep_threshold: Relative eigenvalue cutoff of the diagonally
            normalized local metric; defaults to 1e-12.

    Call ``reset(cell)`` after changing the cell geometry or orbital basis.
    Changing omega, auxbasis, precision, any radius, or linear_dep_threshold
    automatically invalidates the fitted coefficients at the next call.
    '''

    _keys = {'cell', 'kpts', 'auxbasis', 'omega', 'precision', 'pair_rcut',
             'linear_dep_threshold', 'auxcell', 'max_memory', 'verbose',
             'stdout', 'rcut', 'translations', 'integral_rcut', 'metric_rcut',
             'integral_radius', 'metric_radius'}

    def __init__(self, cell, kpts=None, auxbasis=None, omega=None):
        self.cell = cell
        self.kpts = _as_kpts(kpts)
        self.auxbasis = auxbasis
        if omega is None and cell.omega < 0:
            omega = -cell.omega
        self.omega = omega
        self.precision = cell.precision
        self.pair_rcut = None
        self.integral_rcut = None
        self.metric_rcut = None
        self.linear_dep_threshold = 1e-12
        self.stdout, self.verbose = cell.stdout, cell.verbose
        self.max_memory = cell.max_memory
        self.reset()

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.auxcell = None
        self.rcut = None
        self.translations = None
        self._pairs = []
        self._signature = None
        self._build_token = object()
        self._metric_cache = None
        self._bare_mol = None
        self.integral_radius = self.metric_radius = None
        return self

    def _settings(self):
        return (id(self.cell), repr(self.auxbasis), self.omega, self.precision,
                self.pair_rcut, self.integral_rcut, self.metric_rcut,
                self.linear_dep_threshold)

    def _ensure_built(self):
        if self._signature != self._settings():
            self.build()

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('short-range omega = %s; auxbasis = %s', self.omega, self.auxbasis)
        log.info('precision = %g; pair_rcut = %s; linear_dep_threshold = %g',
                 self.precision, self.pair_rcut, self.linear_dep_threshold)
        log.info('integral_rcut = %s; metric_rcut = %s',
                 self.integral_rcut, self.metric_rcut)
        log.info('Literal erfc kernel, including its finite G=0 contribution')
        return self

    def build(self):
        self.reset()
        cell = self.cell
        if self.omega is None or not np.isfinite(self.omega) or self.omega <= 0:
            raise ValueError('omega must be finite and positive for erfc(omega*r)/r')
        if not 0 < self.linear_dep_threshold < 1:
            raise ValueError('linear_dep_threshold must lie between zero and one')
        self.dump_flags()
        t0 = (logger.process_clock(), logger.perf_counter())
        auxcell = incore.make_auxcell(cell, self.auxbasis)
        self.auxcell = auxcell
        self._ao_slices = cell.aoslice_by_atom()
        self._aux_slices = auxcell.aoslice_by_atom()

        if self.pair_rcut is not None:
            rcut = float(self.pair_rcut)
        elif cell.dimension == 0:
            rcut = np.inf  # No atom-pair truncation for the molecular limit.
        else:
            precision = self.precision * 1e-2
            rcut = max(pbcgto.cell.estimate_rcut(cell, precision),
                       rsdf_builder.estimate_ft_rcut(cell, precision).max())
        self.rcut = rcut
        indices, self.translations = _images(cell, rcut)
        atoms = [_atom_mol(cell, a, np.zeros(3)) for a in range(cell.natm)]
        auxatoms = [_atom_mol(auxcell, a, np.zeros(3)) for a in range(cell.natm)]
        coords = cell.atom_coords()

        for index, translation in zip(indices, self.translations):
            opposite = tuple(-n for n in index)
            for a, mola in enumerate(atoms):
                if mola.nbas == 0:
                    continue
                for b, molb0 in enumerate(atoms):
                    if molb0.nbas == 0:
                        continue
                    if (a, b, *index) > (b, a, *opposite):
                        continue
                    if np.linalg.norm(coords[b] + translation - coords[a]) > rcut:
                        continue
                    same = a == b and index == (0, 0, 0)
                    if same:
                        mol, auxmol = mola, auxatoms[a]
                        shls = (0, mol.nbas, 0, mol.nbas, 0, auxmol.nbas)
                    else:
                        molb = _atom_mol(cell, b, translation)
                        auxb = _atom_mol(auxcell, b, translation)
                        mol = gto.conc_mol(mola, molb)
                        auxmol = gto.conc_mol(auxatoms[a], auxb)
                        shls = (0, mola.nbas, mola.nbas, mol.nbas, 0, auxmol.nbas)
                    with auxmol.with_range_coulomb(-self.omega):
                        metric = auxmol.intor('int2c2e', hermi=1)
                    with mol.with_range_coulomb(-self.omega):
                        rhs = mol_incore.aux_e2(mol, auxmol, aosym='s1', shls_slice=shls)
                    c = _fit(metric, rhs, self.linear_dep_threshold)
                    na = auxatoms[a].nao_nr()
                    ca = c[:, :, :na].copy()
                    cb = None if same else c[:, :, na:].copy()
                    self._pairs.append(_Pair(a, b, translation.copy(), ca, cb,
                                             index))
                    if not same:
                        # C^{b0,a(-T)} follows by translation and AO exchange.
                        self._pairs.append(_Pair(b, a, -translation.copy(),
                                                 cb.transpose(1, 0, 2),
                                                 ca.transpose(1, 0, 2), opposite))
        if cell.dimension == 0:
            self.integral_radius = self.metric_radius = 0.
        else:
            precision = self.precision * 1e-2
            self.integral_radius = self.integral_rcut
            if self.integral_radius is None:
                self.integral_radius = max(
                    rsdf_builder.estimate_rcut(cell, auxcell, self.omega, precision).max(),
                    pbcgto.cell.estimate_rcut(cell, precision))
            self.metric_radius = self.metric_rcut
            if self.metric_radius is None:
                self.metric_radius = rsdf_builder.estimate_rs_2c2e_rcut(
                    auxcell, self.omega, precision)
        self._bare_indices, self._bare_translations = _images(cell, self.integral_radius)
        self._metric_indices, self._metric_translations = _images(cell, self.metric_radius)
        # A molecular cluster is used only for bare integrals.  Its lattice
        # radius is independent of the support of the fitted coefficients.
        self._bare_mol = pbctools._build_supcell_(
            cell.to_mol().copy(), cell, self._bare_translations)
        self._signature = self._settings()
        logger.info(self, 'PARI: %d directed atom/image pairs, rcut = %.8g Bohr',
                    len(self._pairs), self.rcut)
        logger.info(self, 'Bare-integral images = %d; metric images = %d',
                    len(self._bare_indices), len(self._metric_indices))
        logger.timer(self, 'PARI local fits', *t0)
        return self

    def _format_m(self, m, kpts):
        '''M follows the user's convention P = M.conj() @ M.T.'''
        blocks = _kpoint_blocks(m, 2)
        if len(blocks) != len(kpts):
            raise ValueError('There must be one orbital factor M per input k-point')
        return tuple(np.array(block, dtype=complex, copy=True) for block in blocks)

    def _empty_intermediate(self, m):
        return tuple(np.zeros((self.cell.nao_nr(), self.auxcell.nao_nr(),
                               block.shape[1]), dtype=complex) for block in m)

    def _build_d(self, m, kpts):
        '''For each pair (A_0,B_T1), accumulate U=0 first, then U=T1.

        Both contributions carry the partner-AO phase exp(-i k.T1).
        '''
        zero = (0, 0, 0)
        d = {zero: self._empty_intermediate(m)}
        for pair in self._pairs:
            a0, a1 = self._ao_slices[pair.a, 2:]
            b0, b1 = self._ao_slices[pair.b, 2:]
            qa0, qa1 = self._aux_slices[pair.a, 2:]
            qb0, qb1 = self._aux_slices[pair.b, 2:]
            phases = np.exp(-1j * kpts.dot(pair.translation))
            contributions = (
                (zero, qa0, qa1, pair.ca),       # U=0: auxiliary functions on A_0
                (pair.image, qb0, qb1, pair.cb), # U=T1: auxiliary functions on B_T1
            )
            for u, q0, q1, coeff in contributions:
                if coeff is None:  # A=B and T1=0: the auxiliary union occurs once.
                    continue
                if u not in d:
                    d[u] = self._empty_intermediate(m)
                for k, (mk, phase) in enumerate(zip(m, phases)):
                    d[u][k][a0:a1, q0:q1] += phase * np.einsum(
                        'mlq,li->mqi', coeff, mk[b0:b1])
        return d

    def _metric_blocks(self):
        '''Actual-image integrals V[V][Q,R] = (Q_0|R_V)_SR.'''
        if self._metric_cache is not None:
            return self._metric_cache
        auxmol = self.auxcell.to_mol()
        blocks = {}
        for index, translation in zip(self._metric_indices, self._metric_translations):
            reverse = tuple(-n for n in index)
            if reverse in blocks:
                blocks[index] = blocks[reverse].T
                continue
            if index == (0, 0, 0):
                with auxmol.with_range_coulomb(-self.omega):
                    blocks[index] = auxmol.intor('int2c2e', hermi=1)
            else:
                shifted = pbctools._build_supcell_(
                    auxmol.copy(), self.auxcell, translation[None, :])
                pairmol = gto.conc_mol(auxmol, shifted)
                nbas = auxmol.nbas
                with pairmol.with_range_coulomb(-self.omega):
                    blocks[index] = pairmol.intor('int2c2e', shls_slice=(
                        0, nbas, nbas, 2 * nbas))
        self._metric_cache = blocks
        return blocks

    def _build_h0(self, m, kpts, d):
        '''H0[T] = H_i^{nu_T Q_0,k}, retaining the bare and fitted terms.'''
        if not any(block.shape[1] for block in m):
            return {}
        cell, mol = self.cell, self._bare_mol
        auxmol = self.auxcell.to_mol()
        h0 = {}
        # The source AO coefficient on sigma_S is M_sigma * exp(-i k.S).
        phases = np.exp(-1j * kpts.dot(self._bare_translations.T))
        source_m = tuple(np.einsum('t,si->tsi', phase, mk).reshape(
            len(self._bare_indices) * cell.nao_nr(), mk.shape[1])
            for phase, mk in zip(phases, m))
        # Stream one external atom/image at a time.  Crucially, this AO-image
        # cluster is independent of pair_rcut: C=0 does not imply (Q|pair)=0.
        with mol.with_range_coulomb(-self.omega):
            for it, index in enumerate(self._bare_indices):
                h0[index] = self._empty_intermediate(m)
                for a in range(cell.natm):
                    sh0, sh1, a0, a1 = self._ao_slices[a]
                    if sh0 == sh1:
                        continue
                    required = 8 * (a1-a0) * mol.nao_nr() * auxmol.nao_nr()
                    if required > self.max_memory * 1e6:
                        raise MemoryError('A bare PARI three-center block exceeds '
                                          'max_memory; increase max_memory or use a smaller system')
                    ints = mol_incore.aux_e2(
                        mol, auxmol, aosym='s1', shls_slice=(
                            it * cell.nbas + sh0, it * cell.nbas + sh1,
                            0, mol.nbas, 0, auxmol.nbas))
                    for k, mk in enumerate(source_m):
                        h0[index][k][a0:a1] += np.einsum('nsq,si->nqi', ints, mk)
        # For external nu_T, its translated fitted density is
        # exp(-i k.T) D_i^{nu_0 R_W,k} on R_(W+T).
        # Set V=W+T, so every metric image V and D image W contributes at T=V-W.
        lattice = cell.lattice_vectors()
        for v, metric in self._metric_blocks().items():
            if not np.any(metric):
                continue
            for w, dw in d.items():
                t = tuple(np.asarray(v) - w)
                if t not in h0:
                    h0[t] = self._empty_intermediate(m)
                phases = np.exp(-1j * kpts.dot(np.dot(t, lattice)))
                for k, (dk, phase) in enumerate(zip(dw, phases)):
                    h0[t][k][...] -= 0.5 * phase * np.einsum('qr,nri->nqi', metric, dk)
        return h0

    def get_intermediates(self, m, kpts=None):
        '''Construct the supplied equations' D and H, for explicit factors M.

        M[k] has shape (nao,nocc_k) and satisfies P[k]=M[k].conj() @ M[k].T.
        The returned PARIIntermediates exposes d[U][k] and h(T,U)[k].
        U,T are actual integer lattice images; Q runs over reference-cell
        auxiliary functions.  No occupied-orbital contraction is hidden in
        an AO density contraction or a Bloch three-center tensor.
        '''
        self._ensure_built()
        kpts = _as_kpts(self.kpts if kpts is None else kpts).copy()
        m = self._format_m(m, kpts)
        d = self._build_d(m, kpts)
        h0 = self._build_h0(m, kpts, d)
        return PARIIntermediates(kpts, self.cell.lattice_vectors(), m,
                                 self.auxcell.nao_nr(), d, h0, self._build_token)

    def get_l(self, m=None, kpts=None, intermediates=None):
        '''Return {T: L^{0T}} using exactly sum_(k,Q_U,i) D* H / Nk.

        Supply M, or reuse the result of get_intermediates via intermediates=.
        T is an integer lattice triple.  Every output T=U+t generated by
        the D[U] and anchored H0[t] supports is included.
        '''
        self._ensure_built()
        data = intermediates or self.get_intermediates(m, kpts)
        if data.token is not self._build_token:
            raise ValueError('Intermediates belong to a different PARI build')
        nao = self.cell.nao_nr()
        l_real = {}
        for u, du in data.d.items():
            phases = np.exp(-1j * data.kpts.dot(np.dot(u, data.lattice)))
            for t, ht in data.h0.items():
                output = tuple(np.asarray(u) + t)
                if output not in l_real:
                    l_real[output] = np.zeros((nao, nao), dtype=complex)
                for dk, hk, phase in zip(du, ht, phases):
                    # This is H_i^{nu_(U+t) Q_U,k}, with the actual Q image.
                    h_image = phase * hk
                    l_real[output] += np.einsum('mqi,nqi->mn', dk.conj(), h_image)
        if not l_real:
            l_real[(0, 0, 0)] = np.zeros((nao, nao), dtype=complex)
        return {t: value / len(data.kpts) for t, value in l_real.items()}

    @staticmethod
    def get_k_realspace(l_real):
        '''K^{0T}=L^{0T}+(L^{0,-T})^dagger on an inversion-closed image set.'''
        indices = set(l_real)
        indices |= {tuple(-np.asarray(t)) for t in indices}
        zero = np.zeros_like(next(iter(l_real.values())))
        return {t: l_real.get(t, zero) + l_real.get(
            tuple(-np.asarray(t)), zero).conj().T for t in sorted(indices)}

    def _fourier_k(self, k_real, bands):
        indices = list(k_real)
        translations = np.asarray(indices).dot(self.cell.lattice_vectors())
        phases = np.exp(1j * bands.dot(translations.T))
        return np.einsum('bt,tmn->bmn', phases, np.asarray(list(k_real.values())))

    def get_k_from_m(self, m, kpts=None, kpts_band=None):
        '''Exchange from the user's M factors, through D, H, L(0,T), K(0,T).

        Default output shape is (nk,nao,nao), including for a single k-point.
        An explicit kpts_band of shape (3,) returns a single (nao,nao) matrix.
        '''
        kpts = _as_kpts(self.kpts if kpts is None else kpts)
        bands = kpts if kpts_band is None else _as_kpts(kpts_band)
        l_real = self.get_l(m, kpts)
        vk = self._fourier_k(self.get_k_realspace(l_real), bands)
        if kpts_band is not None and np.asarray(kpts_band).ndim == 1:
            return vk[0]
        return vk

    def get_k_from_mo(self, mo_coeff, mo_occ, kpts=None, kpts_band=None):
        '''Use M[k]=mo_coeff[k].conj()*sqrt(mo_occ[k]), including fractions.'''
        kpts = _as_kpts(self.kpts if kpts is None else kpts)
        coeff = self._format_m(mo_coeff, kpts)
        occupations = _kpoint_blocks(mo_occ, 1)
        if len(occupations) != len(kpts):
            raise ValueError('There must be one occupation array per k-point')
        m = []
        for c, occ in zip(coeff, occupations):
            if np.iscomplexobj(occ) or np.any(occ < 0):
                raise ValueError('MO occupations must be real and nonnegative')
            keep = occ > 0
            m.append(c[:, keep].conj() * np.sqrt(occ[keep]))
        return self.get_k_from_m(m, kpts, kpts_band)

    def _factor_density(self, density):
        '''P=U w U^dagger -> M=U.conj()*sqrt(w), with no signed fallback.'''
        m = []
        for dm in density:
            if not np.allclose(dm, dm.conj().T, rtol=1e-10, atol=1e-12):
                raise ValueError('The supplied PARI equations require Hermitian densities')
            w, u = linalg.eigh((dm + dm.conj().T) * 0.5)
            tolerance = 100 * np.finfo(float).eps * dm.shape[0] * max(1., abs(w).max())
            if w[0] < -tolerance:
                raise ValueError('The factorization P=M* M^T requires a positive-semidefinite density')
            # Clamp only roundoff-scale negative eigenvalues, and retain all
            # positive occupations independently of the auxiliary-fit threshold.
            keep = w > 0
            m.append(u[:, keep].conj() * np.sqrt(w[keep]))
        return tuple(m)

    def get_k(self, dm_kpts, hermi=1, kpts=None, kpts_band=None, exxdiv=None):
        '''Exchange for Hermitian PSD PySCF densities through the MO equations.

        P[k]=C_occ[k] occ[k] C_occ[k]^dagger is factored into M* M^T.
        Shapes follow the usual (...,nk,nao,nao) convention; for one k-point,
        (...,nao,nao) is also accepted.  Without kpts_band the input shape is
        restored.  There is no density-matrix contraction fallback.
        '''
        if hermi != 1:
            raise ValueError('The supplied D/H/L equations require hermi=1')
        if exxdiv is not None:
            raise ValueError('Short-range PARI uses exxdiv=None')
        kpts = _as_kpts(self.kpts if kpts is None else kpts)
        bands = kpts if kpts_band is None else _as_kpts(kpts_band)
        dm = np.asarray(dm_kpts)
        nao, nk = self.cell.nao_nr(), len(kpts)
        if dm.ndim >= 3 and dm.shape[-3] == nk:
            leading = dm.shape[:-3]
        elif nk == 1:
            leading = dm.shape[:-2]
        else:
            raise ValueError('Density matrices must have shape (...,nkpts,nao,nao)')
        dms = np.asarray(dm, dtype=np.result_type(dm.dtype, np.float64)).reshape(-1, nk, nao, nao)
        # Validate/factor every input before starting expensive lattice work.
        factors = [self._factor_density(density) for density in dms]
        vk = np.asarray([self.get_k_from_m(m, kpts, bands) for m in factors])
        if not np.any(kpts) and not np.any(bands) and not np.iscomplexobj(dm):
            vk = vk.real
        if kpts_band is None:
            return vk.reshape(dm.shape)
        if np.asarray(kpts_band).ndim == 1:
            return vk[:, 0].reshape(leading + (nao, nao))
        return vk.reshape(leading + (len(bands), nao, nao))

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=False, with_k=True, omega=None, exxdiv=None):
        '''Exchange-only interface; optional omega must equal -PARI.omega.'''
        if with_j:
            raise NotImplementedError('Periodic PARI implements short-range exchange only')
        if omega is not None and (self.omega is None or omega >= 0 or
                                  not np.isclose(omega, -self.omega, rtol=0, atol=1e-14)):
            raise ValueError('get_jk omega must equal -PARI.omega')
        vk = self.get_k(dm, hermi, kpts, kpts_band, exxdiv) if with_k else None
        return None, vk

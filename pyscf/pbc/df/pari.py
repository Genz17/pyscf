from dataclasses import dataclass
import numpy as np
from scipy import linalg

from pyscf import gto, lib
from pyscf.df import incore as mol_incore
from pyscf.lib import logger
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import incore, rsdf_builder
from pyscf.pbc.tools import pbc as pbctools

import ctypes

libpari = lib.load_library('libpari')
I, P, D = ctypes.c_int, ctypes.c_void_p, ctypes.c_double

@dataclass
class _Pair:
    a: int
    b: int
    translation: np.ndarray
    ca: np.ndarray
    cb: object  # None for a pair on one atom in one cell (a set union).
    image: tuple

class CPair(ctypes.Structure):
    _fields_ = [(s, ctypes.c_int) for s in ('A', 'B', 'nao_A', 'nao_B', 'naux_A', 'naux_B')]
    _fields_ += [(s, ctypes.POINTER(ctypes.c_double)) for s in ('T1', 'ca', 'cb')]

def ptr(a):
    return a.ctypes.data_as(P)

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




def _images(cell, radius):
    translations = cell.get_lattice_Ls(rcut=radius, dimension=cell.dimension)
    indices = np.rint(translations @ np.linalg.inv(cell.lattice_vectors())).astype(int)
    indices = {tuple(x) for x in indices}
    indices |= {tuple(-np.asarray(x)) for x in indices}
    indices.add((0, 0, 0))
    lattice = cell.lattice_vectors()
    indices = sorted(indices, key=lambda x:x)
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

    _keys = {'cell', 'kpts', 'auxbasis', 'omega', 'precision', 'pair_rcut',
             'auxcell', 'max_memory', 'verbose',
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
        self.stdout, self.verbose = cell.stdout, cell.verbose
        self.max_memory = cell.max_memory
        self.reset()

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.auxcell = None
        self.rcut = None
        self.translations = None
        self._pairs = None
        self._pair_buffers = None
        self._signature = None
        self._build_token = object()
        self._metric_cache = None
        self._bare_mol = None
        self.integral_radius = self.metric_radius = None
        return self

    def _settings(self):
        return (id(self.cell), repr(self.auxbasis), self.omega, self.precision,
                self.pair_rcut, self.integral_rcut, self.metric_rcut)

    def _ensure_built(self):
        if self._signature != self._settings():
            self.build()

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('short-range omega = %s; auxbasis = %s', self.omega, self.auxbasis)
        log.info('integral_rcut = %s; metric_rcut = %s',
                 self.integral_rcut, self.metric_rcut)
        log.info('Literal erfc kernel, including its finite G=0 contribution')
        return self


    def _metric_block(self, index):
        """Return V[index][Q,R] = (Q_0|R_index)_SR."""

        index = tuple(index)
        if self._metric_cache is None:
            self._metric_cache = {}

        blocks = self._metric_cache

        if index not in blocks:
            reverse = tuple(-n for n in index)

            # Store the negative image, or zero, as a C-contiguous matrix.
            key = min(index, reverse)
            auxmol = self.auxcell.to_mol()

            if key == (0, 0, 0):
                with auxmol.with_range_coulomb(-self.omega):
                    matrix = auxmol.intor('int2c2e', hermi=1)
            else:
                translation = np.asarray(key).dot(self.cell.lattice_vectors())

                shifted = pbctools._build_supcell_(auxmol.copy(), self.auxcell, translation[None, :])
                pairmol = gto.conc_mol(auxmol, shifted)

                nbas = auxmol.nbas
                with pairmol.with_range_coulomb(-self.omega):
                    matrix = pairmol.intor('int2c2e', shls_slice=(0, nbas, nbas, 2 * nbas))

            matrix = np.ascontiguousarray(matrix, dtype=np.float64)
            blocks[key] = matrix

            if key != (0, 0, 0):
                opposite = tuple(-n for n in key)
                blocks[opposite] = matrix.T  # Shares the same memory.

        return blocks[index]


    def _metric_blocks(self):
        """Return only the metric images used by the H construction."""
        return {
            index: self._metric_block(index)
            for index in self._metric_indices
        }


    def build(self):
        self.reset()
        cell = self.cell
        aux = self.auxcell = incore.make_auxcell(cell, self.auxbasis)

        self._ao_slices = cell.aoslice_by_atom()
        self._aux_slices = aux.aoslice_by_atom()
        precision = self.precision * 1e-2

        self.rcut = self.pair_rcut
        if self.rcut is None:
            self.rcut = np.inf if cell.dimension == 0 else max(
                pbcgto.cell.estimate_rcut(cell, precision),
                rsdf_builder.estimate_ft_rcut(cell, precision).max())

        indices, self.translations = _images(cell, self.rcut)
        T1 = self.translations

        coords = np.ascontiguousarray(cell.atom_coords())
        nao = np.asarray(self._ao_slices[:, 3] - self._ao_slices[:, 2], dtype=np.int32)
        naux = np.asarray(self._aux_slices[:, 3] - self._aux_slices[:, 2], dtype=np.int32)

        offsets = np.ascontiguousarray(self._aux_slices[:, 2], dtype=np.int32) # aux functions starting index

        sizes = (ctypes.c_size_t * 3)() # number of (pairs, ca, cb)

        libpari._count_local_pairs(len(T1), ptr(T1), ptr(coords), ptr(nao), ptr(naux),cell.natm, ctypes.c_double(self.rcut), sizes)

        self._pairs = (CPair * sizes[0])()
        ca_buffer, cb_buffer = np.empty(sizes[1]), np.empty(sizes[2])

        # C reads only the negative images and the zero image.
        metrics = [np.ascontiguousarray(self._metric_block(i)) for i in indices[:len(T1) // 2 + 1]]
        metric_ptrs = (P * len(metrics))(*(m.ctypes.data for m in metrics))

        status = libpari._local_coeff(
            len(T1), ptr(T1), self._pairs, ptr(ca_buffer), ptr(cb_buffer), sizes,
            ptr(coords), ptr(nao), ptr(naux), ptr(offsets), aux.nao_nr(),
            metric_ptrs,
            ptr(cell._atm), ptr(cell._bas), ptr(cell._env),
            ptr(aux._atm), ptr(aux._bas), ptr(aux._env),
            cell.natm, cell.nbas, cell._env.size, aux.nbas, aux._env.size,
            int(cell.cart), ctypes.c_double(self.omega), ctypes.c_double(self.rcut))
        if status:
            raise RuntimeError(f'_local_coeff failed with status {status}')
        self._pair_buffers = (ca_buffer, cb_buffer, T1)

        self.integral_radius = self.integral_rcut
        self.metric_radius = self.metric_rcut
        if cell.dimension == 0:
            self.integral_radius = self.metric_radius = 0.
        else:
            if self.integral_radius is None:
                self.integral_radius = max(
                    rsdf_builder.estimate_rcut(
                        cell, aux, self.omega, precision).max(),
                    pbcgto.cell.estimate_rcut(cell, precision))
            if self.metric_radius is None:
                self.metric_radius = rsdf_builder.estimate_rs_2c2e_rcut(
                    aux, self.omega, precision)

        self._bare_indices, self._bare_translations = _images(cell, self.integral_radius)

        self._metric_indices, self._metric_translations = _images(
            cell, self.metric_radius)
        self._bare_mol = pbctools._build_supcell_(
            cell.to_mol().copy(), cell, self._bare_translations)
        self._signature = self._settings()
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
        # Build d[U][k][mu,Q,i] from the CPair records.
        # U is the lattice where the aux functions Q sits

        zero = (0, 0, 0)
        d = {zero: self._empty_intermediate(m)}
        inv_lattice = np.linalg.inv(self.cell.lattice_vectors())

        for pair in self._pairs:
            A, B = pair.A, pair.B
            na, nb = pair.nao_A, pair.nao_B
            nqa, nqb = pair.naux_A, pair.naux_B

            a0, a1 = self._ao_slices[A, 2:]
            b0, b1 = self._ao_slices[B, 2:]
            qa0, qa1 = self._aux_slices[A, 2:]
            qb0, qb1 = self._aux_slices[B, 2:]

            # Cartesian translation and its integer lattice index.
            T1 = np.ctypeslib.as_array(pair.T1, shape=(3,))
            image = tuple(np.rint(T1 @ inv_lattice).astype(int))
            reverse = tuple(-x for x in image)
            same = (A == B and image == zero)

            # C stores coefficients as [AO on A, AO on B, auxiliary].
            ca = np.ctypeslib.as_array(
                pair.ca, shape=(na * nb * nqa,)
            ).reshape(na, nb, nqa)

            # For a same-center pair, C allocates no cb coefficients.
            cb = None
            if not same:
                cb = np.ctypeslib.as_array(
                    pair.cb, shape=(na * nb * nqb,)
                ).reshape(na, nb, nqb)

                for u in (image, reverse):
                    if u not in d:
                        d[u] = self._empty_intermediate(m)

            phases = np.exp(-1j * kpts.dot(T1))

            for ik, (mk, phase) in enumerate(zip(m, phases)):
                ma = mk[a0:a1]
                mb = mk[b0:b1]

                # Forward pair (A_0, B_T1), auxiliaries on A_0.
                d[zero][ik][a0:a1, qa0:qa1] += phase * lib.einsum('abq,bi->aqi', ca, mb)

                if same:
                    continue

                # Forward pair (A_0, B_T1), auxiliaries on B_T1.
                d[image][ik][a0:a1, qb0:qb1] += phase * lib.einsum('abq,bi->aqi', cb, mb)

                reverse_phase = phase.conjugate()

                # Reverse pair (B_0, A_-T1), auxiliaries on B_0.
                d[zero][ik][b0:b1, qb0:qb1] += reverse_phase * np.einsum('abq,ai->bqi', cb, ma)

                # Reverse pair (B_0, A_-T1), auxiliaries on A_-T1.
                d[reverse][ik][b0:b1, qa0:qa1] += reverse_phase * np.einsum('abq,ai->bqi', ca, ma)

        return d


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
        L_realspace = {}
        for u, du in data.d.items():
            phases = np.exp(-1j * data.kpts.dot(np.dot(u, data.lattice)))
            for t, ht in data.h0.items():
                output = tuple(np.asarray(u) + t)
                if output not in L_realspace:
                    L_realspace[output] = np.zeros((nao, nao), dtype=complex)
                for dk, hk, phase in zip(du, ht, phases):
                    # This is H_i^{nu_(U+t) Q_U,k}, with the actual Q image.
                    h_image = phase * hk
                    L_realspace[output] += np.einsum('mqi,nqi->mn', dk.conj(), h_image)
        if not L_realspace:
            L_realspace[(0, 0, 0)] = np.zeros((nao, nao), dtype=complex)
        return {t: value / len(data.kpts) for t, value in L_realspace.items()}

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

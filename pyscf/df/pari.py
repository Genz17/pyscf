import os

import contextlib
import numpy as np
import h5py
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf import df
from pyscf.df import incore
from pyscf.df import outcore
from pyscf.df import r_incore
from pyscf.df import addons
from pyscf.df import df_jk
from pyscf.df import DF
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos, iden_coeffs
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf import __config__

from scipy import linalg
from copy import deepcopy

def _aux_shell_blocks(auxmol, block_nao):
    """Yield (sh0, sh1, q0, q1) for whole-shell auxiliary blocks.
    """

    loc = auxmol.ao_loc_nr()
    sh0 = 0

    while sh0 < auxmol.nbas:
        # Always include at least one complete shell.
        sh1 = sh0 + 1

        # Add shells while the function count fits.
        while (sh1 < auxmol.nbas
               and loc[sh1 + 1] - loc[sh0] <= block_nao):
            sh1 += 1

        q0 = int(loc[sh0])
        q1 = int(loc[sh1])
        yield sh0, sh1, q0, q1

        sh0 = sh1

class PARI(lib.StreamObject):

    _keys = {'mol', 'auxmol', 'auxbasis', 'block_nao', 'max_memory',
             'verbose', 'stdout', '_pari_file'}

    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.auxbasis = auxbasis
        self.auxmol = None
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.block_nao = 1000
        self._df_j = df.DF(mol, auxbasis=auxbasis)
        self._built = False
        self._tmpfile = None
        self._tmpfile = lib.NamedTemporaryFile(dir=lib.param.TMPDIR)
        filename = self._tmpfile.name
        self._pari_file = os.fspath(filename)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('auxbasis = %s', self.auxbasis)
        log.info('PARI cache = %s', self._pari_file)
        log.info('block_nao = %s; max_memory = %s MB',
                 self.block_nao, self.max_memory)
        return self
    
    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.auxmol = None
        self._built = False
        self._df_j.reset(self.mol)
        return self

    def _block_size(self, nvec=0):
        """Estimate the auxiliary block size from available memory."""
        nao = self.mol.nao_nr()

        available = max(
            0.0, self.max_memory - lib.current_memory()[0]
        ) * 1e6

        # Approximate space for W and contraction intermediates.
        bytes_per_q = 8 * (3 * nao**2 + 4 * nao * nvec)

        return max(
            1,
            min(
                int(self.block_nao),
                int(0.8 * available / bytes_per_q),
            ),
        )

    def build(self):
        """Build density-independent C and W on disk; return self."""
        self._built = False
        self.dump_flags()
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        self.auxmol = auxmol = df.addons.make_auxmol(mol, self.auxbasis)
        ao_slices = mol.aoslice_by_atom()
        aux_slices = auxmol.aoslice_by_atom()
        nao, naux = mol.nao_nr(), auxmol.nao_nr()

        # This is the only full auxiliary-space matrix held in memory.
        metric = auxmol.intor('int2c2e', hermi=1)

        with h5py.File(self._pari_file, 'w') as f:
            f.attrs['format'] = 'pari'
            f.attrs['complete'] = False
            f['ao_slices'] = ao_slices
            f['aux_slices'] = aux_slices

            f['metric'] = metric
            groups = f.create_group('coeffs')

            # Fit each unordered atom pair only in its own auxiliary domain.
            for A in range(mol.natm):
                ash0, ash1, a0, a1 = ao_slices[A]
                for B in range(A, mol.natm):
                    bsh0, bsh1, b0, b1 = ao_slices[B]
                    atoms = (A,) if A == B else (A, B)
                    indices, integrals = [], []
                    for atom in atoms:
                        qsh0, qsh1, q0, q1 = aux_slices[atom]
                        indices.append(np.arange(q0, q1))
                        # Auxiliary shell numbers are relative to auxmol.
                        integrals.append(df.incore.aux_e2(
                            mol, auxmol, aosym='s1',
                            shls_slice=(ash0, ash1, bsh0, bsh1, qsh0, qsh1)))
                    aux_idx = np.concatenate(indices)
                    ints = np.concatenate(integrals, axis=2)
                    vlocal = metric[np.ix_(aux_idx, aux_idx)]
                    rhs = ints.reshape(-1, len(aux_idx)).T

                    cab = linalg.solve(vlocal, rhs, assume_a='pos').T
                    cab = cab.reshape(a1 - a0, b1 - b0, len(aux_idx))

                    group = groups.create_group(f'{A}_{B}')
                    group['aux_idx'] = aux_idx
                    group['C'] = cab
                    del integrals, ints, rhs, vlocal, cab
            metric_factor = linalg.cho_factor(metric, lower=True, overwrite_a=True, check_finite=False)

            jdata = f.create_dataset('J_coeff', shape=(nao, nao, naux), dtype='f8')

            for sh0, sh1, a0, a1 in _aux_shell_blocks(mol, self.block_nao):
                ints = df.incore.aux_e2(
                    mol, auxmol, aosym='s1',
                    shls_slice=(
                        sh0, sh1,       # First orbital index: this block
                        0, mol.nbas,   # Second orbital index: all AOs
                        0, auxmol.nbas,  # All auxiliary functions
                    ),
                )
                # ints.shape = (a1 - a0, nao, naux)
                rhs = ints.reshape(-1, naux).T

                coeff = linalg.cho_solve(metric_factor, rhs, check_finite=False)
                jdata[a0:a1, :, :] = coeff.T.reshape(a1 - a0, nao, naux)

                del ints, rhs, coeff

            f.attrs['complete'] = True

        self._built = True
        log.timer('PARI build', *t0)
        return self

    def loop(self, blksize=None, dm=None, with_k=True):
        """Yield (q0, q1, W_block, J_block).

        dm=None skips J.
        with_k=False skips construction of W.
        """
        if not self._built:
            self.build()

        if blksize is None:
            blksize = self._block_size()
        blksize = int(blksize)
        if blksize < 1:
            raise ValueError('blksize must be positive')

        mol, auxmol = self.mol, self.auxmol

        with h5py.File(self._pari_file, 'r') as f:
            ao_slices = f['ao_slices'][:]
            groups = f['coeffs']

            for sh0, sh1, q0, q1 in _aux_shell_blocks(auxmol, blksize):
                ints = df.incore.aux_e2(
                    mol, auxmol, aosym='s1',
                    shls_slice=(0, mol.nbas, 0, mol.nbas, sh0, sh1),
                )
                # ints.shape = (nao, nao, q1 - q0)

                # Standard DF Coulomb contribution from RAW integrals.
                jblock = None
                if dm is not None:
                    rho = lib.einsum('ijq,ij->q', ints, dm)
                    jcoeff = f['J_coeff'][:, :, q0:q1]
                    jblock = lib.einsum('ijq,q->ij', jcoeff, rho)
                    del jcoeff, rho

                w = None
                if with_k:
                    w = np.ascontiguousarray(ints.transpose(2, 0, 1))
                del ints

                if with_k:
                    # Read these columns once for all atom pairs.
                    metric_block = f['metric'][:, q0:q1]

                    for A in range(mol.natm):
                        a0, a1 = ao_slices[A, 2:]

                        for B in range(A, mol.natm):
                            b0, b1 = ao_slices[B, 2:]
                            group = groups[f'{A}_{B}']
                            aux_idx = group['aux_idx'][:]
                            cab = group['C'][:]

                            correction = lib.einsum(
                                'abr,rq->qab',
                                cab, metric_block[aux_idx],
                            )

                            w[:, a0:a1, b0:b1] -= 0.5 * correction
                            if A != B:
                                w[:, b0:b1, a0:a1] -= (
                                    0.5 * correction.transpose(0, 2, 1)
                                )

                            del cab, correction

                    del metric_block

                yield q0, q1, w, jblock
                del w, jblock


    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        """Return standard DF J and PARI K for one real symmetric density.

        direct_scf_tol is accepted for API compatibility; this implementation
        does not apply density-based screening.
        """
        if not with_j and not with_k:
            return None, None

        if omega not in (None, 0):
            raise NotImplementedError(
                'Range-separated J/K is not implemented here.'
            )

        dm = np.asarray(dm)
        nao = self.mol.nao_nr()

        if (hermi != 1
                or np.iscomplexobj(dm)
                or dm.shape != (nao, nao)
                or not np.allclose(dm, dm.T, rtol=1e-10, atol=1e-12)):
            raise ValueError(
                'get_jk expects one real symmetric density matrix.'
            )

        vj = np.zeros((nao, nao)) if with_j else None
        vk = np.zeros((nao, nao)) if with_k else None

        if not np.any(dm):
            return vj, vk

        # Factor dm only when exchange is requested.
        nvec = 0
        if with_k:
            eig, vec = linalg.eigh(dm)
            cutoff = np.finfo(float).eps * nao * np.max(np.abs(eig))
            keep = np.abs(eig) > cutoff
            m = vec[:, keep] * np.sqrt(np.abs(eig[keep]))
            signs = np.sign(eig[keep])
            nvec = m.shape[1]

        do_k = with_k and nvec > 0
        if not with_j and not do_k:
            return vj, vk

        if not self._built:
            self.build()

        blksize = self._block_size(nvec=nvec)
        L = np.zeros((nao, nao)) if do_k else None

        with h5py.File(self._pari_file, 'r') as f:
            ao_slices = f['ao_slices'][:]
            groups = f['coeffs']

            for q0, q1, w, jblock in self.loop(
                    blksize, dm=dm if with_j else None, with_k=do_k):

                if with_j:
                    vj += jblock
                del jblock

                if not do_k:
                    continue

                # d[Q, mu, i] = sum_lambda C[mu, lambda, Q] M[lambda, i]
                d = np.zeros((q1 - q0, nao, nvec))

                for A in range(self.mol.natm):
                    a0, a1 = ao_slices[A, 2:]

                    for B in range(A, self.mol.natm):
                        b0, b1 = ao_slices[B, 2:]
                        group = groups[f'{A}_{B}']
                        aux_idx = group['aux_idx'][:]

                        mask = (aux_idx >= q0) & (aux_idx < q1)
                        if not np.any(mask):
                            continue

                        qlocal = aux_idx[mask] - q0
                        cab = group['C'][:][:, :, mask]

                        d[qlocal, a0:a1] += lib.einsum(
                            'abq,bi->qai', cab, m[b0:b1],
                        )
                        if A != B:
                            d[qlocal, b0:b1] += lib.einsum(
                                'abq,ai->qbi', cab, m[a0:a1],
                            )

                        del cab

                h = lib.einsum('qns,si->qni', w, m)
                h *= signs[None, None, :]

                L += lib.einsum('qmi,qni->mn', d, h)
                del w, d, h

        if do_k:
            vk = L + L.T

        return vj, vk



if __name__ == '__main__':
    from pyscf import gto, scf
    import time

    mol = gto.M(
        atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587;',
        basis='ccpvqz', verbose=4)
    mf = scf.RKS(mol)
    mf.xc = 'b3lyp'

    s = time.perf_counter()
    mf.kernel()
    e = time.perf_counter()
    print("NO DF: %.5f" % (e-s))

    mf = scf.RKS(mol).density_fit()
    mf.with_df = PARI(mol, auxbasis = "ccpvqz-jkfit")
    mf.xc = 'b3lyp'
    s = time.perf_counter()
    mf.kernel()
    e = time.perf_counter()
    print("PARI: %.5f" % (e-s))

    mf = scf.RKS(mol).density_fit()
    mf.with_df = DF(mol, auxbasis = "ccpvqz-jkfit")
    mf.xc = 'b3lyp'
    s = time.perf_counter()
    mf.kernel()
    e = time.perf_counter()
    print("DF: %.5f" % (e-s))
    #dm = mf.make_rdm1()  # Includes the RHF factor of two in the occupations.

    #pari = PARI(mol, auxbasis='ccpvqz-jkfit')
    #pari.set(blockdim=32).build()
    #k_pari = pari.get_k(dm)
    #k_exact = mf.get_k(dm=dm)
    #print('PARI K shape:', k_pari.shape)
    #print('max |K_PARI - K_exact|:', np.max(np.abs(k_pari - k_exact)))
    #print('RHF exchange-energy difference:',
    #      -0.25 * np.einsum('ij,ji->', dm, k_pari - k_exact))

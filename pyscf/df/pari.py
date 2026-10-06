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

def _aux_shell_blocks(auxmol, blockdim):
    """Yield shell and function bounds, keeping whole auxiliary shells."""
    loc = auxmol.ao_loc_nr()
    sh0 = 0
    while sh0 < auxmol.nbas:
        sh1 = sh0 + 1
        while sh1 < auxmol.nbas and loc[sh1 + 1] - loc[sh0] <= blockdim:
            sh1 += 1
        yield sh0, sh1, int(loc[sh0]), int(loc[sh1])
        sh0 = sh1


class PARI(lib.StreamObject):

    _keys = {'mol', 'auxmol', 'auxbasis', 'blockdim', 'max_memory',
             'verbose', 'stdout', '_pari_file'}

    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.auxbasis = auxbasis
        self.auxmol = None
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.blockdim = 64
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
        log.info('blockdim = %s; max_memory = %s MB',
                 self.blockdim, self.max_memory)
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
                int(self.blockdim),
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
                    try:
                        cab = linalg.solve(vlocal, rhs, assume_a='pos').T
                    except linalg.LinAlgError as err:
                        raise linalg.LinAlgError(
                            f'PARI metric for atom pair ({A}, {B}) is not '
                            'positive definite; remove local auxiliary '
                            'linear dependence before fitting.') from err
                    cab = cab.reshape(a1 - a0, b1 - b0, len(aux_idx))
                    group = groups.create_group(f'{A}_{B}')
                    group['aux_idx'] = aux_idx
                    group['C'] = cab
                    del integrals, ints, rhs, vlocal, cab

            # Q is GLOBAL here, even though the contracted R domain is local.
            # A contiguous Q-first dataset supports sequential block reads.
            wdata = f.create_dataset('W', (naux, nao, nao), dtype='f8')
            blockdim = self._block_size()
            for qsh0, qsh1, q0, q1 in _aux_shell_blocks(auxmol, blockdim):
                ints = df.incore.aux_e2(
                    mol, auxmol, aosym='s1',
                    shls_slice=(0, mol.nbas, 0, mol.nbas, qsh0, qsh1))
                w = np.ascontiguousarray(ints.transpose(2, 0, 1))
                del ints
                for A in range(mol.natm):
                    a0, a1 = ao_slices[A, 2:]
                    for B in range(A, mol.natm):
                        b0, b1 = ao_slices[B, 2:]
                        group = groups[f'{A}_{B}']
                        aux_idx = group['aux_idx'][:]
                        cab = group['C'][:]
                        correction = np.einsum(
                            'abr,rq->qab', cab, metric[aux_idx, q0:q1],
                            optimize=True)
                        w[:, a0:a1, b0:b1] -= 0.5 * correction
                        if A != B:
                            w[:, b0:b1, a0:a1] -= 0.5 * correction.transpose(0, 2, 1)
                        del cab, correction
                wdata[q0:q1] = w
                del w
                log.debug('PARI W auxiliary functions [%d:%d]', q0, q1)
            f.attrs['complete'] = True

        self._built = True
        log.timer('PARI build', *t0)
        return self

    def loop(self, blksize=None):
        """Yield (q0, q1, W_block), with W_block.shape = (q1-q0, nao, nao)."""
        if not self._built:
            self.build()
        if blksize is None:
            blksize = self._block_size()
        blksize = max(1, int(blksize))
        with h5py.File(self._pari_file, 'r') as f:
            wdata = f['W']
            for q0 in range(0, wdata.shape[0], blksize):
                q1 = min(q0 + blksize, wdata.shape[0])
                yield q0, q1, wdata[q0:q1]

    def get_k(self, dm, hermi=1):
        """Return K[mu,nu] = sum_(lambda,sigma) (mu lambda|nu sigma)_PARI dm.

        Factor dm = M diag(sign) M.T. For a physical density sign is positive;
        allowing negative signs also supports symmetric density differences.
        Eigenvalues below floating-point roundoff relative to dm are dropped.
        K contains no RHF -1/2 prefactor.
        """
        dm = np.asarray(dm)
        nao = self.mol.nao_nr()
        if (hermi != 1 or np.iscomplexobj(dm) or dm.shape != (nao, nao)
                or not np.allclose(dm, dm.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('get_k expects one real symmetric (nao, nao) density.')
        if not self._built:
            self.build()
        eig, vec = linalg.eigh(dm)
        cutoff = np.finfo(float).eps * nao * np.max(np.abs(eig))
        keep = np.abs(eig) > cutoff
        m = vec[:, keep] * np.sqrt(np.abs(eig[keep]))
        signs = np.sign(eig[keep])
        if m.shape[1] == 0:
            return np.zeros((nao, nao))
        L = np.zeros((nao, nao))
        blksize = self._block_size(nvec=m.shape[1])

        with h5py.File(self._pari_file, 'r') as f:
            ao_slices = f['ao_slices'][:]
            groups = f['coeffs']
            for q0, q1, w in self.loop(blksize):
                # D[Q,mu,i] = sum_lambda C[mu,lambda,Q] M[lambda,i]
                d = np.zeros((q1 - q0, nao, m.shape[1]))
                for A in range(self.mol.natm):
                    a0, a1 = ao_slices[A, 2:]
                    for B in range(A, self.mol.natm):
                        b0, b1 = ao_slices[B, 2:]
                        group = groups[f'{A}_{B}']
                        aux_idx = group['aux_idx'][:]
                        mask = (aux_idx >= q0) & (aux_idx < q1)
                        if not np.any(mask):
                            continue
                        # Read one pair into NumPy before selecting its Qs.
                        cab = group['C'][:, :, :][:, :, mask]
                        qlocal = aux_idx[mask] - q0
                        d[qlocal, a0:a1] += np.einsum(
                            'abq,bi->qai', cab, m[b0:b1], optimize=True)
                        if A != B:
                            d[qlocal, b0:b1] += np.einsum(
                                'abq,ai->qbi', cab, m[a0:a1], optimize=True)
                        del cab
                # H[Q,nu,i] = sum_sigma W[Q,nu,sigma] M[sigma,i]
                h = np.einsum('qns,si->qni', w, m, optimize=True)
                h *= signs[None, None, :]
                L += np.einsum('qmi,qni->mn', d, h, optimize=True)
                del w, d, h
        return L + L.T

    def get_j(self, dm, hermi=1, direct_scf_tol=1e-13, omega=None):
        self._df_j.auxbasis = self.auxbasis
        self._df_j.max_memory = self.max_memory
        self._df_j.blockdim = self.blockdim
        self._df_j.verbose = self.verbose
        self._df_j.stdout = self.stdout

        vj, _ = self._df_j.get_jk(
            dm,
            hermi=hermi,
            with_j=True,
            with_k=False,
            direct_scf_tol=direct_scf_tol,
            omega=omega,
        )
        return vj

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        if with_k and omega not in (None, 0):
            raise NotImplementedError(
                "Range-separated PARI K is not implemented."
            )

        vj = vk = None

        if with_j:
            vj = self.get_j(
                dm, hermi=hermi,
                direct_scf_tol=direct_scf_tol,
                omega=omega,
            )

        if with_k:
            vk = self.get_k(dm, hermi=hermi)

        return vj, vk

if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M(
        atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        basis='ccpvqz', verbose=4)
    pari = PARI(mol, auxbasis="ccpvqz-jkfit")
    mf = scf.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    mf = scf.RKS(mol).density_fit()
    mf.with_df = PARI(mol, auxbasis = "ccpvqz-jkfit")
    mf.xc = 'b3lyp'
    mf.kernel()

    mf = scf.RKS(mol).density_fit()
    mf.with_df = DF(mol, auxbasis = "ccpvqz-jkfit")
    mf.xc = 'b3lyp'
    mf.kernel()
    #dm = mf.make_rdm1()  # Includes the RHF factor of two in the occupations.

    #pari = PARI(mol, auxbasis='ccpvqz-jkfit')
    #pari.set(blockdim=32).build()
    #k_pari = pari.get_k(dm)
    #k_exact = mf.get_k(dm=dm)
    #print('PARI K shape:', k_pari.shape)
    #print('max |K_PARI - K_exact|:', np.max(np.abs(k_pari - k_exact)))
    #print('RHF exchange-energy difference:',
    #      -0.25 * np.einsum('ij,ji->', dm, k_pari - k_exact))

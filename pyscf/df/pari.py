import os
import contextlib
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import df
from pyscf.df import incore
from pyscf.df import addons
from pyscf.df import DF
from pyscf.scf import _vhf
from pyscf import gto, scf
from pyscf import __config__

import time
import ctypes
from scipy import linalg
from copy import deepcopy


libpari = lib.load_library("libpari")

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
             'verbose', 'stdout', 'tol'}

    def __init__(self, mol, auxbasis=None):

        if mol.cart:
            raise NotImplementedError("PARI currently supports spherical bases only.")

        self.mol = mol
        self.tol = 1e-12
        self.auxbasis = auxbasis
        self.auxmol = None
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.block_nao = 32
        self._built = False

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('auxbasis = %s', self.auxbasis)
        log.info('block_nao = %s; max_memory = %s MB',
                 self.block_nao, self.max_memory)
        return self
    

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.auxmol = None
        self.cintopt = None
        self._built = False
        return self

    def _block_size(self, nvec=0):

        nao = self.mol.nao_nr()

        available = max(
            0.0, self.max_memory - lib.current_memory()[0]
        ) * 1e6

        bytes_per_q = 8 * (3 * nao**2 + 4 * nao * nvec) # this should be checked

        return max(1,
                min(int(self.block_nao),
                int(0.8 * available / bytes_per_q)))


    def build(self):

        self._built = False
        self.dump_flags()
        log = logger.new_logger(self)
        t0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol

        opt = _vhf._VHFOpt(
            mol, 'int2e', 'CVHFnrs8_prescreen',
            'CVHFnr_int2e_q_cond', 'CVHFnr_dm_cond',
        )

        self.auxmol = auxmol = df.addons.make_auxmol(mol, self.auxbasis)
        self.ao_mnmn = np.ascontiguousarray(opt.get_q_cond(), dtype = np.float64)
        self.schwarz_mask = np.ascontiguousarray(self.ao_mnmn > self.tol, dtype = np.uint8)


        atm_conc, bas_conc, env_conc = gto.mole.conc_env(
            mol._atm, mol._bas, mol._env,
            auxmol._atm, auxmol._bas, auxmol._env)


        ao_loc_conc = gto.moleintor.make_loc(bas_conc, "int3c2e_sph")

        ao_slices = mol.aoslice_by_atom()
        aux_slices = auxmol.aoslice_by_atom()
        nao, naux = mol.nao_nr(), auxmol.nao_nr()

        env_conc = np.ascontiguousarray(env_conc, dtype=np.float64)
        atm_conc = np.ascontiguousarray(atm_conc, dtype=np.int32)
        bas_conc = np.ascontiguousarray(bas_conc, dtype=np.int32)
        ao_loc_conc = np.ascontiguousarray(ao_loc_conc, dtype=np.int32)
        ao_slices = np.ascontiguousarray(ao_slices, dtype=np.int32)
        aux_slices = np.ascontiguousarray(aux_slices, dtype=np.int32)

        metric = auxmol.intor('int2c2e', hermi=1)
        metric_factor = linalg.cho_factor(metric, lower=True, overwrite_a=False, check_finite=False)
        self.metric = np.ascontiguousarray(metric, dtype = np.float64)
        self.metric_factor = metric_factor


        q_aux = np.sqrt(np.abs(metric.diagonal()))
        aux_loc = auxmol.ao_loc_nr()
        self.aux_PP = np.array([q_aux[aux_loc[s]:aux_loc[s + 1]].max() for s in range(auxmol.nbas)])

        npair = int((mol.natm + 1) * mol.natm / 2)


        nao_per_atom = ao_slices[:, 3] - ao_slices[:, 2]
        naux_per_atom = aux_slices[:, 3] - aux_slices[:, 2]
        max_nao = int(nao_per_atom.max())
        max_naux = int(naux_per_atom.max())
        cab_per_pair_size = max_nao**2 * (2 * max_naux)
        cab = np.zeros(npair * cab_per_pair_size, dtype=np.float64)

        cab_offset = np.zeros(npair, dtype = np.int32, order = 'C')
        libpari._local_coeff(self.metric.ctypes.data_as(ctypes.c_void_p), cab.ctypes.data_as(ctypes.c_void_p), cab_offset.ctypes.data_as(ctypes.c_void_p), cab_per_pair_size,
                             atm_conc.ctypes.data_as(ctypes.c_void_p), bas_conc.ctypes.data_as(ctypes.c_void_p), env_conc.ctypes.data_as(ctypes.c_void_p), ao_loc_conc.ctypes.data_as(ctypes.c_void_p),
                             self.schwarz_mask.ctypes.data_as(ctypes.c_void_p),
                             ao_slices.ctypes.data_as(ctypes.c_void_p), aux_slices.ctypes.data_as(ctypes.c_void_p),
                             mol.natm, mol.nbas,
                             mol.natm+auxmol.natm, mol.nbas+auxmol.nbas,
                             nao, naux)

        self.cab = cab
        self.cab_per_pair_size = cab_per_pair_size
        self.cab_offset = cab_offset

        self.cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, "int3c2e_sph")

        self._built = True
        log.timer('PARI build', *t0)
        return self

    def loop(self, blksize=None, vj = None, L = None, signs = None,
             dm=None, m = None, with_j = True, with_k=True):

        if not self._built:
            self.build()

        ao_slices = np.ascontiguousarray(
            self.mol.aoslice_by_atom(), dtype=np.int32)
        aux_slices = np.ascontiguousarray(
            self.auxmol.aoslice_by_atom(), dtype=np.int32)

        if blksize is None:
            blksize = self._block_size()

        blksize = int(blksize)
        if blksize < 1:
            raise ValueError('blksize must be positive')

        mol, auxmol = self.mol, self.auxmol

        do_j = with_j and dm is not None

        #if do_j:
        #    # this part needs to be moved to C
        #    j_tol = self.tol  # Set to zero to disable this screening.

        #    pairs = []
        #    for A in range(mol.natm):
        #        i0, i1, a0, a1 = ao_slices[A]

        #        for B in range(A, mol.natm):
        #            j0, j1, b0, b1 = ao_slices[B]
        #            if a0 == a1 or b0 == b1:
        #                continue

        #            dm_ab = dm[a0:a1, b0:b1]
        #            factor = 1 if A == B else 2
        #            qab = self.ao_mnmn[i0:i1, j0:j1].max()
        #            bound = factor * qab * np.abs(dm_ab).sum()

        #            pairs.append(
        #                (i0, i1, j0, j1, dm_ab, factor, bound)
        #            )


        #    b = np.zeros(auxmol.nao_nr())

        #    for sh0, sh1, q0, q1 in _aux_shell_blocks(auxmol, blksize):
        #        aux_bound = self.aux_PP[sh0:sh1].max()

        #        for i0, i1, j0, j1, dm_ab, factor, bound in pairs:
        #            if bound * aux_bound < j_tol:
        #                continue

        #            ints = df.incore.aux_e2(
        #                mol, auxmol,
        #                aosym="s1",
        #                cintopt=self.cintopt,
        #                shls_slice=(i0, i1, j0, j1, sh0, sh1),
        #            )
        #            b[q0:q1] += factor * lib.einsum(
        #                "ijq,ij->q", ints, dm_ab
        #            )
        #            del ints

        #    rho = linalg.cho_solve(self.metric_factor, b)
        #    del b

        if do_j:
            j_tol = self.tol
            dm = np.ascontiguousarray(dm, dtype=np.float64)
            b = np.zeros(auxmol.nao_nr())

            atm_conc, bas_conc, env_conc = gto.mole.conc_env(
                mol._atm, mol._bas, mol._env,
                auxmol._atm, auxmol._bas, auxmol._env,
            )
            ao_loc_conc = gto.moleintor.make_loc(bas_conc, "int3c2e_sph")

            aux_bound = np.empty(auxmol.nbas, dtype=np.float64)
            for sh0, sh1, q0, q1 in _aux_shell_blocks(auxmol, blksize):
                aux_bound[sh0:sh1] = self.aux_PP[sh0:sh1].max()

            if libpari._loop_j_bbuild(
                b.ctypes.data_as(ctypes.c_void_p),
                dm.ctypes.data_as(ctypes.c_void_p),
                self.ao_mnmn.ctypes.data_as(ctypes.c_void_p),
                aux_bound.ctypes.data_as(ctypes.c_void_p),
                ao_slices.ctypes.data_as(ctypes.c_void_p),
                ao_loc_conc.ctypes.data_as(ctypes.c_void_p),
                mol.natm, mol.nbas,
                atm_conc.ctypes.data_as(ctypes.c_void_p),
                mol.natm + auxmol.natm,
                bas_conc.ctypes.data_as(ctypes.c_void_p),
                mol.nbas + auxmol.nbas,
                env_conc.ctypes.data_as(ctypes.c_void_p),
                self.cintopt,
                ctypes.c_double(j_tol),
            ):
                raise MemoryError("Failed to allocate J-build workspace")

            rho = linalg.cho_solve(self.metric_factor, b)
            del b


        for sh0, sh1, q0, q1 in _aux_shell_blocks(auxmol, blksize):
            ints = df.incore.aux_e2(
                mol, auxmol, aosym='s2ij', cintopt=self.cintopt,
                shls_slice=(0, mol.nbas, 0, mol.nbas, sh0, sh1),
            )

            #jblock = None
            if do_j:
                vj += lib.unpack_tril(ints @ rho[q0:q1])
                #vj += lib.einsum('Q, mnQ -> mn', rho[q0:q1], ints)

            w = None
            if with_k:
                w = lib.unpack_tril(np.ascontiguousarray(ints.T))
                #w = np.ascontiguousarray(ints.transpose(2, 0, 1))
                m = np.ascontiguousarray(m, dtype = np.float64)
                d = np.ascontiguousarray(np.zeros((q1 - q0, mol.nao_nr(), m.shape[-1])), dtype=np.float64)
                nocc = m.shape[-1]
                metric_block = np.ascontiguousarray(self.metric[:, q0:q1])

                h = np.empty_like(d)
                libpari._loop_k_wdbuild(
                                metric_block.ctypes.data_as(ctypes.c_void_p),
                                self.cab.ctypes.data_as(ctypes.c_void_p), self.cab_per_pair_size, self.cab_offset.ctypes.data_as(ctypes.c_void_p),
                                m.ctypes.data_as(ctypes.c_void_p), w.ctypes.data_as(ctypes.c_void_p), d.ctypes.data_as(ctypes.c_void_p), h.ctypes.data_as(ctypes.c_void_p),
                                mol.natm, nocc,
                                ao_slices.ctypes.data_as(ctypes.c_void_p),aux_slices.ctypes.data_as(ctypes.c_void_p), auxmol.nao_nr(), mol.nao_nr(),
                                q0, q1)

                #h = lib.einsum('qns,si->qni', w, m)
                h *= signs[None, None, :]
                L += lib.einsum('qmi,qni->mn', d, h)

                del metric_block
                del w 
        return


    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):

        if not with_j and not with_k:
            return None, None

        if omega not in (None, 0):
            raise NotImplementedError('Range-separated J/K is not implemented here.')

        mo_coeff = getattr(dm, 'mo_coeff', None)
        mo_occ = getattr(dm, 'mo_occ', None)
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

        nvec = 0
        if with_k:
            if mo_coeff is not None and mo_occ is not None:
                mo_coeff = np.asarray(mo_coeff)
                mo_occ = np.asarray(mo_occ)
                keep = mo_occ > 0
                m = mo_coeff[:, keep] * np.sqrt(mo_occ[keep])
                signs = np.ones(m.shape[1])
            else:
                eig, vec = linalg.eigh(dm)
                cutoff = np.finfo(float).eps * nao * np.max(np.abs(eig))
                keep = np.abs(eig) > cutoff
                m = vec[:, keep] * np.sqrt(np.abs(eig[keep]))
                signs = np.sign(eig[keep])

            nvec = m.shape[1]

        do_k = with_k and nvec > 0
        do_j = with_j and dm is not None

        if not do_j and not do_k:
            return vj, vk

        if not self._built:
            self.build()

        blksize = self._block_size(nvec=nvec)

        L = np.zeros((nao, nao)) if do_k else None

        self.loop(blksize, vj=vj, L=L,
                  signs=signs if do_k else None,
                  dm=dm if do_j else None,
                  m=m if do_k else None,
                  with_j=do_j, with_k=do_k)

        if do_k:
            vk = L + L.T

        return vj, vk



if __name__ == '__main__':

    mol = gto.M(
        atom=
        ' \
        O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587; \
        O 10 0 0; H 10 -0.757 0.587; H 10 0.757 0.587; \
        O 20 0 0; H 20 -0.757 0.587; H 20 0.757 0.587; \
        O 30 0 0; H 30 -0.757 0.587; H 30 0.757 0.587; \
        ',
        basis='ccpvtz', verbose=4)
    mol.max_memory = 3
    mf = scf.RKS(mol)
    mf.xc = 'b3lyp'

    #s = time.perf_counter()
    #mf.kernel()
    #e = time.perf_counter()
    #print("NO DF: %.5f" % (e-s))

    mf = scf.RKS(mol).density_fit()
    mf.with_df = PARI(mol, auxbasis = "ccpvtz-jkfit")
    mf.xc = 'b3lyp'
    s = time.perf_counter()
    mf.kernel()
    e = time.perf_counter()
    print("PARI: %.5f" % (e-s))

    mf = scf.RKS(mol).density_fit()
    mf.with_df = DF(mol, auxbasis = "ccpvtz-jkfit")
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

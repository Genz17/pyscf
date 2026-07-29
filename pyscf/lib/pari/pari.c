/* Copyright 2014-2021 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Hong-Zhou Ye <hzyechem@gmail.com>
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "vhf/fblas.h"


static size_t _max_cache_size(CINTIntegralFunction *intor, int *shls_slice,
                              int *atm, int natm, int *bas, int nbas,
                              double *env)
{
    int i, ish;
    int ish0 = shls_slice[0];
    int ish1 = shls_slice[1];
    size_t cache_size = 0;
    size_t n;
    int shls[3];

    for (i = 1; i < 3; i++) {
        if (shls_slice[i*2] < ish0) {
            ish0 = shls_slice[i*2];
        }
        if (shls_slice[i*2+1] > ish1) {
            ish1 = shls_slice[i*2+1];
        }
    }

    for (ish = ish0; ish < ish1; ish++) {
        shls[0] = shls[1] = shls[2] = ish;
        n = (*intor)(NULL, NULL, shls, atm, natm,
                     bas, nbas, env, NULL, NULL);
        if (n > cache_size) {
            cache_size = n;
        }
    }
    return cache_size;
}


static void _zero_shell_pair(double *out, int comp, int i0, int j0,
                             int di, int dj, int naoi, int naoj, int naok)
{
    int ic, j, k;
    const size_t nij = (size_t)naoi * naoj;
    const size_t nijk = nij * naok;

    for (ic = 0; ic < comp; ic++) {
        for (k = 0; k < naok; k++) {
            for (j = j0; j < j0+dj; j++) {
                memset(out + (size_t)ic*nijk + (size_t)k*nij
                       + (size_t)j*naoi + i0, 0, sizeof(double)*di);
            }
        }
    }
}


/*
 * Evaluate three-center integrals in the same layout as
 * df.incore.aux_e2(..., aosym='s1').
 *
 * Args:
 *     out:
 *         F-contiguous output with layout [naoi,naoj,naok,comp].
 *     shls_slice:
 *         AO1, AO2, and auxiliary shell ranges.
 *     ao_loc:
 *         AO offsets for the concatenated AO and auxiliary bases.
 *     shlpr_mask:
 *         Optional dense Boolean AO shell-pair mask.
 *
 * Returns:
 *     The integral values are written to out. Skipped shell-pair blocks
 *     are set to zero.
 */
void fill_aux_e2(CINTIntegralFunction *intor, double *out, int comp,
                 int *shls_slice, int *ao_loc,
                 const uint8_t *shlpr_mask, int nbas_ao, CINTOpt *cintopt,
                 int *atm, int natm, int *bas, int nbas, double *env)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];
    const int njsh = jsh1 - jsh0;

    const int naoi = ao_loc[ish1] - ao_loc[ish0];
    const int naoj = ao_loc[jsh1] - ao_loc[jsh0];
    const int naok = ao_loc[ksh1] - ao_loc[ksh0];
    int dims[] = {naoi, naoj, naok};
    const size_t nij = (size_t)naoi * naoj;
    const size_t nshl_pair = (size_t)(ish1-ish0) * njsh;
    const size_t cache_size = _max_cache_size(
        intor, shls_slice, atm, natm, bas, nbas, env);

#pragma omp parallel
{
    size_t ijsh;
    int ish, jsh, ksh;
    int i0, j0, k0, di, dj;
    int shls[3];
    double *cache = malloc(sizeof(double) * cache_size);

#pragma omp for schedule(dynamic, 4)
    for (ijsh = 0; ijsh < nshl_pair; ijsh++) {
        ish = ijsh / njsh + ish0;
        jsh = ijsh % njsh + jsh0;
        i0 = ao_loc[ish] - ao_loc[ish0];
        j0 = ao_loc[jsh] - ao_loc[jsh0];

        if (shlpr_mask != NULL &&
            !shlpr_mask[(size_t)ish*nbas_ao+jsh]) {
            di = ao_loc[ish+1] - ao_loc[ish];
            dj = ao_loc[jsh+1] - ao_loc[jsh];
            _zero_shell_pair(out, comp, i0, j0, di, dj,
                             naoi, naoj, naok);
            continue;
        }

        shls[0] = ish;
        shls[1] = jsh;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
            k0 = ao_loc[ksh] - ao_loc[ksh0];
            shls[2] = ksh;
            (*intor)(out + i0 + (size_t)naoi*j0 + nij*k0,
                     dims, shls, atm, natm, bas, nbas, env,
                     cintopt, cache);
        }
    }
    free(cache);
}
}


static size_t _max_shell_size(int shlpr0, int shlpr1,
                              const int *shlpr, int ksh0, int ksh1,
                              const int *ao_loc)
{
    int ijsh, ish, jsh, ksh;
    size_t dij;
    size_t max_dij = 0;
    size_t max_dk = 0;

    for (ijsh = shlpr0; ijsh < shlpr1; ijsh++) {
        ish = shlpr[ijsh*2  ];
        jsh = shlpr[ijsh*2+1];
        dij = ((size_t)ao_loc[ish+1] - ao_loc[ish]) *
              (ao_loc[jsh+1] - ao_loc[jsh]);
        if (dij > max_dij) {
            max_dij = dij;
        }
    }
    for (ksh = ksh0; ksh < ksh1; ksh++) {
        if (ao_loc[ksh+1] - ao_loc[ksh] > max_dk) {
            max_dk = ao_loc[ksh+1] - ao_loc[ksh];
        }
    }
    return max_dij * max_dk;
}


static void _copy_ints_sparse(double *out, const double *buf, int comp,
                              int64_t row0, int p0, int di, int dj, int dk,
                              int naux, int64_t nrow, int diagonal)
{
    int ic, i, j, k;
    int64_t row;
    const size_t dij = (size_t)di * dj;
    const size_t dijk = dij * dk;
    const size_t nrowaux = (size_t)nrow * naux;
    double *pout;
    const double *pbuf;

    for (ic = 0; ic < comp; ic++) {
        pout = out + (size_t)ic*nrowaux;
        pbuf = buf + (size_t)ic*dijk;
        row = row0;
        if (diagonal) {
            for (i = 0; i < di; i++) {
                for (j = i; j < dj; j++, row++) {
                    for (k = 0; k < dk; k++) {
                        pout[(size_t)row*naux+p0+k] =
                            pbuf[i + (size_t)di*(j + (size_t)dj*k)];
                    }
                }
            }
        } else {
            for (i = 0; i < di; i++) {
                for (j = 0; j < dj; j++, row++) {
                    for (k = 0; k < dk; k++) {
                        pout[(size_t)row*naux+p0+k] =
                            pbuf[i + (size_t)di*(j + (size_t)dj*k)];
                    }
                }
            }
        }
    }
}


static void _zero_ints_sparse(double *out, int comp,
                              int64_t row0, int64_t row1,
                              int p0, int dk, int naux, int64_t nrow)
{
    int ic;
    int64_t row;
    const size_t nrowaux = (size_t)nrow * naux;
    double *pout;

    for (ic = 0; ic < comp; ic++) {
        pout = out + (size_t)ic*nrowaux;
        for (row = row0; row < row1; row++) {
            memset(pout + (size_t)row*naux+p0, 0, sizeof(double)*dk);
        }
    }
}


/*
 * Evaluate selected AO shell pairs and pack them as
 * out[comp,naopair,naux] in C-order. For diagonal AO shells, only
 * i <= j is stored.
 *
 * Args:
 *     out:
 *         C-contiguous packed output.
 *     shlpr0, shlpr1:
 *         Range of shell-pair jobs in shlpr.
 *     shlpr:
 *         Retained AO shell pairs.
 *     aopair_loc:
 *         Packed AO-pair offsets for each shell pair.
 *     shls_slice:
 *         AO1, AO2, and auxiliary shell ranges.
 *
 * Returns:
 *     The selected three-center integrals are written to out.
 */
void fill_aux_e2_sparse(CINTIntegralFunction *intor, double *out, int comp,
                        int shlpr0, int shlpr1, const int *shlpr,
                        const int64_t *aopair_loc, int *shls_slice,
                        int *ao_loc, CINTOpt *cintopt,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];
    const int naux = ao_loc[ksh1] - ao_loc[ksh0];
    const int64_t aopair0 = aopair_loc[shlpr0];
    const int64_t nrow = aopair_loc[shlpr1] - aopair0;
    const size_t cache_size = _max_cache_size(
        intor, shls_slice, atm, natm, bas, nbas, env);
    const size_t buf_size = _max_shell_size(
        shlpr0, shlpr1, shlpr, ksh0, ksh1, ao_loc) * comp;

#pragma omp parallel
{
    int ijsh, ish, jsh, ksh;
    int di, dj, dk, p0;
    int has_value;
    int dims[3];
    int shls[3];
    int64_t row0, row1;
    double *buf = malloc(sizeof(double) * buf_size);
    double *cache = malloc(sizeof(double) * cache_size);

#pragma omp for schedule(dynamic, 4)
    for (ijsh = shlpr0; ijsh < shlpr1; ijsh++) {
        ish = shlpr[ijsh*2  ];
        jsh = shlpr[ijsh*2+1];
        di = ao_loc[ish+1] - ao_loc[ish];
        dj = ao_loc[jsh+1] - ao_loc[jsh];
        row0 = aopair_loc[ijsh  ] - aopair0;
        row1 = aopair_loc[ijsh+1] - aopair0;

        shls[0] = ish;
        shls[1] = jsh;
        dims[0] = di;
        dims[1] = dj;

        for (ksh = ksh0; ksh < ksh1; ksh++) {
            dk = ao_loc[ksh+1] - ao_loc[ksh];
            p0 = ao_loc[ksh] - ao_loc[ksh0];
            shls[2] = ksh;
            dims[2] = dk;
            has_value = (*intor)(buf, dims, shls, atm, natm,
                                 bas, nbas, env, cintopt, cache);
            if (has_value) {
                _copy_ints_sparse(out, buf, comp, row0, p0, di, dj, dk,
                                  naux, nrow, ish == jsh);
            } else {
                _zero_ints_sparse(out, comp, row0, row1,
                                  p0, dk, naux, nrow);
            }
        }
    }
    free(buf);
    free(cache);
}
}


/* Scatter one packed atom-pair Coulomb block to a symmetric AO matrix. */
static void _scatter_jpair(double *vj, const double *jpair,
                           int64_t shlpr0, int64_t shlpr1,
                           const int *shlpr, const int64_t *aopair_loc,
                           const int *ao_loc, int nao)
{
    const int64_t aopair0 = aopair_loc[shlpr0];

    for (int64_t ijsh = shlpr0; ijsh < shlpr1; ijsh++) {
        const int ish = shlpr[ijsh*2  ];
        const int jsh = shlpr[ijsh*2+1];
        const int i0 = ao_loc[ish];
        const int j0 = ao_loc[jsh];
        const int di = ao_loc[ish+1] - i0;
        const int dj = ao_loc[jsh+1] - j0;
        int64_t row = aopair_loc[ijsh] - aopair0;

        if (ish == jsh) {
            for (int i = 0; i < di; i++) {
                for (int j = i; j < dj; j++, row++) {
                    const double value = jpair[row];
                    vj[(size_t)(i0+i)*nao+j0+j] += value;
                    if (i != j) {
                        vj[(size_t)(j0+j)*nao+i0+i] += value;
                    }
                }
            }
        } else {
            for (int i = 0; i < di; i++) {
                for (int j = 0; j < dj; j++, row++) {
                    const double value = jpair[row];
                    vj[(size_t)(i0+i)*nao+j0+j] += value;
                    vj[(size_t)(j0+j)*nao+i0+i] += value;
                }
            }
        }
    }
}


/*
 * Evaluate and metric-correct each canonical AO atom pair. Retain the
 * packed G[naopair,aux] layout. If vj is not NULL, contract the raw
 * integrals with rho before applying the metric correction.
 *
 * For an AO atom pair BC and auxiliary atom A,
 *
 *     G^A_BC = (BC|A)
 *            - 1/2 d^B_BC (B|A)
 *            - 1/2 d^C_BC (C|A).
 */
static void _PARIfill_g(CINTIntegralFunction *intor, double *G,
                        double *vj, const double *rho,
                        const double *coeff, const double *j2c,
                        int naux, int npair, const int *pair_atoms,
                        const int64_t *pair_aopair_loc,
                        const int64_t *shlpr_loc,
                        const int *shlpr, const int64_t *aopair_loc,
                        const int *ao_loc, const int64_t *coeff_offsets,
                        const int64_t *aux_loc, int *shls_slice,
                        CINTOpt *cintopt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D0 = 0;
    const double D1 = 1;
    const double DMHALF = -.5;
    const int INC1 = 1;
    const int ksh0 = shls_slice[4];
    const int ksh1 = shls_slice[5];
    const int nao = ao_loc[shls_slice[1]];
    const size_t cache_size = _max_cache_size(
        intor, shls_slice, atm, natm, bas, nbas, env);
    const size_t intbuf_size = _max_shell_size(
        0, shlpr_loc[npair], shlpr, ksh0, ksh1, ao_loc);
    int max_nrow = 0;
    for (int pair = 0; pair < npair; pair++) {
        const int nrow = pair_aopair_loc[pair+1] -
                         pair_aopair_loc[pair];
        if (nrow > max_nrow) {
            max_nrow = nrow;
        }
    }
    memset(G, 0, sizeof(double) *
           (size_t)pair_aopair_loc[npair]*naux);

#pragma omp parallel
{
    double *intbuf = malloc(sizeof(double) * intbuf_size);
    double *jpair = vj == NULL ? NULL :
                    malloc(sizeof(double) * max_nrow);
    double *cache = malloc(sizeof(double) * cache_size);

#pragma omp for schedule(dynamic, 1)
    for (int pair = 0; pair < npair; pair++) {
        const int atom0 = pair_atoms[pair*2  ];
        const int atom1 = pair_atoms[pair*2+1];
        const int64_t aopair0 = pair_aopair_loc[pair];
        const int nrow = pair_aopair_loc[pair+1] - aopair0;
        const int64_t shlpr0 = shlpr_loc[pair];
        const int64_t shlpr1 = shlpr_loc[pair+1];
        double *gpair = G + (size_t)aopair0*naux;

        for (int64_t ijsh = shlpr0; ijsh < shlpr1; ijsh++) {
            const int ish = shlpr[ijsh*2  ];
            const int jsh = shlpr[ijsh*2+1];
            const int di = ao_loc[ish+1] - ao_loc[ish];
            const int dj = ao_loc[jsh+1] - ao_loc[jsh];
            const int64_t row0 = aopair_loc[ijsh] - aopair0;
            int dims[3] = {di, dj, 0};
            int shls[3] = {ish, jsh, 0};

            for (int ksh = ksh0; ksh < ksh1; ksh++) {
                const int dk = ao_loc[ksh+1] - ao_loc[ksh];
                const int p0 = ao_loc[ksh] - ao_loc[ksh0];
                dims[2] = dk;
                shls[2] = ksh;
                const int has_value = (*intor)(
                    intbuf, dims, shls, atm, natm, bas, nbas, env,
                    cintopt, cache);
                if (has_value) {
                    _copy_ints_sparse(
                        gpair, intbuf, 1, row0, p0, di, dj, dk,
                        naux, nrow, ish == jsh);
                }
            }
        }

        if (vj != NULL) {
            dgemv_(&TRANS_T, &naux, &nrow, &D1, gpair, &naux,
                   rho, &INC1, &D0, jpair, &INC1);
            _scatter_jpair(vj, jpair, shlpr0, shlpr1,
                           shlpr, aopair_loc, ao_loc, nao);
        }

        const int naux0 = aux_loc[atom0+1] - aux_loc[atom0];
        const double *coeff0 = coeff + coeff_offsets[pair*3];
        const double *j2c0 = j2c + (size_t)aux_loc[atom0]*naux;
        dgemm_(&TRANS_N, &TRANS_N, &naux, &nrow, &naux0,
               &DMHALF, j2c0, &naux, coeff0, &naux0,
               &D1, gpair, &naux);

        if (atom0 != atom1) {
            const int naux1 = aux_loc[atom1+1] - aux_loc[atom1];
            const double *coeff1 = coeff + coeff_offsets[pair*3+1];
            const double *j2c1 = j2c + (size_t)aux_loc[atom1]*naux;
            dgemm_(&TRANS_N, &TRANS_N, &naux, &nrow, &naux1,
                   &DMHALF, j2c1, &naux, coeff1, &naux1,
                   &D1, gpair, &naux);
        }

    }
    free(intbuf);
    free(jpair);
    free(cache);
}
}


/*
 * Build one packed, metric-corrected G panel.
 *
 * Args:
 *     G:
 *         C-contiguous output with layout [naopair,naux_A].
 *     coeff:
 *         Contiguous PARI coefficient storage.
 *     j2c:
 *         Metric panel with layout [naux,naux_A].
 *     pair_atoms, pair_aopair_loc, shlpr_loc, shlpr, aopair_loc:
 *         Shared packed AO-pair layout.
 *
 * Returns:
 *     The metric-corrected G panel is written to G.
 */
void PARIfill_g(CINTIntegralFunction *intor, double *G,
                const double *coeff, const double *j2c,
                int naux, int npair, const int *pair_atoms,
                const int64_t *pair_aopair_loc, const int64_t *shlpr_loc,
                const int *shlpr, const int64_t *aopair_loc,
                const int *ao_loc, const int64_t *coeff_offsets,
                const int64_t *aux_loc, int *shls_slice, CINTOpt *cintopt,
                int *atm, int natm, int *bas, int nbas, double *env)
{
    _PARIfill_g(intor, G, NULL, NULL, coeff, j2c, naux, npair,
                pair_atoms, pair_aopair_loc, shlpr_loc, shlpr,
                aopair_loc, ao_loc, coeff_offsets, aux_loc, shls_slice,
                cintopt, atm, natm, bas, nbas, env);
}


/*
 * Build one G panel and accumulate the raw integrals into J.
 *
 * The Coulomb contraction is performed before the metric correction,
 *
 *     vj[mu,nu] += sum_P (mu nu|P) rho[P].
 *
 * Args:
 *     G:
 *         C-contiguous output with layout [naopair,naux_A].
 *     vj:
 *         Dense symmetric AO Coulomb matrix updated in place.
 *     rho:
 *         Fitted Coulomb density for the current auxiliary atom.
 *
 * Returns:
 *     G is overwritten and vj is accumulated in place.
 */
void PARIfill_gj(CINTIntegralFunction *intor, double *G,
                 double *vj, const double *rho,
                 const double *coeff, const double *j2c,
                 int naux, int npair, const int *pair_atoms,
                 const int64_t *pair_aopair_loc,
                 const int64_t *shlpr_loc,
                 const int *shlpr, const int64_t *aopair_loc,
                 const int *ao_loc, const int64_t *coeff_offsets,
                 const int64_t *aux_loc, int *shls_slice,
                 CINTOpt *cintopt, int *atm, int natm,
                 int *bas, int nbas, double *env)
{
    _PARIfill_g(intor, G, vj, rho, coeff, j2c, naux, npair,
                pair_atoms, pair_aopair_loc, shlpr_loc, shlpr,
                aopair_loc, ao_loc, coeff_offsets, aux_loc, shls_slice,
                cintopt, atm, natm, bas, nbas, env);
}


/*
 * Half-transform sparse AO-pair data. Directed shell-pair jobs are
 * grouped by their target AO shell.
 *
 * Args:
 *     out:
 *         C-contiguous output with layout [nmo,naux,nao_out].
 *     mo_coeff:
 *         C-contiguous AO-to-MO coefficients [nao,nmo].
 *     data:
 *         Packed AO-pair data.
 *     target_loc, source_shell, data_offset, edge_kind:
 *         Directed target/source shell-pair representation.
 *
 * Returns:
 *     The half-transformed tensor is written to out.
 */
void PARIhalf_transform(double *out, const double *mo_coeff,
                        const double *data,
                        int nmo, int naux, int nao_out, int nbas,
                        const int *ao_loc, const int *out_ao_loc,
                        const int64_t *target_loc,
                        const int *source_shell,
                        const int64_t *data_offset,
                        const uint8_t *edge_kind, int offset_scale)
{
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D1 = 1;
    int max_shell = 0;
    int max_source = 0;

    for (int ish = 0; ish < nbas; ish++) {
        const int di = ao_loc[ish+1] - ao_loc[ish];
        if (di > max_shell) {
            max_shell = di;
        }
        int nsource = 0;
        for (int64_t edge = target_loc[ish];
             edge < target_loc[ish+1]; edge++) {
            const int ssh = source_shell[edge];
            nsource += ao_loc[ssh+1] - ao_loc[ssh];
        }
        if (nsource > max_source) {
            max_source = nsource;
        }
    }
    memset(out, 0, sizeof(double) * (size_t)nmo*naux*nao_out);

#pragma omp parallel
{
    const size_t cbuf_size = (size_t)max_source*max_shell*naux;
    double *cbuf = malloc(sizeof(double) * cbuf_size);
    double *dbuf = malloc(
        sizeof(double) * (size_t)nmo*max_shell*naux);
    double *mobuf = malloc(sizeof(double) * (size_t)max_source*nmo);

#pragma omp for schedule(dynamic, 1)
    for (int tsh = 0; tsh < nbas; tsh++) {
        const int64_t edge0 = target_loc[tsh];
        const int64_t edge1 = target_loc[tsh+1];
        if (edge0 == edge1) {
            continue;
        }

        const int t0 = out_ao_loc[tsh];
        const int dt = ao_loc[tsh+1] - ao_loc[tsh];
        const int ndp = dt * naux;
        memset(dbuf, 0, sizeof(double) * (size_t)nmo*ndp);

        int nsource = 0;
        for (int64_t edge = edge0; edge < edge1; edge++) {
            const int ssh = source_shell[edge];
            const int s0 = ao_loc[ssh];
            const int ds = ao_loc[ssh+1] - s0;
            const double *c0 = data + data_offset[edge]*offset_scale;
            double *cp = cbuf + (size_t)nsource*ndp;

            if (edge_kind[edge] == 0) {
                memcpy(cp, c0, sizeof(double) * (size_t)ds*ndp);
            } else if (edge_kind[edge] == 1) {
                for (int s = 0; s < ds; s++) {
                    for (int t = 0; t < dt; t++) {
                        for (int p = 0; p < naux; p++) {
                            cp[((size_t)s*dt+t)*naux+p] =
                                c0[((size_t)t*ds+s)*naux+p];
                        }
                    }
                }
            } else {
                int64_t row = 0;
                for (int s = 0; s < ds; s++) {
                    for (int t = s; t < dt; t++, row++) {
                        for (int p = 0; p < naux; p++) {
                            const double v = c0[(size_t)row*naux+p];
                            cp[((size_t)s*dt+t)*naux+p] = v;
                            cp[((size_t)t*dt+s)*naux+p] = v;
                        }
                    }
                }
            }
            memcpy(mobuf + (size_t)nsource*nmo,
                   mo_coeff + (size_t)s0*nmo,
                   sizeof(double) * (size_t)ds*nmo);
            nsource += ds;
        }
        dgemm_(&TRANS_N, &TRANS_T, &ndp, &nmo, &nsource,
               &D1, cbuf, &ndp, mobuf, &nmo,
               &D1, dbuf, &ndp);

        for (int i = 0; i < nmo; i++) {
            for (int t = 0; t < dt; t++) {
                for (int p = 0; p < naux; p++) {
                    out[((size_t)i*naux+p)*nao_out+t0+t] =
                        dbuf[((size_t)i*dt+t)*naux+p];
                }
            }
        }
    }
    free(cbuf);
    free(dbuf);
    free(mobuf);
}
}


/*
 * Scatter NBX-active AO rows into L.
 *
 * Args:
 *     L:
 *         Dense AO matrix updated in place.
 *     buf:
 *         Contiguous active-row buffer [nactive,nao].
 *     ao_idx:
 *         Global AO index for every active row.
 *
 * Returns:
 *     The rows in buf are accumulated into L.
 */
void PARIscatter_l_nbx(double *L, const double *buf, const int *ao_idx,
                       int nao, int nactive)
{
#pragma omp parallel for schedule(static)
    for (int ia = 0; ia < nactive; ia++) {
        double *Lrow = L + (size_t)ao_idx[ia]*nao;
        const double *brow = buf + (size_t)ia*nao;
        for (int nu = 0; nu < nao; nu++) {
            Lrow[nu] += brow[nu];
        }
    }
}

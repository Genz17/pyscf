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
 * out[naoi,naoj,naok,comp] in F-order
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

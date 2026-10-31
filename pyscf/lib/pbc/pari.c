#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <cint.h>
#include <cint_funcs.h>
#include "gto/gto.h"
#include "vhf/fblas.h"

double vector_norm_squared(double *vec, int length) {
	double res = 0.;
	for (int i = 0; i < length; i++) {
		res += vec[i] * vec[i];
	}

	return res;
}

typedef struct {
    int A, B, nao_A, nao_B, naux_A, naux_B;
	double *T1, *ca, *cb;
	// ca of size nao_A * nao_B * naux_A
	// cb of size nao_A * nao_B * naux_B
} PARILoopPair;

/* Shared selection rule: 1 means keep, 0 means skip, -1 means invalid input.
 * sizea/sizeb count doubles, not bytes. No integrals are evaluated here.
 */

static int _pari_pair_sizes(int A, int B, int zero_image,
    						const double *translation,
    						const double *coords,
    						const int *nao, const int *naux,
    						double pair_rcut, int *same,
    						size_t *sizea, size_t *sizeb) {

    double distance2 = 0.;
    for (int xyz = 0; xyz < 3; xyz++) {
        double dr = coords[(size_t)B * 3 + xyz] + translation[xyz] - coords[(size_t)A * 3 + xyz];
        distance2 += dr * dr;
    }

    if (distance2 > pair_rcut * pair_rcut) return 0;

    /* The auxiliary basis occurs once for the same atom and cell. */
    *same = (A == B && zero_image);

    size_t npair = (size_t)nao[A] * nao[B];

    /* Numbers of doubles required for this pair. */
    *sizea = npair * naux[A];
    *sizeb = *same ? 0 : npair * naux[B];

    return 1;
}

int _count_local_pairs(int T1_num, const double *T1,
                       const double *coords,
                       const int *nao, const int *naux,
                       int mol_natm, double pair_rcut, size_t sizes[3]) {

    int middle = T1_num / 2;

    for (int it = 0; it <= middle; it++) {

        const double *translation = T1 + (size_t)it * 3;
        int zero_image = (it == middle);

        for (int A = 0; A < mol_natm; A++) {
			
            for (int B = zero_image ? A : 0; B < mol_natm; B++) {

                int same;
                size_t sizea, sizeb;
                int selected = _pari_pair_sizes(
                    A, B, zero_image, translation, coords,
                    nao, naux, pair_rcut, &same, &sizea, &sizeb);

                if (!selected) continue; // skipped

                sizes[0]++;
                sizes[1] += sizea;
                sizes[2] += sizeb;
            }
        }
    }

    return 0;
}

static void _pari_copy_metric(double *metric, int nq,
                              int A, int B, int same,
                              const int *naux, const int *aux_offsets,
                              int naux_cell,
                              const double *v0, const double *vt) {

    int nqa = naux[A], nqb = naux[B];
    int a0 = aux_offsets[A], b0 = aux_offsets[B];

    /* Upper-left block: (aux_A0 | aux_A0). */
    for (int q = 0; q < nqa; q++) {
        for (int r = 0; r < nqa; r++) {
            metric[q + (size_t)nq * r] =
                v0[(size_t)(a0 + q) * naux_cell + a0 + r];
        }
    }

    /* Same atom and image: include its auxiliary basis only once. */
    if (same) return;

    /* Lower-right block: (aux_BT | aux_BT) = (aux_B0 | aux_B0). */
    for (int q = 0; q < nqb; q++) {
        for (int r = 0; r < nqb; r++) {
            metric[nqa + q + (size_t)nq * (nqa + r)] =
                v0[(size_t)(b0 + q) * naux_cell + b0 + r];
        }
    }

    /* Off-diagonal blocks: (aux_A0 | aux_BT) and its transpose. */
    for (int q = 0; q < nqa; q++) {
        for (int r = 0; r < nqb; r++) {
            double value =
                vt[(size_t)(a0 + q) * naux_cell + b0 + r];

            metric[q + (size_t)nq * (nqa + r)] = value;
            metric[nqa + r + (size_t)nq * q] = value;
        }
    }
}

int _local_coeff(int T1_num, double *T1,
    			 PARILoopPair *pairs,
    			 double *ca, double *cb,
    			 const size_t sizes[3],
    			 const double *coords,
    			 const int *nao, const int *naux,
    			 const int *aux_offsets, int naux_cell,
    			 const double *const *metric_blocks,
    			 const int *_atm, const int *_bas, const double *_env,
    			 const int *_aux_atm, const int *_aux_bas, const double *_aux_env,
    			 int mol_natm, int mol_nbas, int mol_env_size,
    			 int aux_nbas, int aux_env_size, int cart,
    			 double omega, double pair_rcut) {

    extern CINTIntegralFunction int2c2e_sph, int2c2e_cart;
    extern CINTIntegralFunction int3c2e_sph, int3c2e_cart;
    extern CINTOptimizerFunction int2c2e_optimizer, int3c2e_optimizer;
    extern void dsyev_(const char *, const char *, const int *, double *,
                       const int *, double *, double *, const int *, int *);
	extern void dpotrf_(const char *, const int *, double *,
                   const int *, int *);

    int *atm = NULL, *bas = NULL, *loc = NULL;
    double *env = NULL, *storage = NULL;
    CINTOpt *opt3 = NULL;
    int status = 0;

    /* Make four environment copies once; only atom/basis records and
     * coordinate slots change between pairs. Never shift the caller's env.
     */
    size_t max_nbas = 2 * ((size_t)mol_nbas + aux_nbas);
    size_t nenv = 2 * ((size_t)mol_env_size + aux_env_size) + 12;

    atm = malloc(4 * ATM_SLOTS * sizeof(*atm));
    bas = malloc(max_nbas * BAS_SLOTS * sizeof(*bas));
    loc = malloc((max_nbas + 1) * sizeof(*loc));
    env = malloc(nenv * sizeof(*env));

    memcpy(env, _env, (size_t)mol_env_size * sizeof(*env));
    memcpy(env + mol_env_size, _env, (size_t)mol_env_size * sizeof(*env));
    memcpy(env + 2 * mol_env_size, _aux_env, (size_t)aux_env_size * sizeof(*env));
    memcpy(env + 2 * mol_env_size + aux_env_size, _aux_env,
           (size_t)aux_env_size * sizeof(*env));
    env[PTR_RANGE_OMEGA] = -omega;

    size_t ipair = 0, offa = 0, offb = 0;
    int middle = T1_num / 2;
    for (int it = 0; it <= middle; it++) {
        double *translation = T1 + (size_t)it * 3;
        int zero_image = (it == middle);
        for (int A = 0; A < mol_natm; A++) {
            for (int B = zero_image ? A : 0; B < mol_natm; B++) {
                int same;
                size_t sizea, sizeb;
                int selected = _pari_pair_sizes(
                    A, B, zero_image, translation, coords,
                    nao, naux, pair_rcut, &same, &sizea, &sizeb);

                if (!selected) continue;
                if (ipair >= sizes[0] || sizea > sizes[1] - offa ||
                    sizeb > sizes[2] - offb) { status = -3; goto cleanup; }

                int na = nao[A], nb = nao[B];
                int nq = naux[A] + (same ? 0 : naux[B]);
                size_t npair_size = (size_t)na * nb;
                /* A_0, B_T, aux(A_0), aux(B_T), with independent coordinate
                 * pointers. In the same-center case each basis occurs once.
                 * Original basis coefficients and environments are preserved.
                 */
                int starts[4], ends[4];
                int env_offsets[4] = {0, mol_env_size, 2 * mol_env_size,
                                     2 * mol_env_size + aux_env_size};
                int coord_start = 2 * mol_env_size + 2 * aux_env_size;

                int nbas = 0;
                for (int group = 0; group < 4; group++) {
                    int atom = group % 2 ? B : A;
                    int auxiliary = group >= 2;
                    const int *source_atm = auxiliary ? _aux_atm : _atm;
                    const int *source_bas = auxiliary ? _aux_bas : _bas;
                    int source_nbas = auxiliary ? aux_nbas : mol_nbas;
                    int *record = atm + group * ATM_SLOTS;
                    memcpy(record, source_atm + (size_t)atom * ATM_SLOTS,
                           ATM_SLOTS * sizeof(*record));
                    record[PTR_COORD] = coord_start + group * 3;
                    record[PTR_ZETA] += env_offsets[group];
                    if (record[PTR_FRAC_CHARGE] != 0) {
                        record[PTR_FRAC_CHARGE] += env_offsets[group];
                    }
                    for (int xyz = 0; xyz < 3; xyz++) {
                        env[record[PTR_COORD] + xyz] = coords[(size_t)atom * 3 + xyz]
                            + (group % 2 ? translation[xyz] : 0.);
                    }
                    if (same && group % 2) {
                        starts[group] = starts[group - 1];
                        ends[group] = ends[group - 1];
                        continue;
                    }
                    starts[group] = nbas;
                    for (int ish = 0; ish < source_nbas; ish++) {
                        const int *shell = source_bas + (size_t)ish * BAS_SLOTS;
                        if (shell[ATOM_OF] != atom) continue;
                        int *target = bas + (size_t)nbas * BAS_SLOTS;
                        memcpy(target, shell, BAS_SLOTS * sizeof(*target));
                        target[ATOM_OF] = group;
                        target[PTR_EXP] += env_offsets[group];
                        target[PTR_COEFF] += env_offsets[group];
                        nbas++;
                    }
                    ends[group] = nbas;
                }

                loc[0] = 0;
                for (int ish = 0; ish < nbas; ish++) {
                    loc[ish + 1] = loc[ish] + (cart ? CINTcgto_cart(ish, bas)
                                                   : CINTcgto_spheric(ish, bas));
                }

                CINTIntegralFunction *intor3 = cart ? int3c2e_cart : int3c2e_sph;
                int slices3[6] = {starts[0], ends[0], starts[1], ends[1],
                                  starts[2], ends[3]};

                size_t cache3 = GTOmax_cache_size((int (*)())intor3, slices3, 3,
                                                 atm, 4, bas, nbas, env);
                size_t ncache = cache3;

                int npair = (int)npair_size;
                int lwork = 3 * nq;  /* DSYEV requires at least 3*nq-1. */
                size_t ncoeff = npair_size * nq;
                size_t nstorage = (size_t)nq * nq + 2 * ncoeff + 2 * nq
                                  + (size_t)lwork + ncache;

                storage = calloc(nstorage, sizeof(*storage));

                double *metric = storage;
                double *rhs = metric + (size_t)nq * nq;
                double *projected = rhs + ncoeff;
                double *scale = projected + ncoeff;
                double *eigenvalues = scale + nq;
                double *work = eigenvalues + nq;
                double *cache = work + lwork;
                int3c2e_optimizer(&opt3, atm, 4, bas, nbas, env);

				_pari_copy_metric(metric, nq, A, B, same, naux, aux_offsets, naux_cell, metric_blocks[middle], metric_blocks[it]);

                int dims3[3] = {na, nb, nq};
                for (int qsh = starts[2]; qsh < ends[3]; qsh++) {
                    for (int jsh = starts[1]; jsh < ends[1]; jsh++) {
                        for (int ish = starts[0]; ish < ends[0]; ish++) {
                            int shls[3] = {ish, jsh, qsh};
                            size_t offset = loc[ish] - loc[starts[0]] + (size_t)na * (loc[jsh] - loc[starts[1]]) + npair_size * (loc[qsh] - loc[starts[2]]);

                            intor3(rhs + offset, dims3, shls,
                                   atm, 4, bas, nbas, env, opt3, cache);
                        }
                    }
                }
                CINTdel_optimizer(&opt3);

				for (int q = 0; q < nq; q++) {
					double diagonal = metric[q + (size_t)nq * q];

					if (!(diagonal > 0.) || !isfinite(diagonal)) {
						status = -4;
						goto cleanup;
					}

					scale[q] = sqrt(diagonal);
				}

				/* Normalize the metric and the three-center integrals. */
				for (int q = 0; q < nq; q++) {
					for (int r = 0; r < nq; r++) {
						metric[r + (size_t)nq * q] /= scale[r] * scale[q];
					}
					for (size_t p = 0; p < npair_size; p++) {
						rhs[p + npair_size * q] /= scale[q];
					}
				}

				int info = 0;
				double one = 1.;

				/* Factor the normalized metric: V_normalized = L L^T. */
				dpotrf_("L", &nq, metric, &nq, &info);

				if (info != 0) {
					status = info > 0 ? -4 : -5;
					goto cleanup;
				}

				/* Solve X L^T = B_normalized. */
				dtrsm_("R", "L", "T", "N",
					   &npair, &nq, &one,
					   metric, &nq, rhs, &npair);

				/* Solve Y L = X. */
				dtrsm_("R", "L", "N", "N",
					   &npair, &nq, &one,
					   metric, &nq, rhs, &npair);

                PARILoopPair *pair = pairs + ipair;
                pair->A = A;
                pair->B = B;
                pair->nao_A = na;
                pair->nao_B = nb;
                pair->naux_A = naux[A];
                pair->naux_B = naux[B];
                pair->T1 = translation;
                pair->ca = sizea ? ca + offa : NULL;
                pair->cb = sizeb ? cb + offb : NULL;
                for (int i = 0; i < na; i++) {
                    for (int j = 0; j < nb; j++) {
                        size_t p = i + (size_t)na * j;
                        size_t output = (size_t)i * nb + j;
                        for (int q = 0; q < naux[A]; q++) {
                            pair->ca[output * naux[A] + q] =
                                rhs[p + npair_size * q] / scale[q];
                        }
                        if (!same) {
                            for (int q = 0; q < naux[B]; q++) {
                                pair->cb[output * naux[B] + q] =
                                    rhs[p + npair_size * (naux[A] + q)]
                                    / scale[naux[A] + q];
                            }
                        }
                    }
                }

                ipair++;
                offa += sizea;
                offb += sizeb;
                free(storage);
                storage = NULL;
            }
        }
    }
    if (ipair != sizes[0] || offa != sizes[1] || offb != sizes[2]) status = -3;

cleanup:
    if (opt3 != NULL) CINTdel_optimizer(&opt3);
    free(storage);
    free(env);
    free(loc);
    free(bas);
    free(atm);
    return status;
}

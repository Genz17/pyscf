#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <cint.h>
#include <cint_funcs.h>
#include "gto/gto.h"
#include "vhf/fblas.h"

typedef struct {
    int A, B, nao_A, nao_B, naux_A, naux_B;
	double *T1, *ca, *cb;
	// ca of size nao_A * nao_B * naux_A
	// cb of size nao_A * nao_B * naux_B
} PARILoopPair;

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

    size_t ipair = 0, offa = 0, offb = 0;
    int middle = T1_num / 2;

    for (int it = 0; it <= middle; it++) {
        double *translation = T1 + (size_t)it * 3;
        int zero_image = (it == middle);

        for (int A = 0; A < mol_natm; A++) {
            for (int B = zero_image ? A : 0; B < mol_natm; B++) {
                int same;
                size_t sizea, sizeb;

                if (!_pari_pair_sizes(
                        A, B, zero_image, translation, coords,
                        nao, naux, pair_rcut, &same, &sizea, &sizeb)) {
                    continue;
                }

                pairs[ipair++] = (PARILoopPair) {
                    .A = A,
                    .B = B,
                    .nao_A = nao[A],
                    .nao_B = nao[B],
                    .naux_A = naux[A],
                    .naux_B = naux[B],
                    .T1 = translation,
                    .ca = ca + offa,
                    .cb = cb + offb
                };

                offa += sizea;
                offb += sizeb;
            }
        }
    }

    extern CINTIntegralFunction int3c2e_sph, int3c2e_cart;
    extern CINTOptimizerFunction int3c2e_optimizer;

    extern void dsyev_(const char *, const char *, const int *, double *,
                       const int *, double *, double *, const int *, int *);
    extern void dpotrf_(const char *, const int *, double *,
                        const int *, int *);

    size_t max_nbas = 2 * ((size_t)mol_nbas + aux_nbas);
    size_t nenv = 2 * ((size_t)mol_env_size + aux_env_size) + 12;
    const size_t npairs = ipair;

    #pragma omp parallel
    {
        /* Each thread allocates its own workspace. */
        int *atm = malloc(4 * ATM_SLOTS * sizeof(*atm));
        int *bas = malloc(max_nbas * BAS_SLOTS * sizeof(*bas));
        int *loc = malloc((max_nbas + 1) * sizeof(*loc));
        double *env = malloc(nenv * sizeof(*env));

        memcpy(env, _env, (size_t)mol_env_size * sizeof(*env));
        memcpy(env + mol_env_size, _env, (size_t)mol_env_size * sizeof(*env));
        memcpy(env + 2 * mol_env_size, _aux_env, (size_t)aux_env_size * sizeof(*env));
        memcpy(env + 2 * mol_env_size + aux_env_size, _aux_env, (size_t)aux_env_size * sizeof(*env));

        env[PTR_RANGE_OMEGA] = -omega;

        #pragma omp for schedule(dynamic, 1)
        for (size_t ip = 0; ip < npairs; ip++) {
            PARILoopPair *pair = pairs + ip;

            int A = pair->A;
            int B = pair->B;
            double *translation = pair->T1;

            /* Recover the image index from the stored T1 pointer. */
            int it = (int)((translation - T1) / 3);
            int same = (A == B && it == middle);

            int na = pair->nao_A;
            int nb = pair->nao_B;
            int nq = pair->naux_A + (same ? 0 : pair->naux_B);
            size_t npair_size = (size_t)na * nb;

            double *storage = NULL;
            CINTOpt *opt3 = NULL;

            /* A_0, B_T, aux(A_0), aux(B_T), with independent coordinate
             * pointers. In the same-center case each basis occurs once.
             * Original basis coefficients and environments are preserved.
             */

            int starts[4], ends[4];
            int env_offsets[4] = {0, mol_env_size, 2 * mol_env_size, 2 * mol_env_size + aux_env_size};
            int coord_start = 2 * mol_env_size + 2 * aux_env_size;

            int nbas = 0;
            for (int group = 0; group < 4; group++) {
                int atom = group % 2 ? B : A;
                int auxiliary = group >= 2;
                const int *source_atm = auxiliary ? _aux_atm : _atm;
                const int *source_bas = auxiliary ? _aux_bas : _bas;
                int source_nbas = auxiliary ? aux_nbas : mol_nbas;

                int *record = atm + group * ATM_SLOTS;
                memcpy(record, source_atm + (size_t)atom * ATM_SLOTS, ATM_SLOTS * sizeof(*record));

                record[PTR_COORD] = coord_start + group * 3;
                record[PTR_ZETA] += env_offsets[group];

                if (record[PTR_FRAC_CHARGE] != 0) {
                    record[PTR_FRAC_CHARGE] += env_offsets[group];
                }
                for (int xyz = 0; xyz < 3; xyz++) {
                    env[record[PTR_COORD] + xyz] = coords[(size_t)atom * 3 + xyz] + (group % 2 ? translation[xyz] : 0.);
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
                loc[ish + 1] = loc[ish] + (cart ? CINTcgto_cart(ish, bas) : CINTcgto_spheric(ish, bas));
            }

            CINTIntegralFunction *intor3 = cart ? int3c2e_cart : int3c2e_sph;
            int slices3[6] = {starts[0], ends[0], starts[1], ends[1], starts[2], ends[3]};

            size_t ncache = GTOmax_cache_size((int (*)())intor3, slices3, 3, atm, 4, bas, nbas, env);

            int npair = (int)npair_size;
            int lwork = 3 * nq;  /* DSYEV requires at least 3*nq-1. */
            size_t ncoeff = npair_size * nq;
            size_t nstorage = (size_t)nq * nq + 2 * ncoeff + 2 * nq + (size_t)lwork + ncache;

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

                        intor3(rhs + offset, dims3, shls, atm, 4, bas, nbas, env, opt3, cache);
                    }
                }
            }
            CINTdel_optimizer(&opt3);

            for (int q = 0; q < nq; q++) {
                double diagonal = metric[q + (size_t)nq * q];
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

            /* Solve X L^T = B_normalized. */
            dtrsm_("R", "L", "T", "N",
                   &npair, &nq, &one,
                   metric, &nq, rhs, &npair);

            /* Solve Y L = X. */
            dtrsm_("R", "L", "N", "N",
                   &npair, &nq, &one,
                   metric, &nq, rhs, &npair);

            for (int i = 0; i < na; i++) {
                for (int j = 0; j < nb; j++) {
                    size_t p = i + (size_t)na * j;
                    size_t output = (size_t)i * nb + j;

                    for (int q = 0; q < naux[A]; q++) {
                        pair->ca[output * naux[A] + q] = rhs[p + npair_size * q] / scale[q];
                    }

                    if (!same) {
                        for (int q = 0; q < naux[B]; q++) {
                            pair->cb[output * naux[B] + q] = rhs[p + npair_size * (naux[A] + q)] / scale[naux[A] + q];
                        }

                    }
                }
            }

            free(storage);
        }  /* End for (ip): one registered pair per iteration. */

        free(env);
        free(loc);
        free(bas);
        free(atm);
    }  /* End OpenMP parallel region. */

    return 0;
}

void PARIbuild_d_k(size_t npairs,
    			   const PARILoopPair *pairs,
    			   int naux, int ncol, 
				   const int *ao_offsets, const int *aux_offsets,
				   const int *image_ids, const int *reverse_ids,
    			   const double *kpt,
    			   const double complex *m,
    			   double complex *const *out) {

    double complex *d0 = out[0];

    for (size_t ip = 0; ip < npairs; ip++) {
        const PARILoopPair *pair = pairs + ip;

        int A = pair->A;
        int B = pair->B;
        int na = pair->nao_A;
        int nb = pair->nao_B;
        int nqa = pair->naux_A;
        int nqb = pair->naux_B;

        int a0 = ao_offsets[A];
        int b0 = ao_offsets[B];
        int qa0 = aux_offsets[A];
        int qb0 = aux_offsets[B];

        int image = image_ids[ip];
        int reverse = reverse_ids[ip];
        int same = (A == B && image == 0);

        const double *T1 = pair->T1;
        double theta = kpt[0] * T1[0] + kpt[1] * T1[1] + kpt[2] * T1[2];

        double complex phase = cos(theta) - I * sin(theta);
        double complex reverse_phase = conj(phase);

        double complex *dplus = out[image];
        double complex *dminus = out[reverse];

        for (int a = 0; a < na; a++) {
            const double complex *ma = m + (size_t)(a0 + a) * ncol;

            for (int b = 0; b < nb; b++) {

                const double complex *mb = m + (size_t)(b0 + b) * ncol;
                size_t ab = (size_t)a * nb + b;

                /* Contributions from coefficients on auxiliary atom A. */
                for (int q = 0; q < nqa; q++) {
                    double c = pair->ca[ab * nqa + q];

                    /* Forward: d[0][a,qa,i] += phase * ca[a,b,q] * m[b,i]. */
                    double complex *forward = d0 + ((size_t)(a0 + a) * naux + qa0 + q) * ncol;

                    double complex cf = phase * c;

                    for (int i = 0; i < ncol; i++) {
                        forward[i] += cf * mb[i];
                    }

                    if (!same) {
                        /*
                         * Reverse:
                         * d[-T1][b,qa,i] += conj(phase) * ca[a,b,q] * m[a,i].
                         */
                        double complex *backward = dminus + ((size_t)(b0 + b) * naux + qa0 + q) * ncol;

                        double complex cr = reverse_phase * c;
                        for (int i = 0; i < ncol; i++) {
                            backward[i] += cr * ma[i];
                        }
                    }
                }

                /* Same-atom, same-image pairs have no cb coefficients. */
                if (same) continue;

                /* Contributions from coefficients on auxiliary atom B. */
                for (int q = 0; q < nqb; q++) {
                    double c = pair->cb[ab * nqb + q];

                    /* Forward: d[T1][a,qb,i] += phase * cb[a,b,q] * m[b,i]. */
                    double complex *forward = dplus + ((size_t)(a0 + a) * naux + qb0 + q) * ncol;

                    /*
                     * Reverse:
                     * d[0][b,qb,i] += conj(phase) * cb[a,b,q] * m[a,i].
                     */
                    double complex *backward = d0 + ((size_t)(b0 + b) * naux + qb0 + q) * ncol;

                    double complex cf = phase * c;
                    double complex cr = reverse_phase * c;

                    for (int i = 0; i < ncol; i++) {
                        forward[i] += cf * mb[i];
                        backward[i] += cr * ma[i];
                    }
                }
            }
        }
    }
}

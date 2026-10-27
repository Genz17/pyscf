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

extern CINTIntegralFunction int3c2e_sph;
extern CINTOptimizerFunction int3c2e_optimizer;

extern void GTOnr3c_drv(
    int (*intor)(), void (*fill)(), double *eri, int comp,
    int *shls_slice, int *ao_loc, CINTOpt *cintopt,
    int *atm, int natm, int *bas, int nbas, double *env);

extern void GTOnr3c_fill_s1(
    int (*intor)(), double *out, double *buf, int comp, int jobid,
    int *shls_slice, int *ao_loc, CINTOpt *cintopt,
    int *atm, int natm, int *bas, int nbas, double *env);

extern void dpotrf_(const char *uplo, const int *n,
                   double *a, const int *lda, int *info);

int fit_pair(double *ints, const double *metric, const int *aux_idx,
             int na, int nb, int nq, int naux) {

	// ints: (\mu\nu|P) for the A, B atoms
	// metric: (P|Q) overlap matrix
	// aux_idx: index of aux function in the aux_mol

	// na: nao on atom A
	// nb: nao on atom B
	// nq: number of aux functions on atom A and atom B
	// naux: number of aux functions in total

    int npair = na * nb;
    double *vlocal = malloc((size_t)nq * nq * sizeof(double));

	// C -> F order: metric is in C order
    for (int s = 0; s < nq; s++) {
        for (int r = 0; r < nq; r++) {
            vlocal[r + (size_t)nq * s] =
                metric[(size_t)aux_idx[r] * naux + aux_idx[s]];
        }
    }

    int info;
    dpotrf_("L", &nq, vlocal, &nq, &info); // this is the LU decomposition for the aux block.

    if (info == 0) {
        double one = 1.0;

        /* ints <- ints * L^{-T} */
        dtrsm_("R", "L", "T", "N", &npair, &nq, &one,
               vlocal, &nq, ints, &npair);

        /* ints <- ints * L^{-1} */
        dtrsm_("R", "L", "N", "N", &npair, &nq, &one,
               vlocal, &nq, ints, &npair);
    }

    free(vlocal);
    return info;
}

static void int3c2e_masked(double *out, int *shls_slice, int *loc, CINTOpt *opt,
    					   int *atm, int natm, int *bas, int nbas, double *env,
    					   const uint8_t *mask, int mol_nbas,
						   double *ao_mnmn, double *aux_PP, double tol)
{
    int dims[3] = {loc[shls_slice[1]] - loc[shls_slice[0]],
                   loc[shls_slice[3]] - loc[shls_slice[2]],
                   loc[shls_slice[5]] - loc[shls_slice[4]]};
    size_t n_munu = (size_t)dims[0] * dims[1];

    memset(out, 0, n_munu * dims[2] * sizeof(*out));

    size_t ncache = GTOmax_cache_size(int3c2e_sph, shls_slice, 3,
                                    atm, natm, bas, nbas, env);
    double *cache = malloc(ncache * sizeof(*cache));

    for (int j = shls_slice[2]; j < shls_slice[3]; j++) {
        for (int i = shls_slice[0]; i < shls_slice[1]; i++) {
            if (!mask[(size_t)i * mol_nbas + j]) continue;

            size_t ij = (loc[i] - loc[shls_slice[0]]) + (size_t)dims[0] * (loc[j] - loc[shls_slice[2]]);

            for (int k = shls_slice[4]; k < shls_slice[5]; k++) {

				if (ao_mnmn[i * mol_nbas + j] * aux_PP[k-mol_nbas] < tol) {
					continue;
				}


                int shls[3] = {i, j, k};
                size_t offset = ij + n_munu * (loc[k] - loc[shls_slice[4]]);

                int3c2e_sph(out + offset, dims, shls,
                            atm, natm, bas, nbas, env, opt, cache);
            }
        }
    }
    free(cache);
}

int _int3c2e_masked_parallelized(
    double *out, int *shls_slice, int *loc, CINTOpt *opt,
    int *atm, int natm, int *bas, int nbas, double *env,
    const uint8_t *mask, int mol_nbas,
	double *ao_mnmn, double *aux_PP,
	double tol)
{

    int nao = loc[shls_slice[1]] - loc[shls_slice[0]];
    int nq = loc[shls_slice[5]] - loc[shls_slice[4]];
    size_t n_munu = (size_t)nao * (nao + 1) / 2;

    //memset(out, 0, n_munu * nq * sizeof(*out));

    size_t ncache = GTOmax_cache_size(int3c2e_sph, shls_slice, 3, atm, natm, bas, nbas, env);
    int max_ao = GTOmax_shell_dim(loc, shls_slice, 2);
    int max_aux = GTOmax_shell_dim(loc, shls_slice + 4, 1);
    size_t nbuf = (size_t)max_ao * max_ao * max_aux;

    #pragma omp parallel 
    {
        double *buf = malloc((nbuf + ncache) * sizeof(*buf));
        double *cache = buf + nbuf;

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int j = shls_slice[2]; j < shls_slice[3]; j++) {
            for (int i = shls_slice[0]; i < shls_slice[1]; i++) {

                if (i < j || !mask[(size_t)i * mol_nbas + j])
                    continue;

                int di = loc[i+1] - loc[i];
                int dj = loc[j+1] - loc[j];
                int i0 = loc[i] - loc[shls_slice[0]];
                int j0 = loc[j] - loc[shls_slice[2]];

                for (int k = shls_slice[4]; k < shls_slice[5]; k++) {
					
					if (ao_mnmn[i * mol_nbas + j] * aux_PP[k-mol_nbas] < tol) {
						continue;
					}

                    int shls[3] = {i, j, k};

                    if (!int3c2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                        continue;
                    }

                    int dk = loc[k+1] - loc[k];
                    int q0 = loc[k] - loc[shls_slice[4]];

                    for (int q = 0; q < dk; q++) {
                        for (int a = 0; a < di; a++) {
                            size_t mu = i0 + a;
                            size_t p = mu * (mu+1) / 2 + j0;
                            int count = i == j ? a+1 : dj;

                            for (int b = 0; b < count; b++) {
                                out[p+b + n_munu * (q0+q)] = buf[a + (size_t)di * (b + dj*q)];
                            }
                        }
                    }
					

                }
            }
        }
        free(buf);
    }
    return 0;
}


int _atom_pair_screen(int *keep_pair, const uint8_t *schwarz_mask,
					  int *ao_slice,
					  int mol_natm, int mol_nbas) {

	#pragma omp parallel for collapse(2) schedule(dynamic, 1)
	for (int A = 0; A < mol_natm; A++) {
		for (int B = 0; B < mol_natm; B++) {
			if (B < A) continue;
			int count = A * (mol_natm + mol_natm - A + 1) / 2 + B - A;
			
			for (int i = ao_slice[4*A]; i < ao_slice[4*A+1] && !keep_pair[count]; i++) {
				for (int j = ao_slice[4*B]; j < ao_slice[4*B+1]; j++) {
					if (schwarz_mask[(size_t)i * mol_nbas + j]) {
						keep_pair[count] = 1;
						break;
					}
				}
			}
		}
	}

	return 0;

}


int _local_coeff(double *aux_QP, double *cab, int *cab_offset, int cab_per_pair_size,
				 int *atm_conc, int *bas_conc, double *env_conc, int *ao_loc_conc,
				 const uint8_t *schwarz_mask, int *keep_pair, int *keep_pair_cumsum,
				 int *ao_slice, int *aux_slice,
				 int mol_natm, int mol_nbas, 
				 int conc_natm, int conc_nbas,
				 int nao, int naux, 
				 CINTOpt *opt, double tol,
				 double *ao_mnmn, double *aux_PP) {


	//CINTOpt *opt = NULL;
	//int3c2e_optimizer(&opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc);

	// npair: natm * (natm + 1) / 2
	// cab is of size npair * (max_atm nao) * (max_atm nao)
	// cab_offset is of size npair


	#pragma omp parallel for collapse(2) schedule(dynamic, 1)
	for (int A = 0; A < mol_natm; A++) {
		for (int B = 0; B < mol_natm; B++) {
			if (B < A) continue;
			int count = A * (mol_natm + mol_natm - A + 1) / 2 + B - A;
			
			//int keep_pair = 0;

			//for (int i = ao_slice[4*A]; i < ao_slice[4*A+1] && !keep_pair; i++) {
			//	for (int j = ao_slice[4*B]; j < ao_slice[4*B+1]; j++) {
			//		if (schwarz_mask[(size_t)i * mol_nbas + j]) {
			//			keep_pair = 1;
			//			break;
			//		}
			//	}
			//}

			if (!keep_pair[count]) {
				cab_offset[count] = 0;
				continue;
			}


			int ao_A = ao_slice[4*A+3] - ao_slice[4*A+2];
			int ao_B = ao_slice[4*B+3] - ao_slice[4*B+2];


			if (A == B) {
				int shls_slice[6] = {ao_slice[4*A], ao_slice[4*A+1],
				   				 	 ao_slice[4*B], ao_slice[4*B+1],
									 mol_nbas + aux_slice[4*A], mol_nbas + aux_slice[4*A+1]};

				int ao_aux = aux_slice[4*A+3] - aux_slice[4*A+2];
				int *aux_idx = malloc(sizeof(int) * ao_aux);
				for (int ii=0; ii < ao_aux; ii++) {
					aux_idx[ii] = aux_slice[4*A+2] + ii;
				}	

				double *out = malloc((size_t)ao_A * ao_B * ao_aux * sizeof(double)); // (\mu\nu| Q) eris, sorted by ao inx


				int3c2e_masked(out, shls_slice, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc, schwarz_mask, mol_nbas,
							   ao_mnmn, aux_PP, tol);

				int info = fit_pair(out, aux_QP, aux_idx, ao_A, ao_B, ao_aux, naux);
				
				cab_offset[count] = ao_A * ao_B * ao_aux;

				for (int jj=0; jj < ao_A * ao_B * ao_aux; jj++) {
					cab[(size_t)cab_per_pair_size*(keep_pair_cumsum[count]-1) + jj] = out[jj]; // cab: F order, (na, nb, np)
				}

				free(out);
				free(aux_idx);

			} else {

				int shls_slice_A[6] = {ao_slice[4*A], ao_slice[4*A+1],
									 ao_slice[4*B], ao_slice[4*B+1],
									 mol_nbas + aux_slice[4*A], mol_nbas + aux_slice[4*A+1]};

				int shls_slice_B[6] = {ao_slice[4*A], ao_slice[4*A+1],
									 ao_slice[4*B], ao_slice[4*B+1],
									 mol_nbas + aux_slice[4*B], mol_nbas + aux_slice[4*B+1]};

				int ao_aux_A = aux_slice[4*A+3] - aux_slice[4*A+2];
				int ao_aux_B = aux_slice[4*B+3] - aux_slice[4*B+2];
				int ao_aux = ao_aux_A + ao_aux_B;

				int *aux_idx = malloc(sizeof(int) * (ao_aux_A + ao_aux_B));

				for (int ii=0; ii < ao_aux_A + ao_aux_B; ii++) {
					if (ii < ao_aux_A) {
						aux_idx[ii] = aux_slice[4*A+2] + ii;
					} else {
						aux_idx[ii] = aux_slice[4*B+2] + ii - ao_aux_A;
					}
				}	

				double *out_A = malloc((size_t)ao_A * ao_B * ao_aux_A * sizeof(double)); // (\mu\nu| Q) eris, sorted by ao inx
				double *out_B = malloc((size_t)ao_A * ao_B * ao_aux_B * sizeof(double)); // (\mu\nu| Q) eris, sorted by ao inx
				double *out = malloc((size_t)ao_A * ao_B * ao_aux * sizeof(double));

				int3c2e_masked(out_A, shls_slice_A, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc, schwarz_mask, mol_nbas,
						ao_mnmn, aux_PP, tol);
				int3c2e_masked(out_B, shls_slice_B, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc, schwarz_mask, mol_nbas,
						ao_mnmn, aux_PP, tol);

				size_t nA = (size_t)ao_A * ao_B * ao_aux_A;
				size_t nB = (size_t)ao_A * ao_B * ao_aux_B;
				memcpy(out,      out_A, nA * sizeof(double));
				memcpy(out + nA, out_B, nB * sizeof(double));

				int info = fit_pair(out, aux_QP, aux_idx, ao_A, ao_B, ao_aux, naux);
				cab_offset[count] = ao_A * ao_B * ao_aux;

				for (int jj=0; jj < ao_A * ao_B * ao_aux; jj++) {
					cab[(size_t)cab_per_pair_size* (keep_pair_cumsum[count]-1) + jj] = out[jj]; // cab: F order, (na, nb, np)
				}

				free(out); 
				free(out_A);
				free(out_B);
				free(aux_idx);

			}
		}
	}
	//CINTdel_optimizer(&opt);
	return 0;
}

//
//int _loop_k_wdbuild(double *metric_block, 
//				   double *cab, int cab_per_pair_size, int *cab_offset,
//				   double *m, double *w, double *d, 
//				   int mol_natm, int nocc,
//				   int *ao_slice, int *aux_slice, int naux, int nao,
//				   int aux_q0, int aux_q1) {
//
//	#pragma omp parallel for collapse(2) schedule(dynamic, 1)
//	for (int A = 0; A < mol_natm; A++) {
//		for (int B = 0; B < mol_natm; B++) {
//			if (B < A) continue;
//
//			int count = A * (mol_natm + mol_natm - A + 1) / 2 + B - A;
//			size_t shift = (size_t)count * cab_per_pair_size;
//			double *cabnow = cab + shift;
//			int ao_A = ao_slice[4*A+3] - ao_slice[4*A+2];
//			int ao_B = ao_slice[4*B+3] - ao_slice[4*B+2];
//
//			int ao_aux;
//			int *aux_idx;
//
//			if (A == B) {
//				ao_aux = aux_slice[4*A+3] - aux_slice[4*A+2];
//				aux_idx = malloc(sizeof(int) * ao_aux);
//				for (int ii=0; ii < ao_aux; ii++) {
//					aux_idx[ii] = aux_slice[4*A+2] + ii;
//				}	
//			} else {
//				int ao_aux_A = aux_slice[4*A+3] - aux_slice[4*A+2];
//				int ao_aux_B = aux_slice[4*B+3] - aux_slice[4*B+2];
//				ao_aux = ao_aux_A + ao_aux_B;
//
//				aux_idx = malloc(sizeof(int) * (ao_aux_A + ao_aux_B));
//
//				for (int ii=0; ii < ao_aux_A + ao_aux_B; ii++) {
//					if (ii < ao_aux_A) {
//						aux_idx[ii] = aux_slice[4*A+2] + ii;
//					} else {
//						aux_idx[ii] = aux_slice[4*B+2] + ii - ao_aux_A;
//					}
//				}	
//			}
//
//			int nq = aux_q1 - aux_q0;
//			int a0 = ao_slice[4*A+2];
//			int b0 = ao_slice[4*B+2];
//
//			for (int a = 0; a < ao_A; a++) {
//				for (int b = 0; b < ao_B; b++) {
//					for (int aa = 0; aa < ao_aux; aa++) {
//
//						double c = -0.5 * cabnow[a + (size_t)ao_A * (b + (size_t)ao_B * aa)];
//
//						if (aux_idx[aa] >= aux_q0 && aux_idx[aa] < aux_q1) {
//							size_t qoff = (size_t)(aux_idx[aa] - aux_q0) * nao * nocc;
//
//							for (int iocc = 0; iocc < nocc; iocc++) {
//								size_t ia = (size_t)(a0 + a) * nocc + iocc;
//								size_t ib = (size_t)(b0 + b) * nocc + iocc;
//
//								#pragma omp atomic update
//								d[qoff + ia] += -2.0 * c * m[ib];
//
//								if (A != B) {
//									#pragma omp atomic update
//									d[qoff + ib] += -2.0 * c * m[ia];
//								}
//							}
//						}
//
//
//						for (int q = 0; q < nq; q++) {
//							double temp = c * metric_block[(size_t)aux_idx[aa] * naux + q];
//
//
//							w[((size_t)q * nao + a0 + a) * nao + b0 + b] += temp;
//
//							if (A != B) {
//								w[((size_t)q * nao + b0 + b) * nao + a0 + a] += temp;
//							}
//
//
//						}
//					}
//				}
//			}
//
//			free(aux_idx);
//
//		}
//	}
//
//
//	return 0;
//}

typedef struct {
    const double *coeff;
    int a0, b0, na, nb, nab;
    int p0, np, r0, nr;
} PARILoopPair;

int _loop_k_wdbuild(double *metric_block,
                   double *cab, int cab_per_pair_size, int *cab_offset,
                   double *m, double *w, double *d, double *h,
                   int mol_natm, int nocc,
                   int *ao_slice, int *aux_slice, int naux, int nao,
                   int aux_q0, int aux_q1) {

    const int nq = aux_q1 - aux_q0; // number of aux AOs that we want to handle here in each loop

	// atom pair capacity (A, B)
    const size_t capacity = (size_t)mol_natm * ((size_t)mol_natm + 1) / 2;

    PARILoopPair *pairs = malloc(capacity * sizeof(*pairs));

    size_t count = 0, used = 0;
    int max_nab = 0, max_nlocal = 0;

	// max_nab: largest (\mu, \nu) pair for one atom pair
	// max_nlocal: biggest possible aux ao associated with one atom pair

	// pair construction
    for (int A = 0; A < mol_natm; A++) {
        for (int B = A; B < mol_natm; B++, count++) {

			if (cab_offset[count] == 0) {
				continue;
			}

            PARILoopPair pair;

            pair.a0 = ao_slice[4*A+2];
            pair.b0 = ao_slice[4*B+2];
            pair.na = ao_slice[4*A+3] - pair.a0;
            pair.nb = ao_slice[4*B+3] - pair.b0;

            pair.p0 = aux_slice[4*A+2];
            pair.r0 = aux_slice[4*B+2];
            pair.np = aux_slice[4*A+3] - pair.p0;
            pair.nr = A == B ? 0 : aux_slice[4*B+3] - pair.r0;

            const int nlocal = pair.np + pair.nr;

            pair.nab = pair.na * pair.nb;

            pair.coeff = cab + used * (size_t)cab_per_pair_size;
            pairs[used++] = pair;

            if (pair.nab > max_nab) max_nab = pair.nab;
            if (nlocal > max_nlocal) max_nlocal = nlocal;
        }
    }

    int nthreads = omp_get_max_threads();
    if (nthreads > nq) nthreads = nq;

    int tile = 1 + (nq - 1) / nthreads; // tile = ceil(nq/nthreads), s.t. nthreads * tile >= nq

    const size_t metric_size = (size_t)max_nlocal * tile;
    const size_t scratch_size = fmax(1, metric_size + (size_t)max_nab * tile);

    double *scratch = malloc((size_t)nthreads * scratch_size * sizeof(double));

	#pragma omp parallel num_threads(nthreads)
    {
        double *vlocal = scratch + (size_t)omp_get_thread_num() * scratch_size; // V = (P|Q), where Q is specified by the current thread
        double *correction = vlocal + metric_size; // metric = (\mu \nu|Q) where Q is specified by the current thread

        const double one = 1.0;
        const double zero = 0.0;
        const double minus_half = -0.5;

		#pragma omp for schedule(dynamic, 1)
        for (int qb = 0; qb < nq; qb += tile) {
			// each thread takes one qb, and run over that tile

            const int nt = nq - qb < tile ? nq - qb : tile; // to deal with the last portion

            for (size_t ipair = 0; ipair < used; ipair++) {

                const PARILoopPair *pair = pairs + ipair;
                const int na = pair->na;
                const int nb = pair->nb;
                const int nab = pair->nab;
                const int nlocal = pair->np + pair->nr;

                
                //Pack V[P,q] into column-major (nlocal, nt).
                 
                for (int iq = 0; iq < nt; iq++) {
					// iq: specified the current aux ao
                    double *column = vlocal + (size_t)iq * nlocal;

                    for (int p = 0; p < pair->np; p++) {
						column[p] = metric_block[(size_t)(pair->p0 + p) * nq + qb + iq];
                    }

                    for (int p = 0; p < pair->nr; p++) {
                        column[pair->np + p] = metric_block[(size_t)(pair->r0 + p) * nq + qb + iq];
                    }
                }

                /*
                 * correction[ab,q] = -0.5 * C[ab,P] * V[P,q]
                 *
                 * C:          column-major (nab, nlocal)
                 * V:          column-major (nlocal, nt)
                 * correction: column-major (nab, nt)
                 */

                dgemm_("N", "N",
                       &nab, &nt, &nlocal,
                       &minus_half,
                       pair->coeff, &nab,
                       vlocal, &nlocal,
                       &zero,
                       correction, &nab);

				// apply correction
                for (int iq = 0; iq < nt; iq++) {
                    const int q = qb + iq; // this is the index within the input aux_q0 to aux_q1
                    const double *wc = correction + (size_t)iq * nab;
                    double *wq = w + (size_t)q * nao * nao;

                    /* Scatter the pair correction into row-major w. */
                    for (int a = 0; a < na; a++) {
                        for (int b = 0; b < nb; b++) {
                            const double value = wc[a + (size_t)na * b];

                            wq[(size_t)(pair->a0 + a) * nao + pair->b0 + b] += value;

                            if (pair->a0 != pair->b0) {
                                wq[(size_t)(pair->b0 + b) * nao + pair->a0 + a] += value;
                            }
                        }
                    }

                    /*
                     * d receives a contribution only if this global q
                     * belongs to the pair's local auxiliary basis.
                     */
                    const int global_q = aux_q0 + q;
                    int local_q;

                    if (global_q >= pair->p0 && global_q < pair->p0 + pair->np) {
                        local_q = global_q - pair->p0;
                    } else if (global_q >= pair->r0 && global_q < pair->r0 + pair->nr) {
                        local_q = pair->np + global_q - pair->r0;
                    } else {
                        continue;
                    }

                    const double *cq = pair->coeff + (size_t)local_q * nab;
                    double *dq = d + (size_t)q * nao * nocc;

                    /*
                     * Cq is column-major (na, nb).
                     * Row-major M and D are viewed as column-major
                     * matrices with shape (nocc, nao).
                     *
                     * D_A^T += M_B^T * Cq^T
                     */
                    dgemm_("N", "T",
                           &nocc, &na, &nb,
                           &one,
                           m + (size_t)pair->b0 * nocc, &nocc,
                           cq, &na,
                           &one,
                           dq + (size_t)pair->a0 * nocc, &nocc);

                    /* D_B^T += M_A^T * Cq */
                    if (pair->a0 != pair->b0) {
                        dgemm_("N", "N",
                               &nocc, &nb, &na,
                               &one,
                               m + (size_t)pair->a0 * nocc, &nocc,
                               cq, &na,
                               &one,
                               dq + (size_t)pair->b0 * nocc, &nocc);
                    }
                }
            }

			const int ncols = nt * nao;

			dgemm_("N", "N",
			   &nocc, &ncols, &nao,
			   &one,
			   m, &nocc,
			   w + (size_t)qb * nao * nao, &nao,
			   &zero,
			   h + (size_t)qb * nao * nocc, &nocc);

        }
    }

    free(scratch);
    free(pairs);
    return 0;
}

int _loop_j_bbuild(
    double *b, const double *dm, const double *ao_mnmn,
    const double *aux_bound, const int *ao_slice, const int *ao_loc,
    int mol_natm, int mol_nbas,
    int *atm, int natm, int *bas, int nbas, double *env,
    CINTOpt *opt, double tol)

    // b = (\lambda \sigma|Q) D_{\lambda \sigma}, Q: all aux AOs

{
    const int nao = ao_loc[mol_nbas];
    const int naux = ao_loc[nbas] - nao;
    const int naux_shells = nbas - mol_nbas;

    double *bounds = malloc((size_t)mol_natm * mol_natm * sizeof(double));

    /* Compute each atom-pair screening bound once. */
    for (int A = 0; A < mol_natm; A++) {
        for (int B = A; B < mol_natm; B++) {
            double qab = 0.0, dm_sum = 0.0;

            for (int i = ao_slice[4*A]; i < ao_slice[4*A+1]; i++)
                for (int j = ao_slice[4*B]; j < ao_slice[4*B+1]; j++)
                    qab = fmax(qab, ao_mnmn[(size_t)i * mol_nbas + j]);

            for (int i = ao_slice[4*A+2]; i < ao_slice[4*A+3]; i++)
                for (int j = ao_slice[4*B+2]; j < ao_slice[4*B+3]; j++)
                    dm_sum += fabs(dm[(size_t)i * nao + j]);

            bounds[(size_t)A * mol_natm + B] = (A == B ? 1.0 : 2.0) * qab * dm_sum;
        }
    }

    /* Allocate one integral buffer and libcint cache per thread. */
    int max_ao = 0, max_aux = 0;
    for (int s = 0; s < mol_nbas; s++) {
        int dim = ao_loc[s+1] - ao_loc[s];
        if (dim > max_ao) max_ao = dim;
    }
    for (int s = mol_nbas; s < nbas; s++) {
        int dim = ao_loc[s+1] - ao_loc[s];
        if (dim > max_aux) max_aux = dim;
    }

    int slices[6] = {0, mol_nbas, 0, mol_nbas, mol_nbas, nbas};
    size_t cache_size = GTOmax_cache_size(int3c2e_sph, slices, 3, atm, natm, bas, nbas, env);
    size_t buf_size = (size_t)max_ao * max_ao * max_aux;
    size_t stride = buf_size + cache_size;

    int nthreads = omp_get_max_threads();
    if (nthreads > naux_shells) nthreads = naux_shells;

    double *work = malloc((size_t)nthreads * stride * sizeof(double));

    #pragma omp parallel num_threads(nthreads)
    {
        double *buf = work + (size_t)omp_get_thread_num() * stride;
        double *cache = buf + buf_size;

        #pragma omp for schedule(dynamic, 1)
        for (int k = mol_nbas; k < nbas; k++) {
            int dk = ao_loc[k+1] - ao_loc[k];
            double *bk = b + ao_loc[k] - nao;
            double qbound = aux_bound[k - mol_nbas];

            for (int A = 0; A < mol_natm; A++) {
                for (int B = A; B < mol_natm; B++) {
                    if (bounds[(size_t)A * mol_natm + B] * qbound < tol)
                        continue;

                    double factor = A == B ? 1.0 : 2.0;

                    for (int j = ao_slice[4*B]; j < ao_slice[4*B+1]; j++) {
                        int dj = ao_loc[j+1] - ao_loc[j];

                        for (int i = ao_slice[4*A]; i < ao_slice[4*A+1]; i++) {
                            int di = ao_loc[i+1] - ao_loc[i];
                            int shls[3] = {i, j, k};

                            if (!int3c2e_sph(buf, NULL, shls, atm, natm, bas, nbas, env, opt, cache)) {
                                continue;
							}

                            /* Integral layout: (di, dj, dk), Fortran order. */
                            for (int q = 0; q < dk; q++) {
                                double value = 0.0;
                                for (int jj = 0; jj < dj; jj++) {
                                    for (int ii = 0; ii < di; ii++) {
                                        value += dm[(size_t)(ao_loc[i]+ii) * nao + ao_loc[j]+jj] * buf[ii + (size_t)di * (jj + dj*q)];
									}
								}
                                bk[q] += factor * value;
                            }
                        }
                    }
                }
            }
        }
    }

    free(work);
    free(bounds);
    return 0;
}

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
    					   const uint8_t *mask, int mol_nbas)
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
                int shls[3] = {i, j, k};
                size_t offset = ij + n_munu * (loc[k] - loc[shls_slice[4]]);

                int3c2e_sph(out + offset, dims, shls,
                            atm, natm, bas, nbas, env, opt, cache);
            }
        }
    }
    free(cache);
}


int _local_coeff(double *aux_QP, double *cab, int *cab_offset, int cab_per_pair_size,
				 int *atm_conc, int *bas_conc, double *env_conc, int *ao_loc_conc,
				 const uint8_t *schwarz_mask,
				 int *ao_slice, int *aux_slice,
				 int mol_natm, int mol_nbas, 
				 int conc_natm, int conc_nbas,
				 int nao, int naux) {


	CINTOpt *opt = NULL;
	int3c2e_optimizer(&opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc);

	// npair: natm * (natm + 1) / 2
	// cab is of size npair * (max_atm nao) * (max_atm nao)
	// cab_offset is of size npair


#pragma omp parallel for collapse(2) schedule(dynamic, 1)
	for (int A = 0; A < mol_natm; A++) {
		for (int B = 0; B < mol_natm; B++) {
			if (B < A) continue;
			int count = A * (mol_natm + mol_natm - A + 1) / 2 + B - A;
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

				//GTOnr3c_drv(int3c2e_sph, GTOnr3c_fill_s1, out, 1, shls_slice, ao_loc_conc, opt,
				//			atm_conc, conc_natm, bas_conc, conc_nbas, env_conc);


				int3c2e_masked(out, shls_slice, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc, schwarz_mask, mol_nbas);

				int info = fit_pair(out, aux_QP, aux_idx, ao_A, ao_B, ao_aux, naux);
				
				cab_offset[count] = ao_A * ao_B * ao_aux;

				for (int jj=0; jj < ao_A * ao_B * ao_aux; jj++) {
					cab[(size_t)cab_per_pair_size*count + jj] = out[jj]; // cab: F order, (na, nb, np)
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

				//GTOnr3c_drv(int3c2e_sph, GTOnr3c_fill_s1, out_A, 1, shls_slice_A, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc);
				//GTOnr3c_drv(int3c2e_sph, GTOnr3c_fill_s1, out_B, 1, shls_slice_B, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc);
				int3c2e_masked(out_A, shls_slice_A, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc, schwarz_mask, mol_nbas);
				int3c2e_masked(out_B, shls_slice_B, ao_loc_conc, opt, atm_conc, conc_natm, bas_conc, conc_nbas, env_conc, schwarz_mask, mol_nbas);

				size_t nA = (size_t)ao_A * ao_B * ao_aux_A;
				size_t nB = (size_t)ao_A * ao_B * ao_aux_B;
				memcpy(out,      out_A, nA * sizeof(double));
				memcpy(out + nA, out_B, nB * sizeof(double));

				int info = fit_pair(out, aux_QP, aux_idx, ao_A, ao_B, ao_aux, naux);
				cab_offset[count] = ao_A * ao_B * ao_aux;

				for (int jj=0; jj < ao_A * ao_B * ao_aux; jj++) {
					cab[(size_t)cab_per_pair_size*count + jj] = out[jj]; // cab: F order, (na, nb, np)
				}

				free(out); 
				free(out_A);
				free(out_B);
				free(aux_idx);

			}
		}
	}
	CINTdel_optimizer(&opt);
	return 0;
}


int _loop_k_wbuild(double *metric_block, 
				   double *cab, int cab_per_pair_size, int *cab_offset,
				   double *w, int mol_natm,
				   int *ao_slice, int *aux_slice, int naux, int nao,
				   int aux_q0, int aux_q1) {

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
	for (int A = 0; A < mol_natm; A++) {
		for (int B = 0; B < mol_natm; B++) {
			if (B < A) continue;

			int count = A * (mol_natm + mol_natm - A + 1) / 2 + B - A;
			size_t shift = (size_t)count * cab_per_pair_size;
			double *cabnow = cab + shift;
			int ao_A = ao_slice[4*A+3] - ao_slice[4*A+2];
			int ao_B = ao_slice[4*B+3] - ao_slice[4*B+2];

			int ao_aux;
			int *aux_idx;

			if (A == B) {
				ao_aux = aux_slice[4*A+3] - aux_slice[4*A+2];
				aux_idx = malloc(sizeof(int) * ao_aux);
				for (int ii=0; ii < ao_aux; ii++) {
					aux_idx[ii] = aux_slice[4*A+2] + ii;
				}	
			} else {
				int ao_aux_A = aux_slice[4*A+3] - aux_slice[4*A+2];
				int ao_aux_B = aux_slice[4*B+3] - aux_slice[4*B+2];
				ao_aux = ao_aux_A + ao_aux_B;

				aux_idx = malloc(sizeof(int) * (ao_aux_A + ao_aux_B));

				for (int ii=0; ii < ao_aux_A + ao_aux_B; ii++) {
					if (ii < ao_aux_A) {
						aux_idx[ii] = aux_slice[4*A+2] + ii;
					} else {
						aux_idx[ii] = aux_slice[4*B+2] + ii - ao_aux_A;
					}
				}	
			}

			int nq = aux_q1 - aux_q0;
			int a0 = ao_slice[4*A+2];
			int b0 = ao_slice[4*B+2];

			for (int a = 0; a < ao_A; a++) {
				for (int b = 0; b < ao_B; b++) {
					for (int aa = 0; aa < ao_aux; aa++) {
						double c = -0.5 * cabnow[
							a + (size_t)ao_A * (b + (size_t)ao_B * aa)
						];

						for (int q = 0; q < nq; q++) {
							double temp = c * metric_block[
								(size_t)aux_idx[aa] * naux + q
							];

							w[((size_t)q * nao + a0 + a) * nao + b0 + b]
								+= temp;

							if (A != B) {
								w[((size_t)q * nao + b0 + b) * nao + a0 + a]
									+= temp;
							}
						}
					}
				}
			}

			free(aux_idx);

		}
	}


	return 0;
}

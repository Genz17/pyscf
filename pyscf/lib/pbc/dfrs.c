#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "cint.h"
#include "gto/ft_ao.h"
#include "pbc/pbc.h"
#include "vhf/fblas.h"

#define MAX(x, y)       ((x) > (y) ? (x) : (y))
#define MIN(x, y)       ((x) < (y) ? (x) : (y))

/* C version of dfrs._get_k for one restricted density matrix.
 * All arrays are C-contiguous:
 *
 * cderi[nk,naux,npair]  B[t,Q,p]
 * dm[nk,nao,nao]       P[d,lambda,sigma] in real space
 * pairs[npair,3]       (mu,nu,m) of dfrs._pair_list
 * diff[nk,nk]          diff[s,t] = t-s modulo the BvK supercell
 * vk[nk,nao,nao]       output K[l,mu,nu], overwritten
 * blockdim              auxiliary batch size, >= 1
 *
 * Each OpenMP task owns one AO row mu of K. Its intermediate is
 * T[s,sigma,Q] for one auxiliary cell t and a batch of Q, so no reduction
 * or atomic updates are needed. 
 */

void _get_k(double complex *vk, const double complex *cderi,
            const double complex *dm, const int *pairs, const int *diff,
            int nk, int naux, int nao, int npair, int blockdim)
{
    memset(vk, 0, (size_t)nk * nao * nao * sizeof(*vk));

    /* blockdim counts combined (t,Q) entries; must be >= 1. */
    const size_t nfit = (size_t)nk * naux;
    blockdim = (int)MIN((size_t)blockdim, nfit);

#pragma omp parallel
    {
        double complex *buf = malloc(
            (size_t)nk * nao * blockdim * sizeof(*buf));

#pragma omp for schedule(dynamic)
        for (int mu = 0; mu < nao; mu++) {
            for (size_t r0 = 0; r0 < nfit; r0 += blockdim) {
                int n = (int)MIN((size_t)blockdim, nfit - r0);
                memset(buf, 0, (size_t)nk * nao * n * sizeof(*buf));

                /*
                 * T[s,sigma,r] =
                 *   sum_(lambda,m) P[s-m,lambda,sigma] conj(B[t,Q,p])
                 *
                 * r = t*naux + Q
                 */
                for (int p = 0; p < npair; p++) {
                    if (pairs[3*p] != mu) {
                        continue;
                    }

                    int lam = pairs[3*p+1];
                    int m = pairs[3*p+2];

                    /* Consecutive r values already follow this layout. */
                    const double complex *b = cderi + r0*npair + p;

                    for (int s = 0; s < nk; s++) {
                        const double complex *d = dm + ((size_t)diff[m*nk+s]*nao + lam)*nao;

                        for (int sig = 0; sig < nao; sig++) {
                            double complex *v = buf + ((size_t)s*nao + sig)*n;

                            for (int a = 0; a < n; a++) {
                                v[a] += d[sig] * conj(b[(size_t)a*npair]);
                            }
                        }
                    }
                }

                /*
                 * K[s+m,mu,nu] +=
                 *   sum_(t,Q) T[s,sigma,t,Q] B[t-s,Q,p]
                 */
                for (int p = 0; p < npair; p++) {
                    int sig = pairs[3*p];
                    int nu = pairs[3*p+1];
                    int m = pairs[3*p+2];

                    for (int s = 0; s < nk; s++) {
                        int l = diff[diff[s*nk]*nk + m];
                        const double complex *v = buf
                            + ((size_t)s*nao + sig)*n;

                        double complex sum = 0;
                        for (int a = 0; a < n; a++) {
                            size_t r = r0 + a;
                            int t = (int)(r / naux);
                            int Q = (int)(r % naux);

                            size_t shifted = (size_t)diff[s*nk+t]*naux + Q;

                            sum += v[a] * cderi[shifted*npair + p];
                        }

                        vk[((size_t)l*nao + mu)*nao + nu] += sum;
                    }
                }
            }
        }

        free(buf);
    }
}

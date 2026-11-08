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
// BLOCK_SIZE in ft_ao.c: PBC_ft_bvk_nk1s1 evaluates G-vectors in such blocks
#define FT_BLKSIZE      104

/* What dfrs._int3c_kernel and dfrs._ft_kernel pass to PBCfill_nr3c_drv and
 * PBC_ft_bvk_drv, named after the variables of these kernels. Both use
 * bvk_kmesh = 1: the BvK cell is the supercell, and supmol holds its images.
 * For int3c, atm, bas and env also hold the auxiliary supercell, whose shells
 * follow the supercell shells in ao_loc and seg_loc. For FT, atm, bas, env,
 * seg_loc and seg2sh are those of supmol.
 */
typedef struct {
        int *atm;
        int *bas;
        double *env;
        int natm;               // supmol.natm
        int nbas;               // supmol.nbas
        int nbasp;              // supercell nbas
        int *ao_loc;            // cell0_ao_loc
        int *seg_loc;
        int *seg2sh;
        int8_t *cell0_ovlp_mask;// int3c only
        int8_t *ovlp_mask;      // FT only
        int16_t *sindex;        // int3c only
        int log_cutoff;         // int3c only
        CINTOpt *cintopt;       // int3c only
        int cache_size;         // int3c only
} DFRSIntEnv;

typedef int (*DFRSInt3c)(double *, int *, int *, int, float *,
                         CINTEnvVars *, BVKEnvs *);
int PBCint3c2e_sph(double *, int *, int *, int, float *,
                   CINTEnvVars *, BVKEnvs *);
int PBCint3c2e_cart(double *, int *, int *, int, float *,
                    CINTEnvVars *, BVKEnvs *);
void PBCminimal_CINTEnvVars(CINTEnvVars *, int *, int, int *, int,
                            double *, CINTOpt *);
void PBCapprox_bvk_rcond(float *, int, int, BVKEnvs *, int *, int,
                         int *, int, double *, float *);
void PBC_ft_bvk_nk1s1(FPtrIntor, FPtr_eval_gz, int (*)(), double *,
                      double *, int *, CINTEnvVars *, BVKEnvs *);
void PBC_ft_zsort_s1(double *, double *, int, int *, int *,
                     int, int, int, int, int, int, int);
void GTO_Gv_general(double *, double *, double, double, double *,
                    FTEnvVars *, double *);
void ztrsm_(const char *, const char *, const char *, const char *,
            const int *, const int *, const double complex *,
            const double complex *, const int *, double complex *, const int *);

/* x[k,p] = (p|k) for the AO pairs p of the shells (ish, jsh) and the nkaux
 * auxiliary AOs k of the supercell, as PBCfill_nr3c_drv with the fill
 * PBCfill_nr3c_gs1. x has n columns, and screened integrals are not written.
 */
static void int3c_pair(double complex *x, int n, int nkaux, int ish, int jsh,
                       DFRSInt3c intor, DFRSIntEnv *e, double *cache)
{
        int nbasp = e->nbasp;
        if (!e->cell0_ovlp_mask[ish*nbasp+jsh]) {
                return;
        }
        int *ao_loc = e->ao_loc;
        int *seg_loc = e->seg_loc;
        int *seg2sh = e->seg2sh;
        int di = ao_loc[ish+1] - ao_loc[ish];
        int dj = ao_loc[jsh+1] - ao_loc[jsh];
        int ni = seg2sh[seg_loc[ish+1]] - seg2sh[seg_loc[ish]];
        int nj = seg2sh[seg_loc[jsh+1]] - seg2sh[seg_loc[jsh]];
        BVKEnvs bvk = {.ncells = 1, .nkpts = 1, .nbasp = nbasp, .ncomp = 1,
                .seg_loc = seg_loc, .seg2sh = seg2sh, .qindex = e->sindex};
        CINTEnvVars cint;
        PBCminimal_CINTEnvVars(&cint, e->atm, e->natm, e->bas, e->nbas,
                               e->env, e->cintopt);
        float *rcond = malloc(sizeof(float) * 3 * (ni*nj + nj));
        PBCapprox_bvk_rcond(rcond, ish, jsh, &bvk, e->atm, e->natm,
                            e->bas, e->nbas, e->env, rcond + 3*ni*nj);

        int k0 = ao_loc[nbasp];
        int shls[3] = {ish, jsh, 0};
        int cells[2] = {0, 0};
        for (int ksh = nbasp; ao_loc[ksh] < k0 + nkaux; ksh++) {
                shls[2] = ksh;
                if (seg_loc[ksh] == seg_loc[ksh+1] ||
                    !(*intor)(cache, shls, cells, e->log_cutoff, rcond,
                              &cint, &bvk)) {
                        continue;
                }
                // cache holds the integrals in Fortran order [i,j,k]
                int dk = ao_loc[ksh+1] - ao_loc[ksh];
                double complex *xk = x + (size_t)(ao_loc[ksh] - k0) * n;
                for (int k = 0; k < dk; k++) {
                for (int i = 0; i < di; i++) {
                for (int j = 0; j < dj; j++) {
                        xk[k*n+i*dj+j] = cache[(k*dj+j)*di+i];
                } } }
        }
        free(rcond);
}

// PBC_ft_bvk_nk1s1 expects an int-valued sort function
static int ft_sort(double *out, double *in, int fill_zero, int *shls_slice,
                   int *ao_loc, int nkpts, int comp, int nGv,
                   int ish, int jsh, int grid0, int grid1)
{
        PBC_ft_zsort_s1(out, in, fill_zero, shls_slice, ao_loc, nkpts, comp,
                        nGv, ish, jsh, grid0, grid1);
        return 0;
}

/* rho[p,G] for the AO pairs p of the shells (ish, jsh) and the G-vectors
 * gv[3,ng], as PBC_ft_bvk_drv at k = 0.
 */
static void ft_pair(double complex *rho, int ish, int jsh, int ng, double *gv,
                    FPtrIntor intor, DFRSIntEnv *e, double *cache)
{
        int slice[4] = {ish, ish+1, jsh, jsh+1};
        int shls[2] = {ish, jsh};
        int gs[3] = {0, 0, 0};
        double expLkR = 1, expLkI = 0, b = 0;
        BVKEnvs bvk = {.ncells = 1, .nkpts = 1, .nbasp = e->nbasp, .ncomp = 1,
                .nGv = ng, .seg_loc = e->seg_loc, .seg2sh = e->seg2sh,
                .ao_loc = e->ao_loc, .shls_slice = slice, .expLkR = &expLkR,
                .expLkI = &expLkI, .ovlp_mask = e->ovlp_mask, .Gv = gv,
                .b = &b, .gs = gs};
        CINTEnvVars cint;
        PBCminimal_CINTEnvVars(&cint, e->atm, e->natm, e->bas, e->nbas,
                               e->env, NULL);
        PBC_ft_bvk_nk1s1(intor, &GTO_Gv_general, &ft_sort, (double *)rho,
                         cache, shls, &cint, &bvk);
}

// Scratch of PBC_ft_bvk_nk1s1 for up to ng G-vectors, as in PBC_ft_bvk_drv
static size_t ft_cache_size(FPtrIntor intor, DFRSIntEnv *e, int ng)
{
        int blksize = MIN(ng, FT_BLKSIZE);
        int gs[3] = {0, 0, 0};
        double b = 0;
        size_t size = 0;
        for (int sh = 0; sh < e->nbas; sh++) {
                int shls[2] = {sh, sh};
                size_t n = (*intor)(NULL, NULL, shls, NULL, &GTO_Gv_general, 0,
                                    NULL, &b, NULL, gs, ng, blksize, e->atm,
                                    e->natm, e->bas, e->nbas, e->env, NULL);
                size = MAX(size, n * blksize);
        }
        int dmax = 0;
        for (int sh = 0; sh < e->nbasp; sh++) {
                dmax = MAX(dmax, e->ao_loc[sh+1] - e->ao_loc[sh]);
        }
        // plus the real and imaginary parts of one G block
        return size + 2 * dmax * dmax * FT_BLKSIZE;
}

/* C version of dfrs._make_cderi. With p the AO pairs of the retained shell
 * pairs, t the auxiliary cells and q the k-mesh in np.fft.fftn order,
 *
 *   A[q,Q,p] = sum_t exp(-iq.t) (p|Q,t)_SR - delta(q,0) g0 charge[Q] rho_0[p,0]
 *              + sum_G weighted_q[Q,G] rho_q[p,G]
 *   B[t,Q,p] = 1/nk sum_q exp(iq.t) (L_q^{-1} A[q])[Q,p]
 *
 * where rho_q[p,G] is the periodic FT of the AO pair p at the G-vectors of q.
 * Every batch of shell pairs is an OpenMP task; use single-threaded BLAS.
 *
 * cderi[nk,naux,npair] output B[t,Q,p]
 * shells[nshells,4]    (ish, jsh, p0, p1) of dfrs._pair_list
 * batch_size           number of shell pairs per task, >= 1
 * chol[nk,naux,naux]   row-major lower Cholesky factors of dfrs._metric
 * gloc[nk+1]           offsets of the G-vectors of each q
 * gv, weighted         reciprocal[q] = (Gv, weighted) of dfrs._metric, as
 *                      Gv.T[3,ng] at gv+3*gloc[q] and weighted[naux,ng] at
 *                      weighted+naux*gloc[q]
 * charge, g0           charge of dfrs._metric and pi/(eta**2*cell.vol)
 * cart                 cartesian basis
 * sr, ft               the two kernels; sr = NULL when cell.omega > 0
 */
void PBCdfrs_make_cderi(double complex *cderi, int *shells, int nshells,
                        int batch_size, int *kmesh, int naux, int npair,
                        double complex *chol, int *gloc, double *gv,
                        double complex *weighted, double *charge, double g0,
                        int cart, DFRSIntEnv *sr, DFRSIntEnv *ft)
{
        DFRSInt3c int3c = cart ? &PBCint3c2e_cart : &PBCint3c2e_sph;
        FPtrIntor ftint = cart ? &GTO_ft_ovlp_cart : &GTO_ft_ovlp_sph;
        int nk = kmesh[0] * kmesh[1] * kmesh[2];
        int nkaux = nk * naux;
        int nbatch = (nshells + batch_size - 1) / batch_size;

        // The DFT over the k-mesh as a matrix, phase[q,t] = exp(-iq.t)
        double complex *phase = malloc(sizeof(double complex) * nk * nk);
        for (int qt = 0; qt < nk*nk; qt++) {
                int q = qt / nk;
                int t = qt % nk;
                double x = 0;
                for (int d = 2; d >= 0; d--) {
                        x += (double)(q % kmesh[d] * (t % kmesh[d]) % kmesh[d]) / kmesh[d];
                        q /= kmesh[d];
                        t /= kmesh[d];
                }
                phase[qt] = cexp(-2 * M_PI * I * x);
        }

        int ngmax = 1;
        for (int q = 0; q < nk; q++) {
                ngmax = MAX(ngmax, gloc[q+1] - gloc[q]);
        }
        size_t cache_size = ft_cache_size(ftint, ft, ngmax);
        if (sr != NULL) {
                cache_size = MAX(cache_size, (size_t)sr->cache_size);
        }
        const double complex one = 1;
        const double complex zero = 0;
        const double complex inv_nk = 1. / nk;
        double gv0[3] = {0, 0, 0};

#pragma omp parallel for schedule(dynamic)
        for (int batch = 0; batch < nbatch; batch++) {
                int s0 = batch * batch_size;
                int s1 = MIN(s0 + batch_size, nshells);
                int p0 = shells[4*s0+2];
                int n = shells[4*s1-1] - p0;
                int m = naux * n;
                // bt = B[t,Q,p] and aq = A[q,Q,p] of the batch
                double complex *bt = calloc((size_t)nkaux * n, sizeof(double complex));
                double complex *aq = calloc((size_t)nkaux * n, sizeof(double complex));
                double complex *rho = malloc(sizeof(double complex) * n * ngmax);
                double *cache = malloc(sizeof(double) * cache_size);

                if (sr != NULL) {
                        for (int s = s0; s < s1; s++) {
                                int *sh = shells + 4*s;
                                int3c_pair(bt+sh[2]-p0, n, nkaux, sh[0], sh[1],
                                           int3c, sr, cache);
                                ft_pair(rho+sh[2]-p0, sh[0], sh[1], 1, gv0,
                                        ftint, ft, cache);
                        }
                        // aq[q] = sum_t exp(-iq.t) bt[t], i.e. np.fft.fftn
                        zgemm_("N", "N", &m, &nk, &nk, &one, bt, &m,
                               phase, &nk, &zero, aq, &m);
                        // remove the G = 0 term of the SR Coulomb kernel
                        for (int Q = 0; Q < naux; Q++) {
                        for (int p = 0; p < n; p++) {
                                aq[Q*n+p] -= g0 * charge[Q] * creal(rho[p]);
                        } }
                }

                for (int q = 0; q < nk; q++) {
                        double complex *a = aq + (size_t)q * m;
                        int ng = gloc[q+1] - gloc[q];
                        if (ng > 0) {
                                for (int s = s0; s < s1; s++) {
                                        int *sh = shells + 4*s;
                                        ft_pair(rho+(size_t)(sh[2]-p0)*ng, sh[0], sh[1],
                                                ng, gv+3*gloc[q], ftint, ft, cache);
                                }
                                // a[Q,p] += weighted_q[Q,G] rho[p,G]
                                zgemm_("T", "N", &n, &naux, &ng, &one, rho, &ng,
                                       weighted+(size_t)naux*gloc[q], &ng,
                                       &one, a, &n);
                        }
                        // a = L_q^{-1} a, i.e. a^T L_q^T = a^T in column-major
                        ztrsm_("R", "U", "N", "N", &n, &naux, &one,
                               chol+(size_t)q*naux*naux, &naux, a, &n);
                }

                // bt[t] = 1/nk sum_q exp(iq.t) aq[q], i.e. np.fft.ifftn
                zgemm_("N", "C", &m, &nk, &nk, &inv_nk, aq, &m,
                       phase, &nk, &zero, bt, &m);
                for (int r = 0; r < nkaux; r++) {
                        memcpy(cderi+(size_t)r*npair+p0, bt+(size_t)r*n,
                               sizeof(double complex) * n);
                }
                free(cache);
                free(rho);
                free(aq);
                free(bt);
        }
        free(phase);
}

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
 * or atomic updates are needed. Scratch per thread is
 * (nk*nao + 1) * min(blockdim,naux) complex doubles.
 */
void PBCdfrs_get_k(double complex *vk, const double complex *cderi,
                   const double complex *dm, const int *pairs, const int *diff,
                   int nk, int naux, int nao, int npair, int blockdim)
{
        memset(vk, 0, sizeof(double complex) * nk * nao * nao);
        if (npair == 0) {
                return;
        }
        blockdim = MIN(blockdim, naux);

#pragma omp parallel
        {
                double complex *buf = malloc(sizeof(double complex) * nk * nao * blockdim);
                double complex *bp = malloc(sizeof(double complex) * blockdim);

#pragma omp for schedule(dynamic)
                for (int mu = 0; mu < nao; mu++) {
                        for (int a0 = 0; a0 < naux; a0 += blockdim) {
                        int n = MIN(blockdim, naux - a0);
                        for (int t = 0; t < nk; t++) {
                                memset(buf, 0, sizeof(double complex) * nk * nao * n);

                                // T[s,sigma,Q] = sum_(lambda,m) B[t,Q,p]^* P[s-m,lambda,sigma]
                                for (int p = 0; p < npair; p++) {
                                        if (pairs[3*p] != mu) {
                                                continue;
                                        }
                                        int lam = pairs[3*p+1];
                                        int m = pairs[3*p+2];
                                        const double complex *b = cderi + ((size_t)t*naux+a0)*npair+p;
                                        for (int a = 0; a < n; a++) {
                                                bp[a] = conj(b[(size_t)a*npair]);
                                        }
                                        for (int s = 0; s < nk; s++) {
                                                const double complex *d = dm + ((size_t)diff[m*nk+s]*nao+lam)*nao;
                                                for (int sig = 0; sig < nao; sig++) {
                                                        double complex *v = buf + ((size_t)s*nao+sig)*n;
                                                        for (int a = 0; a < n; a++) {
                                                                v[a] += d[sig] * bp[a];
                                                        }
                                                }
                                        }
                                }

                                // K[s+m,mu,nu] += sum_Q T[s,sigma,Q] B[t-s,Q,p]
                                for (int p = 0; p < npair; p++) {
                                        int sig = pairs[3*p];
                                        int nu = pairs[3*p+1];
                                        int m = pairs[3*p+2];
                                        for (int s = 0; s < nk; s++) {
                                                int l = diff[diff[s*nk]*nk+m];
                                                const double complex *b = cderi + ((size_t)diff[s*nk+t]*naux+a0)*npair+p;
                                                const double complex *v = buf + ((size_t)s*nao+sig)*n;
                                                double complex sum = 0;
                                                for (int a = 0; a < n; a++) {
                                                        sum += v[a] * b[(size_t)a*npair];
                                                }
                                                vk[((size_t)l*nao+mu)*nao+nu] += sum;
                                        }
                                }
                        } }
                }
                free(bp);
                free(buf);
        }
}

`pari.py` implements the supplied real-space, occupied-orbital PARI equations for the short-range kernel

$$
w_{\mathrm{SR}}(r)=\frac{\operatorname{erfc}(\omega r)}{r},\qquad \omega>0.
$$

**The calculation explicitly constructs $D_i^{\mu_{\mathbf0}Q,\mathbf k}$ and $H_i^{\nu_{\mathbf T}Q,\mathbf k}$, contracts them into $L^{\mathbf0\mathbf T}$, reconstructs $K^{\mathbf0\mathbf T}$, and Fourier-transforms only the final exchange matrix.** The local fitting coefficients and the intermediate auxiliary indices retain their actual atom/image identities.

| Mathematical quantity | Code representation or entry point |
|---|---|
| Local coefficients $C_Q^{\mu_{\mathbf0}\lambda_{\mathbf T_1}}$ | Local `_Pair` coefficient blocks |
| Occupation-weighted coefficients $M^{\mathbf k}_{\lambda i}$ | Input `m`, one matrix per density k-point |
| $D_i^{\mu_{\mathbf0}Q_{\mathbf U},\mathbf k}$ | `data.d[U][ik][mu, Q, i]` |
| Anchored $H_i^{\nu_{\mathbf t}Q_{\mathbf0},\mathbf k}$ | `data.h0[t][ik][nu, Q, i]` |
| $H_i^{\nu_{\mathbf T}Q_{\mathbf U},\mathbf k}$ | `data.h(T, U)` |
| All MO intermediates | `PARI.get_intermediates(m, kpts=...)` |
| $L^{\mathbf0\mathbf T}$ | `PARI.get_l(m, ...)`, a dictionary of AO matrices |
| $K^{\mathbf0\mathbf T}$ | `PARI.get_k_realspace(l)` |
| Final $K(\mathbf k_{\mathrm{out}})$ | `get_k_from_m`, `get_k_from_mo`, or `get_k` |

The implementation is in [pari.py](pari.py). All three public exchange input routes use the same real-space MO-intermediate calculation. There is no separate direct density-matrix exchange contraction.

Inputs are assumed to have the documented shapes and finite values. Routine shape and range validation is left to NumPy and PySCF; explicit checks retain the kernel, factorization, k-point-count, and stale-intermediate constraints needed for this calculation.

In the formulas below, bold capital letters denote Cartesian lattice translations in Bohr. Dictionary keys such as `U`, `t`, and `T` are integer lattice triples. If the lattice vectors are the rows of $a$, the key $\mathbf n$ represents the Cartesian translation $\mathbf n a$. K-points are Cartesian vectors in inverse Bohr, so phases always use the physical dot product $\mathbf k\cdot(\mathbf n a)$.

An auxiliary function $Q_{\mathbf U}$ is identified by both its reference-cell auxiliary index $Q$ and its image $\mathbf U$. Different images of the same reference-cell auxiliary function are distinct fitting centers.

---

The density-matrix convention is exactly

$$
P_{\lambda\sigma}^{\mathbf T}
=\frac1{N_k}\sum_{\mathbf k}
 e^{-i\mathbf k\cdot\mathbf T}P_{\lambda\sigma}^{\mathbf k},
\qquad
P_{\lambda\sigma}^{\mathbf k}
=\sum_i(M_{\lambda i}^{\mathbf k})^*M_{\sigma i}^{\mathbf k}.
$$

The code accepts a full, equally weighted list of density k-points. $N_k$ is the number of entries in that list. It does not interpret an irreducible k-point list as a weighted full Brillouin-zone sampling.

PySCF's usual orbital convention is

$$
P^{\mathbf k}
=U^{\mathbf k}f^{\mathbf k}(U^{\mathbf k})^\dagger,
$$

where $U^{\mathbf k}$ contains MO coefficients and $f^{\mathbf k}$ contains occupations. Therefore the required input is

$$
\boxed{M^{\mathbf k}=(U^{\mathbf k})^*\sqrt{f^{\mathbf k}}.}
$$

In array notation, with occupations stored as a vector:

```python
m_k = mo_coeff_k.conj() * np.sqrt(mo_occ_k)[None, :]
```

Fractional occupations are included through their square roots. There is no extra factor of $1/2$. The number of columns can differ between k-points because occupied-orbital indices are contracted only within a given k-point.

`get_k_from_m(m, ...)` accepts the $M$ matrices directly. `get_k_from_mo(mo_coeff, mo_occ, ...)` constructs them from ordinary PySCF orbitals and nonnegative occupations.

For convenience, `get_k(dm, hermi=1, ...)` factors each Hermitian positive-semidefinite input density:

$$
P^{\mathbf k}=U\,\operatorname{diag}(w)\,U^\dagger,
\qquad
M^{\mathbf k}=U^*\sqrt{\operatorname{diag}(w)}.
$$

These factorization vectors need not be physical SCF orbitals. Only the identity $P=M^*M^T$ matters. No AO-overlap factor is required for this algebraic factorization of the supplied density matrix.

Genuinely negative density eigenvalues and non-Hermitian densities are rejected. Tiny negative eigenvalues attributable to roundoff may be clipped to zero. The expression $P=M^*M^T$ cannot represent an indefinite or non-Hermitian density, and the supplied Hermitian reconstruction of exchange does not apply to a general non-Hermitian input. A zero density has an empty orbital factor and produces zero exchange.

The public density interface requires `hermi=1`; `hermi=0` is unsupported. Every positive density eigenvalue is retained, independently of the auxiliary fitting threshold. The roundoff tolerance for negative eigenvalues is $100\,\epsilon\,n_{\mathrm{AO}}\max(1,\max_i|w_i|)$, with $\epsilon$ the double-precision machine epsilon.

---

For each localized AO pair $\mu_{\mathbf0}\lambda_{\mathbf T_1}$, the auxiliary fitting space is the union

$$
\mathcal S_{\mu,\lambda,\mathbf T_1}
=\mathcal A(\mu_{\mathbf0})\cup\mathcal A(\lambda_{\mathbf T_1}).
$$

The fit is

$$
\widetilde{\mu_{\mathbf0}\lambda_{\mathbf T_1}}
=\sum_{Q_{\mathbf U}\in\mathcal S_{\mu,\lambda,\mathbf T_1}}
 C_{Q_{\mathbf U}}^{\mu_{\mathbf0}\lambda_{\mathbf T_1}}Q_{\mathbf U}.
$$

All coefficients outside that support are zero. For a pair whose atoms are $A_{\mathbf0}$ and $B_{\mathbf T_1}$, the code stores coefficients on the first atom in `ca` and coefficients on the second atom in `cb`.

| Pair | Fitting support |
|---|---|
| Same atom, same image | One atomic auxiliary basis; `cb=None` |
| Different atoms | Both atomic auxiliary bases |
| Same reference-cell atom, different images | Two distinct auxiliary centers |

The same-atom, same-image exception implements a set union: it prevents double counting a single auxiliary basis. Images with $\mathbf T_1\ne\mathbf0$ remain distinct even when the two atom indices are equal.

The local coefficients solve

$$
\sum_{R_{\mathbf V}\in\mathcal S}
 V^{\mathrm{loc}}_{Q_{\mathbf U},R_{\mathbf V}}
 C_{R_{\mathbf V}}^{\mu_{\mathbf0}\lambda_{\mathbf T_1}}
=b_{Q_{\mathbf U}}^{\mu\lambda,\mathbf T_1},
$$

where

$$
V^{\mathrm{loc}}_{Q_{\mathbf U},R_{\mathbf V}}
=(Q_{\mathbf U}|R_{\mathbf V})_{\mathrm{SR}},
\qquad
b_{Q_{\mathbf U}}^{\mu\lambda,\mathbf T_1}
=(Q_{\mathbf U}|\mu_{\mathbf0}\lambda_{\mathbf T_1})_{\mathrm{SR}}.
$$

These are molecular integrals on the selected atom/images. The local fitting metric is not a periodized metric over the whole auxiliary crystal.

PySCF's integral engine uses a negative range parameter for the erfc kernel:

```python
with auxmol.with_range_coulomb(-self.omega):
    metric = auxmol.intor('int2c2e', hermi=1)

with mol.with_range_coulomb(-self.omega):
    rhs = mol_incore.aux_e2(
        mol, auxmol, aosym='s1', shls_slice=shls)
```

The same short-range kernel is used in the local metric, the local right-hand side, and all later three- and two-center integrals. Extracting atom/image molecular objects preserves the cell's basis normalization, contraction coefficients, and spherical/Cartesian representation.

The local metric is solved through a diagonally normalized eigendecomposition. Define

$$
S_{PP}=\sqrt{V^{\mathrm{loc}}_{PP}},
\qquad
\overline V=S^{-1}V^{\mathrm{loc}}S^{-1}=U W U^T.
$$

Eigenvectors satisfying $w_i>\tau w_{\max}$ are retained, where $\tau$ is `linear_dep_threshold`. The regularized solution is

$$
C=S^{-1}U_rW_r^{-1}U_r^TS^{-1}b.
$$

This handles nearly dependent and duplicated auxiliary functions. The fitting threshold controls this local numerical regularization; it is unrelated to occupied-orbital truncation.

Pairs related by

$$
(A_{\mathbf0},B_{\mathbf T_1})
\longleftrightarrow(B_{\mathbf0},A_{-\mathbf T_1})
$$

are fitted once and stored in both directed orientations. Swapping the AO indices and auxiliary blocks preserves the symmetry of the real localized product and its fitted representation.

---

The first occupied-orbital intermediate is constructed explicitly:

$$
\boxed{
D_i^{\mu_{\mathbf0}Q_{\mathbf U},\mathbf k}
=\sum_{\lambda,\mathbf T_1}
 C_{Q_{\mathbf U}}^{\mu_{\mathbf0}\lambda_{\mathbf T_1}}
 M_{\lambda i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf T_1}.
}
$$

**For each directed atom pair $(A_{\mathbf0},B_{\mathbf T_1})$, the computation accumulates the auxiliary contribution with $\mathbf U=\mathbf0$ first, then the contribution with $\mathbf U=\mathbf T_1$.** The two coefficient blocks come from the same joint local fit over the auxiliary union.

For this pair, let $\mu\in A$ and $\lambda\in B$. The two steps are:

1. **$\mathbf U=\mathbf0$: auxiliary functions on $A_{\mathbf0}$.** Contract `ca` with the partner-atom orbitals and accumulate into `D[0]`:

$$
\Delta D_{\mu Qi}^{(A),\mathbf k}
=\sum_{\lambda\in B}
C_{Q_{\mathbf0}}^{\mu_{\mathbf0}\lambda_{\mathbf T_1}}
M_{\lambda i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf T_1},
\qquad Q\in\mathcal A(A).
$$

2. **$\mathbf U=\mathbf T_1$: auxiliary functions on $B_{\mathbf T_1}$.** Contract `cb` with the same partner-atom orbitals and accumulate into `D[T1]`:

$$
\Delta D_{\mu Qi}^{(B),\mathbf k}
=\sum_{\lambda\in B}
C_{Q_{\mathbf T_1}}^{\mu_{\mathbf0}\lambda_{\mathbf T_1}}
M_{\lambda i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf T_1},
\qquad Q\in\mathcal A(B).
$$

Both steps use $e^{-i\mathbf k\cdot\mathbf T_1}$. This phase belongs to the partner AO $\lambda_{\mathbf T_1}$, so it remains present in the first step even though its auxiliary cell is $\mathbf U=\mathbf0$.

The ordered contributions in `_build_d()` are:

```python
contributions = (
    (zero,       qa0, qa1, pair.ca),  # First: U=0,  Q on A_0
    (pair.image, qb0, qb1, pair.cb),  # Then:  U=T1, Q on B_T1
)
for u, q0, q1, coeff in contributions:
    if coeff is None:
        continue
    # Allocate D[u] if needed, then for every input k-point:
    D[u][ik][a0:a1, q0:q1] += phase * np.einsum(
        'mlq,li->mqi', coeff, M_B)
# phase = exp(-1j * k.dot(T1_cartesian)) in both steps
```

This is a two-center order for each pair, not a partition into two disjoint sets of cell labels:

| Pair case | First contribution | Second contribution |
|---|---|---|
| $\mathbf T_1\ne\mathbf0$ | `D[0]`, auxiliary slice of $A$ | `D[T1]`, auxiliary slice of $B$ |
| $\mathbf T_1=\mathbf0$, $A\ne B$ | `D[0]`, auxiliary slice of $A$ | The same `D[0]`, auxiliary slice of $B$ |
| $\mathbf T_1=\mathbf0$, $A=B$ | `D[0]`, auxiliary slice of $A$ | Omitted (`cb=None`) to count the union once |

After these updates for all retained pairs, `D[U]` contains the sum over the appropriate $\lambda,\mathbf T_1$ contributions. Each completed `D[U]` is then used once in the auxiliary-image sum for $L$; the accumulated `D[0]` is not counted again for every local pair.

The two pair contributions also make the corresponding $H$ arguments explicit. For a fixed output cell $\mathbf T$, their contributions to $L^{\mathbf0\mathbf T}$ are

$$
\begin{aligned}
\Delta L_{\mu\nu}^{\mathbf0\mathbf T}
=\frac1{N_k}\sum_{\mathbf k,i}\Bigg[
&\sum_{Q\in\mathcal A(A)}
\left(\Delta D_{\mu Qi}^{(A),\mathbf k}\right)^*
H_i^{\nu_{\mathbf T}Q_{\mathbf0},\mathbf k}\\
{}+&\sum_{Q\in\mathcal A(B)}
\left(\Delta D_{\mu Qi}^{(B),\mathbf k}\right)^*
H_i^{\nu_{\mathbf T}Q_{\mathbf T_1},\mathbf k}
\Bigg].
\end{aligned}
$$

The second term is omitted for the shared same-atom, same-image case. Using the anchored $H_0$ defined below, the first term uses $H_0[\mathbf T]$, whereas the second uses $e^{-i\mathbf k\cdot\mathbf T_1}H_0[\mathbf T-\mathbf T_1]$. Equivalently, when looping over stored anchored images $\mathbf t$, the first contribution reaches output cell $\mathbf T=\mathbf t$ and the second reaches $\mathbf T=\mathbf T_1+\mathbf t$. This is the relation $\mathbf T=\mathbf U+\mathbf t$ for the two allowed auxiliary centers of a local pair.

`data.d[U]` is a tuple over density k-points. Its entry at k-point `ik` has shape

```text
(nao_unit_cell, naux_unit_cell, nocc_at_this_kpoint)
```

Only auxiliary images reached by nonzero retained local coefficient blocks need to be stored. Unused atom/auxiliary portions of an image block are zero. Missing image blocks are interpreted as zero.

The sparse image dictionary is an implementation of the full auxiliary-crystal sum with zero-extended coefficients. It does not merge different auxiliary images or Fourier-transform the auxiliary index.

---

The second occupied-orbital intermediate is

$$
\boxed{
H_i^{\nu_{\mathbf T}Q_{\mathbf U},\mathbf k}
=\sum_{\sigma,\mathbf T_2}
 M_{\sigma i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf T_2}
 \left[
 (\sigma_{\mathbf T_2}\nu_{\mathbf T}|Q_{\mathbf U})_{\mathrm{SR}}
 -\frac12\sum_{R,\mathbf V}
 C_{R_{\mathbf V}}^{\sigma_{\mathbf T_2}\nu_{\mathbf T}}
 (R_{\mathbf V}|Q_{\mathbf U})_{\mathrm{SR}}
 \right].
}
$$

For a fixed left AO pair $(A_{\mathbf0},B_{\mathbf T_1})$, read the construction in the same order as the two contributions to $D$: **first $\mathbf U=\mathbf0$, with $Q$ on $A$; then $\mathbf U=\mathbf T_1$, with $Q$ on $B$.** In each case, $H$ includes the full sum over the right AO pair $\sigma_{\mathbf T_2}\nu_{\mathbf T}$. Choosing $\mathbf U$ selects the location of $Q$; it does not restrict $\sigma,\mathbf T_2$ to the left pair.

The auxiliary indices in this expression have different roles:

| Index | Role in constructing $H$ |
|---|---|
| $Q_{\mathbf U}$ | Fixed auxiliary function whose $H$ value is being constructed; selected from the left pair's auxiliary union |
| $R_{\mathbf V}$ | Summed auxiliary function in the fit of the right pair; supported on $\mathcal A(\sigma_{\mathbf T_2})\cup\mathcal A(\nu_{\mathbf T})$ |

**First, construct $H$ for $\mathbf U=\mathbf0$.** The code stores this reference-cell auxiliary case as

$$
H_{0,i}^{\nu_{\mathbf t}Q,\mathbf k}
:=H_i^{\nu_{\mathbf t}Q_{\mathbf0},\mathbf k}.
$$

Here $\mathbf t$ runs over the external AO's images. For this first case, $\mathbf t=\mathbf T$. The subscript $0$ in $H_0$ specifies the cell of $Q$, not the cell of $\nu$. `_build_h0()` computes all reference-cell auxiliary functions together; the first contribution from the fixed left pair uses the slice $Q\in\mathcal A(A)$.

Construct $H_0$ by adding the bare three-center term and then subtracting half the fitted term.

The bare three-center contribution is

$$
J_i^{\nu_{\mathbf t}Q,\mathbf k}
=\sum_{\sigma,\mathbf S}
 M_{\sigma i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf S}
 (\sigma_{\mathbf S}\nu_{\mathbf t}|Q_{\mathbf0})_{\mathrm{SR}}.
$$

It is obtained from molecular three-center integrals involving an AO-image cluster and the reference-cell auxiliary basis. The source-AO image $\mathbf S$ carries the phase $e^{-i\mathbf k\cdot\mathbf S}$ before contraction with $M^{\mathbf k}$.

In `_build_h0()`, `source_m[ik]` contains these phased orbital coefficients, with $(\mathbf S,\sigma)$ combined into one source-AO index. For each external atom in image $\mathbf t$, `ints` has axes $(\nu,(\mathbf S,\sigma),Q)$. The code initializes the corresponding part of `h0[t][ik]` with

```python
# source_m[ik][(S, sigma), i] = exp(-1j * k.dot(S_cartesian)) * M[sigma, i]
h0[t][ik][a0:a1] += np.einsum('nsq,si->nqi', ints, source_m[ik])
```

This bare contribution has its own integral-image cutoff. It cannot generally be restricted to the retained fitting-pair list: setting a pair's fitting coefficients to zero does not set its bare three-center integral to zero.

The fitted correction can reuse the already constructed $D$ intermediate. Define the real-space auxiliary metric blocks

$$
V_{QR}^{\mathbf V}:=(Q_{\mathbf0}|R_{\mathbf V})_{\mathrm{SR}}.
$$

These are individual molecular two-center integral blocks at a specified image separation. There is no k-space auxiliary metric in this contraction.

To see the reuse of $D$, define $\mathcal F$ as the quantity multiplied by $-1/2$, so that $H_0=J-\tfrac12\mathcal F$:

$$
\mathcal F_i^{\nu_{\mathbf t}Q,\mathbf k}
=\sum_{\sigma,\mathbf S}\sum_{R,\mathbf V}
 M_{\sigma i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf S}
 C_{R_{\mathbf V}}^{\sigma_{\mathbf S}\nu_{\mathbf t}}
 (R_{\mathbf V}|Q_{\mathbf0})_{\mathrm{SR}}.
$$

Exchange the order of the AO pair and translate it by $-\mathbf t$. With $\mathbf S'=\mathbf S-\mathbf t$ and $\mathbf W=\mathbf V-\mathbf t$,

$$
C_{R_{\mathbf V}}^{\sigma_{\mathbf S}\nu_{\mathbf t}}
=C_{R_{\mathbf W}}^{\nu_{\mathbf0}\sigma_{\mathbf S'}},
\qquad
 e^{-i\mathbf k\cdot\mathbf S}
=e^{-i\mathbf k\cdot\mathbf t}e^{-i\mathbf k\cdot\mathbf S'}.
$$

Therefore

$$
\begin{aligned}
\mathcal F_i^{\nu_{\mathbf t}Q,\mathbf k}
&=e^{-i\mathbf k\cdot\mathbf t}
 \sum_{R,\mathbf W}(Q_{\mathbf0}|R_{\mathbf W+\mathbf t})_{\mathrm{SR}}
 \left[\sum_{\sigma,\mathbf S'}
 C_{R_{\mathbf W}}^{\nu_{\mathbf0}\sigma_{\mathbf S'}}
 M_{\sigma i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf S'}\right]\\
&=e^{-i\mathbf k\cdot\mathbf t}
 \sum_{R,\mathbf W}V_{QR}^{\mathbf W+\mathbf t}
 D_i^{\nu_{\mathbf0}R_{\mathbf W},\mathbf k}.
\end{aligned}
$$

The $D$ on the last line is the same stored intermediate, now read with external AO $\nu$ and auxiliary index $R$. For each right pair $\nu_{\mathbf0}\sigma_{\mathbf S'}$, its coefficient support has the two cases $\mathbf W=\mathbf0$ on the atom of $\nu$ and $\mathbf W=\mathbf S'$ on the atom of $\sigma$. `_build_d()` has already accumulated both contributions. The sum over $\mathbf W$ therefore includes all stored `D` images; $\mathbf W$ is independent of the left pair's choice $\mathbf U=\mathbf0$ or $\mathbf T_1$.

The anchored second intermediate is consequently

$$
\boxed{
H_{0,i}^{\nu_{\mathbf t}Q,\mathbf k}
=J_i^{\nu_{\mathbf t}Q,\mathbf k}
-\frac12e^{-i\mathbf k\cdot\mathbf t}
 \sum_{R,\mathbf W}V_{QR}^{\mathbf W+\mathbf t}
 D_i^{\nu_{\mathbf0}R_{\mathbf W},\mathbf k}.
}
$$

To evaluate this correction, `_build_h0()` loops over the stored metric images $\mathbf V$ and the stored $D$ images $\mathbf W$. The required metric image is $\mathbf V=\mathbf W+\mathbf t$, so each such pair contributes to `h0[t]` at $\mathbf t=\mathbf V-\mathbf W$. Schematically, after initializing `h0` with the bare contribution:

```python
for v, metric_v in metric_blocks.items():  # metric_v[Q, R] = (Q_0 | R_v)_SR
    for w, dw in D.items():               # dw[ik][nu, R, i] = D^(nu_0 R_w)
        t = tuple(np.asarray(v) - w)
        # Allocate h0[t] as zero if there was no bare contribution at t.
        for ik, k in enumerate(kpts):
            phase = np.exp(-1j * k.dot(np.dot(t, lattice)))
            h0[t][ik][...] -= 0.5 * phase * np.einsum(
                'qr,nri->nqi', metric_v, dw[ik])
```

Here the metric contracts the $R$ axis of `D`, leaving $(\nu,Q,i)$. The factor $e^{-i\mathbf k\cdot\mathbf t}$ comes from translating the right AO pair to $\nu_{\mathbf0}$ in the derivation above. It is present even though $Q$ is in cell zero.

This is exactly the fitted correction in the supplied $H_i$ definition, with the internal source-AO contraction reused through $D$.

`data.h0[t]` is a tuple over density k-points with the same array-axis convention as `data.d[U]`. The external AO belongs to image $\mathbf t$; the auxiliary center belongs to image zero.

For the first contribution of the fixed left pair, set $\mathbf t=\mathbf T$ and select $Q\in\mathcal A(A)$:

$$
\boxed{
\underbrace{H_i^{\nu_{\mathbf T}Q_{\mathbf0},\mathbf k}}_{\mathbf U=\mathbf0}
=J_i^{\nu_{\mathbf T}Q,\mathbf k}
-\frac12e^{-i\mathbf k\cdot\mathbf T}
 \sum_{R,\mathbf W}V_{QR}^{\mathbf W+\mathbf T}
 D_i^{\nu_{\mathbf0}R_{\mathbf W},\mathbf k}.
}
$$

This is `data.h(T, (0, 0, 0))`, or simply the stored `data.h0[T]` when that image is present.

---

**Then, obtain $H$ for $\mathbf U=\mathbf T_1$.** For the second contribution of the same left pair, $Q\in\mathcal A(B)$ and the required quantity is $H_i^{\nu_{\mathbf T}Q_{\mathbf T_1},\mathbf k}$. Translate every center in its defining expression by $-\mathbf T_1$:

| Original center | Center after translation |
|---|---|
| $Q_{\mathbf T_1}$ | $Q_{\mathbf0}$ |
| $\nu_{\mathbf T}$ | $\nu_{\mathbf t}$, where $\mathbf t=\mathbf T-\mathbf T_1$ |
| $\sigma_{\mathbf T_2}$ | $\sigma_{\mathbf S}$, where $\mathbf S=\mathbf T_2-\mathbf T_1$ |
| $R_{\mathbf V}$ in the right-pair fit | $R_{\mathbf Z}$, where $\mathbf Z=\mathbf V-\mathbf T_1$ |

The localized integrals and local fitting coefficients are invariant under this simultaneous translation. In particular,

$$
\begin{aligned}
(\sigma_{\mathbf T_2}\nu_{\mathbf T}|Q_{\mathbf T_1})_{\mathrm{SR}}
&=(\sigma_{\mathbf S}\nu_{\mathbf t}|Q_{\mathbf0})_{\mathrm{SR}},\\
C_{R_{\mathbf V}}^{\sigma_{\mathbf T_2}\nu_{\mathbf T}}
(R_{\mathbf V}|Q_{\mathbf T_1})_{\mathrm{SR}}
&=C_{R_{\mathbf Z}}^{\sigma_{\mathbf S}\nu_{\mathbf t}}
(R_{\mathbf Z}|Q_{\mathbf0})_{\mathrm{SR}}.
\end{aligned}
$$

The orbital phase separates as

$$
e^{-i\mathbf k\cdot\mathbf T_2}
=e^{-i\mathbf k\cdot\mathbf T_1}e^{-i\mathbf k\cdot\mathbf S}.
$$

The remaining sum over $\sigma,\mathbf S$ and $R,\mathbf Z$ is precisely the $H_0$ already constructed. Therefore

$$
\boxed{
\underbrace{H_i^{\nu_{\mathbf T}Q_{\mathbf T_1},\mathbf k}}_{\mathbf U=\mathbf T_1}
=e^{-i\mathbf k\cdot\mathbf T_1}
 H_{0,i}^{\nu_{\mathbf T-\mathbf T_1}Q,\mathbf k}.
}
$$

This phase multiplies **both** the bare term and the fitted correction. Substituting the expression for $H_0$ gives the explicit second case:

$$
\boxed{
H_i^{\nu_{\mathbf T}Q_{\mathbf T_1},\mathbf k}
=e^{-i\mathbf k\cdot\mathbf T_1}
 J_i^{\nu_{\mathbf T-\mathbf T_1}Q,\mathbf k}
-\frac12e^{-i\mathbf k\cdot\mathbf T}
 \sum_{R,\mathbf W}V_{QR}^{\mathbf W+\mathbf T-\mathbf T_1}
 D_i^{\nu_{\mathbf0}R_{\mathbf W},\mathbf k}.
}
$$

The correction has the same overall phase $e^{-i\mathbf k\cdot\mathbf T}$ as in the first case, because

$$
e^{-i\mathbf k\cdot\mathbf T_1}
e^{-i\mathbf k\cdot(\mathbf T-\mathbf T_1)}
=e^{-i\mathbf k\cdot\mathbf T}.
$$

Its metric image is now $\mathbf W+\mathbf T-\mathbf T_1$, the cell translation of $R_{\mathbf W+\mathbf T}$ relative to $Q_{\mathbf T_1}$. The same `D[W]` is reused in both cases.

For a fixed left pair and output image, the two selections are therefore:

| Contribution, in order | Auxiliary slice | Anchored image $\mathbf t$ | Required $H$ |
|---|---|---|---|
| First: $\mathbf U=\mathbf0$ | $Q\in\mathcal A(A)$ | $\mathbf T$ | $H_0[\mathbf T]$ |
| Then: $\mathbf U=\mathbf T_1$ | $Q\in\mathcal A(B)$ | $\mathbf T-\mathbf T_1$ | $e^{-i\mathbf k\cdot\mathbf T_1}H_0[\mathbf T-\mathbf T_1]$ |

When $\mathbf T_1=\mathbf0$ and $A\ne B$, both rows use the same image block with different atomic auxiliary slices. When also $A=B$, the second contribution is omitted to count the auxiliary union once.

For an arbitrary stored auxiliary image $\mathbf U$, the same identity reads

$$
\boxed{
H_i^{\nu_{\mathbf T}Q_{\mathbf U},\mathbf k}
=e^{-i\mathbf k\cdot\mathbf U}
 H_{0,i}^{\nu_{\mathbf T-\mathbf U}Q,\mathbf k}.
}
$$

`data.h(T, U)` implements this identity. It returns the tuple of $H$ arrays for the supplied actual output and auxiliary images. Both arguments use integer lattice triples; the phase is evaluated after conversion to Cartesian translations.

The actual execution builds all of `D`, constructs the shared `h0` once, and then uses this identity in `get_l()`. For each left pair, `_build_d()` accumulates its $\mathbf U=\mathbf0$ contribution before its $\mathbf U=\mathbf T_1$ contribution. After all pairs have been accumulated, the outer loop in `get_l()` visits each stored `D[U]` once, starting with `D[0]`; each block is contracted with all stored `h0[t]` blocks. The two rows above describe how each pair's two coefficient blocks select their matching $H$; $H_0$ is shared across all pairs.

In the output loop, $\mathbf t$ is the stored image of $\nu$ relative to $Q$. Consequently $\mathbf T=\mathbf U+\mathbf t$: the first case contributes to $\mathbf T=\mathbf t$, and the second to $\mathbf T=\mathbf T_1+\mathbf t$. The same translation must be used for both the bare and fitted parts of $H$.

---

The first exchange half is formed in real space exactly as supplied:

$$
\boxed{
L_{\mu\nu}^{\mathbf0\mathbf T}
=\frac1{N_k}\sum_{\mathbf k,i}\sum_{Q,\mathbf U}
 \left(D_i^{\mu_{\mathbf0}Q_{\mathbf U},\mathbf k}\right)^*
 H_i^{\nu_{\mathbf T}Q_{\mathbf U},\mathbf k}.
}
$$

Substituting the translated $H$ gives an implementation using the stored image dictionaries:

$$
L_{\mu\nu}^{\mathbf0\mathbf T}
=\frac1{N_k}\sum_{\mathbf k,Q,i,\mathbf U}
 \left(D_i^{\mu_{\mathbf0}Q_{\mathbf U},\mathbf k}\right)^*
 e^{-i\mathbf k\cdot\mathbf U}
 H_{0,i}^{\nu_{\mathbf T-\mathbf U}Q,\mathbf k}.
$$

For each stored pair of image keys $\mathbf U$ and $\mathbf t$, the code contributes to the output image

$$
\mathbf T=\mathbf U+\mathbf t.
$$

The contraction for that image pair and one density k-point is schematically

```python
L[U + t] += phase * np.einsum(
    'mqi,nqi->mn', D[U][ik].conj(), H0[t][ik]) / nkpts
# phase = exp(-1j * k.dot(U_cartesian))
```

The auxiliary index and the occupied-orbital index are contracted. The two external AO indices remain. The complex conjugation is on $D$, exactly as in the original equation.

Expanding the two intermediates recovers the original density expression. Their phases and orbital factors combine as

$$
\begin{aligned}
\frac1{N_k}\sum_{\mathbf k,i}
 \left(M_{\lambda i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf T_1}\right)^*
 M_{\sigma i}^{\mathbf k}e^{-i\mathbf k\cdot\mathbf T_2}
&=\frac1{N_k}\sum_{\mathbf k}
 e^{-i\mathbf k\cdot(\mathbf T_2-\mathbf T_1)}
 P_{\lambda\sigma}^{\mathbf k}\\
&=P_{\lambda\sigma}^{\mathbf T_2-\mathbf T_1}.
\end{aligned}
$$

Consequently,

$$
\begin{aligned}
L_{\mu\nu}^{\mathbf0\mathbf T}
={}&\sum_{\lambda,\sigma,\mathbf T_1,\mathbf T_2}
 \sum_{Q,\mathbf U}
 C_{Q_{\mathbf U}}^{\mu_{\mathbf0}\lambda_{\mathbf T_1}}
 \Bigg[(Q_{\mathbf U}|\sigma_{\mathbf T_2}\nu_{\mathbf T})_{\mathrm{SR}}\\
&\hspace{3em}-\frac12\sum_{R,\mathbf V}
 C_{R_{\mathbf V}}^{\sigma_{\mathbf T_2}\nu_{\mathbf T}}
 (Q_{\mathbf U}|R_{\mathbf V})_{\mathrm{SR}}\Bigg]
 P_{\lambda\sigma}^{\mathbf T_2-\mathbf T_1}.
\end{aligned}
$$

There is exactly one factor $1/N_k$, inherited from the density-matrix Fourier series. The sums over actual auxiliary images and actual output images do not introduce additional averaging factors.

`get_l(m, ...)` returns the matrices $L^{\mathbf0\mathbf T}$ in a dictionary keyed by their integer lattice translations. Its contents are independent of the eventual output band k-points.

When `intermediates=data` is supplied, `get_l` uses the factors and k-points stored in `data`; separate `m` and `kpts` arguments are unused.

---

For Hermitian $P^{\mathbf k}$,

$$
\left(P_{\lambda\sigma}^{\mathbf T}\right)^*
=P_{\sigma\lambda}^{-\mathbf T}.
$$

Translation invariance, symmetry of real localized AO products, and this density identity give the supplied robust exchange reconstruction:

$$
\boxed{
K_{\mu\nu}^{\mathbf0\mathbf T}
=L_{\mu\nu}^{\mathbf0\mathbf T}
+\left(L_{\nu\mu}^{\mathbf0,-\mathbf T}\right)^*.
}
$$

In matrix notation, `get_k_realspace(l)` constructs

```python
K[T] = L[T] + L[-T].conj().T
```

The output image set includes both each retained translation and its negative. Missing $L$ blocks are zero. This guarantees

$$
K^{\mathbf0,-\mathbf T}=\left(K^{\mathbf0\mathbf T}\right)^\dagger
$$

for the finite representation, without requiring time-reversal symmetry of the sampled density matrices.

Expanding the two halves gives the robust three-term approximation

$$
(\widetilde{\mu\lambda}|\sigma\nu)_{\mathrm{SR}}
+(\mu\lambda|\widetilde{\sigma\nu})_{\mathrm{SR}}
-(\widetilde{\mu\lambda}|\widetilde{\sigma\nu})_{\mathrm{SR}}.
$$

Each $L$ contributes half of the fitted/fitted subtraction. Together they give the full subtraction rather than a fitted/fitted-only exchange approximation.

Only after these real-space exchange blocks have been constructed does the implementation evaluate

$$
\boxed{
K_{\mu\nu}(\mathbf k_{\mathrm{out}})
=\sum_{\mathbf T}e^{i\mathbf k_{\mathrm{out}}\cdot\mathbf T}
 K_{\mu\nu}^{\mathbf0\mathbf T}.
}
$$

There is no $1/N_k$ in this final transform. Each requested output k-point is evaluated separately. The output points can differ from the density sampling points.

The translation sums are not reduced modulo a Born–von Kármán mesh. Input k-point sampling defines the density Fourier series; it does not identify different localized auxiliary centers or restrict the exchange output to one finite set of BvK image representatives.

---

The infinite real-space sums require numerical truncations. The implementation separates the three relevant domains:

| Setting | Domain it controls |
|---|---|
| `pair_rcut` | Retained localized AO pairs and therefore the support of fitted coefficients and $D$ |
| `integral_rcut` | AO-image domain for the bare three-center contribution to anchored $H$ |
| `metric_rcut` | Auxiliary-image domain for the real-space two-center metric used in the fitted correction |
| `precision` | Target used by the lattice-cutoff estimators, internally tightened by a factor of 100 |
| `linear_dep_threshold` | Retained eigenspace of each normalized local fitting metric |
| `auxbasis` | Atomic auxiliary functions available to the local fits |
| `max_memory` | Maximum size in MB of one bare three-center atom/image block; not a cap on total intermediate storage |

When explicit radii are omitted, conservative AO-overlap and short-range integral estimates from PySCF are used. The bare-integral estimate accounts for both AO decay and the short-range three-center interaction; the auxiliary metric uses a short-range two-center estimate. These are estimates rather than rigorous bounds on the accumulated exchange error.

The bare and fitted parts of $H$ must have independent domains. For example, decreasing `pair_rcut` sets some local coefficients to zero, but the corresponding bare three-center integral can still contribute to $J$. The implementation does not discard that contribution merely because the pair lacks a retained fit.

The correction can generate anchored output images $\mathbf t=\mathbf V-\mathbf W$, combining metric-image keys $\mathbf V$ with $D$-image keys $\mathbf W$. The final $L$ then has output images $\mathbf T=\mathbf U+\mathbf t$. Thus the output exchange range is generated by the actual contraction supports; it is not assumed equal to any one of the input cutoffs.

The displayed equations describe the full lattice sums. The finite implementation follows the same tensor contractions with truncated domains and regularized local solves. Its numerical accuracy must be checked by enlarging the separate lattice cutoffs, tightening cutoff-estimator precision when radii are automatic, and varying the fitting basis or metric threshold as appropriate. With explicit radii, changing `precision` does not enlarge those domains. The molecular integrals also retain libcint's own numerical screening. Tightening cutoff-estimator precision alone does not remove the intrinsic PARI fitting error.

---

The kernel convention is the literal $\operatorname{erfc}(\omega r)/r$ interaction. In three dimensions its Fourier transform is

$$
\widehat w_{\mathrm{SR}}(\mathbf G)
=\frac{4\pi}{G^2}\left(1-e^{-G^2/(4\omega^2)}\right),
\qquad
\widehat w_{\mathrm{SR}}(\mathbf0)=\frac{\pi}{\omega^2}.
$$

The finite zero-mode contribution is retained by the real-space integrals. No zero-average background subtraction or Ewald exchange correction is applied. Accordingly `exxdiv` must be `None`. Comparisons with a higher-level GDF/RSDF result require matching both the kernel convention and the fitting approximation.

The public `omega` parameter is positive and measured in inverse Bohr. Molecular short-range integral calls use `-omega`, matching libcint's convention. `get_jk` is an exchange-only wrapper: requesting Coulomb $J$ is unsupported, and its optional signed range parameter must select the same fixed short-range kernel.

The returned $K$ has the usual positive exchange-operator convention. Neither an exchange minus sign nor an RHF factor of $1/2$ is included; those belong in the caller's Fock-matrix assembly.

---

**A comparison with existing PySCF short-range exchange.** For a periodic `KRHF` or `KRKS` object, the standard call is `mf.get_k(dm_kpts=dm_kpts, omega=-omega)`, with `mf.exxdiv=None`. The negative argument selects $\operatorname{erfc}(\omega r)/r$; the `PARI` constructor takes the positive magnitude.

There is a zero-mode convention to match in this checkout. Standard three-dimensional periodic GDF removes the finite $\mathbf G+\mathbf q=\mathbf0$ component, whereas `PARI` retains $\widehat w_{\mathrm{SR}}(0)=\pi/\omega^2$. For **a 3D cell, a full list of equally weighted distinct k-points, and exchange evaluated at those same k-points**, its exact exchange contribution is

$$
\Delta K^{\mathbf k}
=\frac{\pi}{\omega^2\Omega N_k}
 S^{\mathbf k}P^{\mathbf k}S^{\mathbf k},
\qquad
S^{\mathbf k}_{\mu\nu}=\langle\mu_{\mathbf k}|\nu_{\mathbf k}\rangle,
\qquad \Omega=\texttt{cell.vol}.
$$

Only the source k-point equal to the output k-point contributes to this zero mode, giving the factor $1/N_k$. The two AO-pair integrals at zero momentum are overlap matrices, giving $SPS$. Setting `exxdiv=None` disables the Ewald exchange correction; the finite short-range term above still needs to be added for this comparison. This formula is specific to the sampling and dimension stated here.

The following small example uses the same density for both implementations. An initial-guess density is sufficient to compare exchange builders; for an existing converged calculation, replace it with `dm_kpts = mf.make_rdm1()`.

```python
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.df.pari import PARI

cell = gto.M(
    a=np.eye(3) * 10.0,
    atom='H 0 0 0; H 1.4 0 0',
    unit='Bohr',
    basis='sto-3g',
    precision=1e-9,
    verbose=0,
)
kpts = cell.make_kpts([2, 1, 1])
omega = 0.5  # inverse Bohr
auxbasis = 'weigend'

# Existing PySCF Gaussian density fitting, using one fixed density.
mf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit(auxbasis=auxbasis)
dm_kpts = mf.get_init_guess()  # shape (nkpts, nao, nao)
vk_gdf_zero = mf.get_k(dm_kpts=dm_kpts, omega=-omega)

# Restore the finite zero mode to match PARI's literal erfc kernel.
s = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
vk_g0 = (
    np.pi / (omega**2 * cell.vol * len(kpts))
    * np.einsum('kmp,kpq,kqn->kmn', s, dm_kpts, s)
)
vk_gdf = vk_gdf_zero + vk_g0

pari = PARI(cell, kpts=kpts, auxbasis=auxbasis, omega=omega)
vk_pari = pari.get_k(dm_kpts)

delta = vk_pari - vk_gdf
print('K shape:', vk_pari.shape)
print('max |K_PARI - K_GDF|:', np.max(np.abs(delta)))
print('relative Frobenius difference:',
      np.linalg.norm(delta) / np.linalg.norm(vk_gdf))
```

This example was run with the repository's `.venv`. It produced shape `(2, 2, 2)`, a maximum matrix-element difference of approximately $1.76\times10^{-5}$ Hartree, and a relative Frobenius difference of $3.20\times10^{-5}$. These are example results, not an equality tolerance or an error bound for other cells.

GDF fits AO products using the full periodic auxiliary basis; PARI uses the robust two-atom local fits derived above. The reference here is the GDF fit with its zero mode removed, plus the exact zero-mode term. Adding that term matches the kernel convention but does not make the fitting constructions identical. Differences can therefore remain after converging lattice cutoffs, even with the same `auxbasis`.

For a reference without Gaussian auxiliary fitting error, use PySCF's analytical Fourier-transform builder `AFTDF`. Continuing the example above:

```python
from pyscf.pbc import df

aft = df.AFTDF(cell, kpts=kpts)
# aft.mesh = [101, 101, 101]  # Increase the reciprocal mesh to check convergence.
_, vk_aft_zero = aft.get_jk(
    dm_kpts, with_j=False, with_k=True, omega=-omega, exxdiv=None)
vk_aft = vk_aft_zero + vk_g0
print('max |K_PARI - K_AFT|:', np.max(np.abs(vk_pari - vk_aft)))
```

`AFTDF` uses the same zero-mode convention here. Its accuracy depends on the reciprocal mesh and AO lattice sums. In this example the default mesh was `[81, 81, 81]` and the PARI–AFT maximum difference was approximately $1.62\times10^{-4}$ Hartree. To check mesh convergence, create a fresh `AFTDF` object for each mesh and set `aft.mesh` before its first `get_jk` call, since the range-separated builder is cached. Converge the AFT mesh and PARI lattice cutoffs separately before attributing a residual difference to the local fit. Neither reference comparison should assert machine-precision equality for a general finite auxiliary basis.

The GDF example uses the supported SR route through `density_fit()` and `rsdf_builder.py`. The separate public `RSDF`/`RSGDF` class in this checkout does not support this `omega=-omega` route. Both examples compare the positive $K_{\mathrm{SR}}$ matrices; their contribution to a restricted Fock matrix is $-\tfrac12K_{\mathrm{SR}}$.

---

A direct orbital-based calculation is:

```python
import numpy as np
from pyscf.pbc.df.pari import PARI

pari = PARI(cell, kpts=kpts, auxbasis='weigend', omega=0.5)
pari.precision = 1e-10

# Ordinary PySCF MO coefficients and occupation vectors, one entry per k-point.
vk = pari.get_k_from_mo(mo_coeff, mo_occ)

# The equivalent explicit M convention used in the equations.
m = [u.conj() * np.sqrt(f)[None, :]
     for u, f in zip(mo_coeff, mo_occ)]
vk = pari.get_k_from_m(m)

# The density route factors P and calls the same MO-intermediate engine.
vk = pari.get_k(dm_kpts)
```

The direct M/MO interfaces operate on one density set at a time. Their output shapes are:

| Output-point argument | `get_k_from_m` / `get_k_from_mo` result |
|---|---|
| `kpts_band=None` | `(nkpts, nao, nao)`, including when `nkpts=1` |
| A single vector with shape `(3,)` | `(nao, nao)` |
| An array with shape `(nband, 3)` | `(nband, nao, nao)` |

`get_k(dm_kpts)` also accepts density batches. Without `kpts_band`, its output has exactly the density input's shape. With explicit band points, it preserves the density-batch axes and replaces the sampled k-point axis with the band axis, or removes that axis for a single band vector. For example, an unrestricted density of shape `(2, nkpts, nao, nao)` produces exchange of the same shape by default.

The individual mathematical stages can also be inspected:

```python
data = pari.get_intermediates(m)
l_real = pari.get_l(intermediates=data)
k_real = pari.get_k_realspace(l_real)

# D for the reference-cell auxiliary functions at the first input k-point.
d_zero = data.d[(0, 0, 0)][0]

# H for external AO image T and auxiliary image U, as a tuple over k-points.
T = (1, 0, 0)
U = (0, 0, 0)
h_at_T_U = data.h(T, U)
```

For clarity, the final Fourier transform of the returned real-space exchange dictionary can be written explicitly as

```python
keys = list(k_real)
translations = np.asarray(keys) @ cell.lattice_vectors()
blocks = np.stack([k_real[key] for key in keys])
phases = np.exp(1j * np.asarray(band_kpts).reshape(-1, 3) @ translations.T)
vk_band = np.einsum('kt,tmn->kmn', phases, blocks)
```

This last expression is independent of how the density was supplied. The $D$, $H$, and $L$ constructions depend on the density sampling k-points; the final Fourier phases depend on the requested output k-points.

Call `reset(cell)` after changing the existing cell's geometry or orbital basis. Local fits and integral domains must be rebuilt when their defining parameters change. An intermediates object belongs to the cell, parameters, density factors, and k-point list used to create it; it should not be reused after changing any of them.

This remains a correctness-oriented real-space implementation. Explicit AO-image clusters, local coefficient blocks, auxiliary metric blocks, and the $D/H$ dictionaries can all consume substantial memory and time. Following the occupied-orbital equations does not by itself establish a performance improvement; locality, screening, blocking, and numerical convergence still determine the practical cost.

The revised implementation was checked in the repository's `.venv` against independently fitted molecular-lattice tensors, comparing the intermediates themselves rather than only the final exchange matrix:

| Validation | Observed maximum discrepancy |
|---|---|
| Actual-image $D$ | $7.2\times10^{-16}$ |
| Anchored $H_0$ | $7.8\times10^{-16}$ |
| Translated $H(T,U)$ | $1.4\times10^{-15}$ |
| Real-space $L^{\mathbf0\mathbf T}$ | $4.7\times10^{-16}$ |
| Final Fourier $K$ in the same tests | $8.9\times10^{-16}$ |
| Primitive k-point cells versus equivalent Gamma supercells | $1.8\times10^{-15}$ |
| Automatic cutoffs versus the independently evaluated reciprocal formulation in 1D–3D test cells | $5.0\times10^{-14}$ |

The intermediate tests include deliberately small fitting radii, including `pair_rcut=0`, while retaining the independent bare-integral range. Additional checks cover fractional occupations, different occupied ranks at different k-points, complex densities without time-reversal symmetry, density batches, zero density, redundant auxiliary functions, Cartesian functions, and rejection of stale intermediates after a rebuild. These checks validate the implementation and its conventions; they do not bound PARI fitting errors or lattice truncation errors for other systems.

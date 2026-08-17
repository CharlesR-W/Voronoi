# Approximate Factorization of Voronoi-Quantized Residual Computation

Assume residual activations have already been quantized into a finite set of Voronoi cells

\[
\mathcal C=\{1,\ldots,N\},
\]

with coarse state

\[
c=q(h).
\]

Introduce the formal state space

\[
\mathcal H_{\mathcal C}
=
\operatorname{span}\{|c\rangle:c\in\mathcal C\}
\simeq \mathbb C^N.
\]

The network induces a family of transition operators on this space. Depending on the experiment these might be layer-to-layer transitions, transitions conditioned on token/context class, or other computational primitives:

\[
P_\alpha(j|i)
=
\Pr(c_{\mathrm{out}}=j\mid c_{\mathrm{in}}=i,\alpha).
\]

The basic computational object is the algebra

\[
\mathcal A
=
\operatorname{alg}\{P_\alpha\}.
\]

The goal is to determine whether this apparently large discrete computation contains approximate symmetries, coarse variables, or approximately factored subsystems.

## 1. Approximate symmetries / commutant

A transformation \(X\) of cell space is an exact symmetry when

\[
[X,P_\alpha]=0
\qquad\forall\alpha.
\]

Define the commutator map

\[
\mathcal C(X)
=
\big([X,P_1],\ldots,[X,P_m]\big)
\]

and corresponding symmetry energy

\[
E_{\rm sym}(X)
=
\sum_\alpha
\|[X,P_\alpha]\|^2.
\]

Then:

- zero modes of \(\mathcal C^\dagger\mathcal C\) are exact commutant elements;
- low eigenmodes are approximate symmetries;
- permutation-like low modes correspond most directly to approximate relabelings/interchangeabilities of Voronoi cells.

If desired, include the residual geometry through the centroid Gram matrix

\[
K_{ij}=v_i^\top v_j
\]

and search for transformations preserving both geometry and computation:

\[
E(X)
=
\lambda_K\|[X,K]\|^2
+
\sum_\alpha\lambda_\alpha\|[X,P_\alpha]\|^2.
\]

## 2. Approximate factorization

A stronger structural hypothesis is that a cell secretly encodes several latent discrete variables,

\[
c\leftrightarrow(z_1,\ldots,z_k),
\]

so that

\[
\mathcal H_{\mathcal C}
\approx
\mathcal H_1\otimes\cdots\otimes\mathcal H_k.
\]

The strongest possible factorization would make transitions product operators,

\[
P_\alpha\approx
P_{\alpha,1}\otimes\cdots\otimes P_{\alpha,k},
\]

but this is probably too restrictive.

More realistically, seek a decomposition into local operations plus interactions:

\[
L_\alpha
\approx
\sum_i L_{\alpha,i}
+
\sum_{i<j}L_{\alpha,ij}
+
\sum_{i<j<k}L_{\alpha,ijk}
+\cdots,
\]

where

\[
L_{\alpha,i}
=
I\otimes\cdots\otimes
\tilde L_{\alpha,i}
\otimes\cdots\otimes I.
\]

Small higher-order terms imply an approximately factored computation.

A natural interaction score is, e.g.,

\[
\epsilon_{\rm int}
=
\frac{
\sum_{\alpha,i<j}\|L_{\alpha,ij}\|^2+\cdots
}{
\sum_\alpha\|L_\alpha\|^2
}.
\]

This gives a graded notion of subsystem independence rather than requiring an exact tensor product.

## 3. Approximately commuting subalgebras

The intrinsic operator-algebraic formulation is to search for subalgebras

\[
\mathcal A_1,\ldots,\mathcal A_k
\]

such that

\[
[\mathcal A_i,\mathcal A_j]\approx0
\qquad (i\neq j)
\]

and together they explain most of the observed transition algebra.

For an exact bipartite factorization,

\[
\mathcal A_1
\simeq
\operatorname{End}(\mathcal H_1)\otimes I,
\]

\[
\mathcal A_2
\simeq
I\otimes\operatorname{End}(\mathcal H_2).
\]

This is more general than searching for a large global commutant: a computation can have meaningful subsystems even when the commutant of the full algebra is trivial.

## 4. Operator-Schmidt diagnostic

Given a candidate factorization

\[
\mathcal H=\mathcal H_A\otimes\mathcal H_B,
\]

decompose a transition operator as

\[
P_\alpha
=
\sum_r
\sigma_{\alpha r}
A_{\alpha r}\otimes B_{\alpha r}.
\]

The operator-Schmidt spectrum \(\{\sigma_{\alpha r}\}\) measures how entangling/non-factorized the operation is.

A nearly local computation should have most of its weight in terms of the form

\[
A\otimes I,
\qquad
I\otimes B,
\]

with relatively little weight in genuinely joint operators.

This provides a continuous measure of factorization quality for any proposed tensor-product structure.

## 5. Discovering the latent factors

Rather than assuming a decomposition of the cells, search for coordinates

\[
c\mapsto(z_1(c),\ldots,z_k(c))
\]

that simultaneously simplify geometry and dynamics.

Useful signatures include:

1. **Approximate Cartesian-product transition graph**
   \[
   G_{\mathcal C}\approx G_1\square G_2,
   \]
   corresponding to
   \[
   L\approx L_1\otimes I+I\otimes L_2.
   \]

2. **Approximately commuting coordinate observables**
   \[
   [Z_i,Z_j]\approx0,
   \]
   whose joint approximate eigenvalues label cells by tuples
   \[
   (z_1,\ldots,z_k).
   \]

3. **Low interaction complexity**
   after rewriting transition operators in the candidate factor coordinates.

The resulting interaction terms

\[
L_{ij},L_{ijk},\ldots
\]

define an effective interaction graph or hypergraph among computational variables.

## 6. Multiscale / “zoomable” computation

We can make the construction hierarchical by introducing a sequence of increasingly fine partitions of cell space,

\[
\Pi_0
\prec
\Pi_1
\prec
\cdots
\prec
\Pi_L,
\]

where \(\Pi_0\) groups many Voronoi cells into coarse macrostates and \(\Pi_L\) approaches the original cell partition.

Each level defines a coarse state space

\[
\mathcal H^{(0)},
\mathcal H^{(1)},
\ldots,
\mathcal H^{(L)}
\]

and a coarse-graining map

\[
C_{\ell}:
\mathcal H^{(\ell+1)}
\to
\mathcal H^{(\ell)}.
\]

Transition operators are correspondingly coarse-grained,

\[
P_\alpha^{(\ell)}
\approx
C_\ell
P_\alpha^{(\ell+1)}
R_\ell,
\]

with \(R_\ell\) some reconstruction/inclusion map.

This gives a filtration of computational descriptions:

\[
\mathcal A^{(0)}
\subseteq
\mathcal A^{(1)}
\subseteq
\cdots
\subseteq
\mathcal A^{(L)},
\]

where increasing \(\ell\) reveals progressively finer distinctions.

The intended semantics is something like:

\[
\text{coarse level:}\qquad A\to B,
\]

while after refining \(A\),

\[
A=A_0\cup A_1,
\]

we may discover

\[
A_0\to B,
\qquad
A_1\to C.
\]

Thus an apparent coarse rule

\[
A\to B
\]

is really a quotient computation that ignores a finer condition.

A useful way to formalize when refinement is necessary is by the failure of **lumpability**. A coarse partition is adequate when cells grouped into the same macrostate have nearly identical transition probabilities into every other macrostate. For cells \(i,j\) in the same coarse block \(A\),

\[
\sum_{k\in B}P(k|i)
\approx
\sum_{k\in B}P(k|j)
\]

for every coarse block \(B\).

The deviation

\[
E_{\rm lump}(A)
=
\sum_{i,j\in A}
\sum_B
\left|
P(B|i)-P(B|j)
\right|^2
\]

is therefore a natural trigger for “zooming in.” Large error says that the coarse concept \(A\) hides computationally important subcases.

One can then recursively refine only the offending regions, giving an adaptive hierarchy rather than uniformly increasing resolution everywhere.

The overall picture is therefore

\[
\boxed{
\text{Voronoi cells}
\rightarrow
\text{transition algebra}
\rightarrow
\text{approximate symmetries / commuting subalgebras}
\rightarrow
\text{latent tensor factors}
\rightarrow
\text{local + interaction decomposition},
}
\]

with a parallel multiscale hierarchy

\[
\boxed{
\text{coarse quotient computation}
\rightarrow
\text{identify non-lumpable states}
\rightarrow
\text{refine locally}
\rightarrow
\text{recover conditional exceptions and finer computations}.
}
\]

The main experimental target is not necessarily an exact symmetry or exact tensor product, but a low-complexity description in which most computation is captured by a few approximately independent factors plus sparse, measurable interaction terms.
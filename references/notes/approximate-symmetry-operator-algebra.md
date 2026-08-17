# Approximate Symmetry for Computation

The main conceptual move is this:

> **Do not usually make the group itself approximate. Keep the symmetry relation exact as a reference object, and measure how far your computation is from satisfying it.**

For computational purposes, the natural language is therefore less “approximate group theory” than **representation theory + operator algebras + perturbation/stability theory**.

This gives you quantitative notions such as

\[
\text{symmetry-breaking strength},
\qquad
\text{distance to the nearest symmetric computation},
\qquad
\text{approximate conserved quantities},
\]

and, importantly, spectral methods for *discovering* approximate symmetries.

---

## 1. Exact symmetry as a commutant

Suppose a computation acts on a state space \(V\), and for now suppose one step is a linear operator

\[
T:V\to V.
\]

Let a group \(G\) act on \(V\) through a representation

\[
g\mapsto U_g.
\]

Then \(G\) is a symmetry of \(T\) if

\[
U_g T U_g^{-1}=T,
\]

or equivalently

\[
[T,U_g]=0
\qquad \forall g\in G.
\]

This immediately introduces the **commutant**. For a set of operators \(\mathcal S\),

\[
\mathcal S'
=
\{X:[X,S]=0\;\forall S\in\mathcal S\}.
\]

Thus

\[
T\in U(G)'
\]

means that \(T\) respects the symmetry \(G\).

Conversely, if your computation consists of many elementary operations

\[
T_1,\ldots,T_m,
\]

form the operator algebra

\[
\mathcal A=\operatorname{alg}(T_1,\ldots,T_m).
\]

Then

\[
\mathcal A'
\]

is the algebra of transformations invisible to the computation: operators commuting with **every possible composition** of its primitives.

The unitary elements of \(\mathcal A'\) are therefore exact symmetries of the computation.

This is why the algebraic viewpoint is useful. The fundamental object need not be a group at all. You can start with a computation, construct its operator algebra, and then ask what commutes with it.

For maps between different spaces, the appropriate condition is an **intertwining relation**

\[
U_{\rm out}(g)T
=
T U_{\rm in}(g).
\]

Everything below generalizes by replacing commutation with intertwining.

---

## 2. Make symmetry approximate by measuring the residual

The most immediate definition is

\[
\epsilon_g(T)
=
\|U_gTU_g^\dagger-T\|.
\]

Then an aggregate symmetry defect might be

\[
\epsilon_G^2(T)
=
\int_G
\|U_gTU_g^\dagger-T\|^2\,dg.
\]

For a finite group,

\[
\epsilon_G^2(T)
=
\frac1{|G|}
\sum_{g\in G}
\|U_gTU_g^\dagger-T\|^2.
\]

There is already an important modeling choice here: **which norm?**

The operator norm,

\[
\|A\|_{\rm op},
\]

says the symmetry approximately holds for every state.

The Hilbert-Schmidt/Frobenius norm,

\[
\|A\|_{\rm HS}^2=\operatorname{Tr}(A^\dagger A),
\]

measures an average violation over directions.

For computation or neural systems you may want a data-weighted quantity. If states have covariance \(\Sigma\),

\[
\|A\|_\Sigma^2
=
\mathbb E_x\|Ax\|^2
=
\operatorname{Tr}(A^\dagger A\Sigma).
\]

Then you can say:

> The computation may be very far from symmetric globally, while being extremely symmetric on the subspace of states it actually visits.

That distinction is likely essential for applications to learned computation.

---

## 3. The most useful construction: project onto the symmetric computation

Suppose \(G\) is compact. Define the **group twirl**

\[
\boxed{
\mathcal P_G(T)
=
\int_G U_gTU_g^\dagger\,dg.
}
\]

For a finite group this is simply

\[
\mathcal P_G(T)
=
\frac1{|G|}
\sum_g U_gTU_g^\dagger.
\]

It satisfies

\[
U_h\mathcal P_G(T)U_h^\dagger
=
\mathcal P_G(T),
\]

because Haar measure is invariant under \(g\mapsto hg\).

Moreover,

\[
\mathcal P_G^2=\mathcal P_G.
\]

With the Hilbert-Schmidt inner product it is self-adjoint, so it is literally the **orthogonal projection onto the commutant** \(U(G)'\).

Therefore

\[
\boxed{
T_{\rm sym}=\mathcal P_G(T)
}
\]

is the nearest exactly \(G\)-symmetric operator in Hilbert-Schmidt distance, and

\[
T
=
T_{\rm sym}+T_{\rm break}
\]

gives a canonical decomposition into symmetric and symmetry-breaking pieces.

Thus a very clean definition of approximate symmetry is simply

\[
\boxed{
d_G(T)
=
\|T-\mathcal P_G(T)\|.
}
\]

This is already most of what I would use operationally.

### Tiny example

Take the reflection

\[
R=
\begin{pmatrix}
1&0\\
0&-1
\end{pmatrix},
\qquad
T=
\begin{pmatrix}
a&b\\
c&d
\end{pmatrix}.
\]

Then

\[
RTR
=
\begin{pmatrix}
a&-b\\
-c&d
\end{pmatrix},
\]

so

\[
\mathcal P(T)
=
\frac12(T+RTR)
=
\begin{pmatrix}
a&0\\
0&d
\end{pmatrix}.
\]

The off-diagonal part

\[
T_{\rm break}
=
\begin{pmatrix}
0&b\\
c&0
\end{pmatrix}
\]

is precisely the symmetry-breaking component.

So approximate reflection symmetry simply means

\[
|b|^2+|c|^2\ll |a|^2+|d|^2.
\]

This is much more useful than saying vaguely that the system “sort of has a \(\mathbb Z_2\) symmetry.”

---

## 4. Lie groups: commutators become a symmetry energy

For a continuous symmetry, write

\[
U(\theta)=e^{-i\theta Q}
\]

for Hermitian generator \(Q\).

Then

\[
U(\theta)TU(\theta)^\dagger
=
T-i\theta[Q,T]+O(\theta^2).
\]

Hence infinitesimal symmetry is

\[
[Q,T]=0,
\]

and a natural approximate-symmetry energy is

\[
\boxed{
\mathcal E_Q(T)
=
\|[Q,T]\|_{\rm HS}^2.
}
\]

For generators \(Q_a\),

\[
\mathcal E(T)
=
\sum_a\|[Q_a,T]\|_{\rm HS}^2.
\]

Now define the superoperator

\[
\boxed{
\mathcal L_{\rm sym}(T)
=
\sum_a[Q_a,[Q_a,T]].
}
\]

Then

\[
\langle T,\mathcal L_{\rm sym}T\rangle_{\rm HS}
=
\sum_a\|[Q_a,T]\|_{\rm HS}^2
\geq0.
\]

So \(\mathcal L_{\rm sym}\) is a positive operator acting **on the space of operators**.

Its kernel is

\[
\ker\mathcal L_{\rm sym}
=
\{T:[Q_a,T]=0\ \forall a\},
\]

the exactly invariant algebra.

This is perhaps the nicest spectral formulation of approximate symmetry:

> **Exact symmetries are zero modes of a symmetry Laplacian; approximate symmetries are low-energy modes.**

This operator is closely related to the adjoint-representation Casimir.

---

## 5. Approximate selection rules fall out immediately

Suppose

\[
Q|i\rangle=q_i|i\rangle.
\]

Then

\[
[Q,T]_{ij}
=
(q_i-q_j)T_{ij},
\]

and therefore

\[
\boxed{
\|[Q,T]\|_{\rm HS}^2
=
\sum_{ij}(q_i-q_j)^2|T_{ij}|^2.
}
\]

Exact symmetry says

\[
q_i\neq q_j
\quad\Longrightarrow\quad
T_{ij}=0.
\]

This is the usual selection rule.

But now you have an **approximate selection rule**. If

\[
|q_i-q_j|\geq\Delta
\]

for some set of pairs, then

\[
\sum_{|q_i-q_j|\geq\Delta}|T_{ij}|^2
\leq
\frac{\|[Q,T]\|_{\rm HS}^2}{\Delta^2}.
\]

So a small commutator quantitatively bounds the amount of mixing between sectors with substantially different charges.

This is exactly the sort of statement you want from approximate symmetry: not merely “the symmetry is slightly broken,” but

\[
\text{symmetry violation}
\quad\Rightarrow\quad
\text{controlled violation of its consequences}.
\]

More generally, if the lowest nonzero eigenvalue of \(\mathcal L_{\rm sym}\) is \(\lambda_1\), then

\[
\boxed{
\|T-\mathcal P_G T\|_{\rm HS}^2
\leq
\frac{\mathcal E(T)}{\lambda_1}.
}
\]

This is just a Poincaré-type inequality obtained by expanding \(T\) in eigenvectors of \(\mathcal L_{\rm sym}\).

The spectral gap \(\lambda_1\) tells you how rigid the symmetry is.

---

## 6. Representation theory tells you what the commutant looks like

Suppose

\[
V
\simeq
\bigoplus_\lambda
M_\lambda\otimes V_\lambda,
\]

where \(V_\lambda\) is an irreducible representation and \(M_\lambda\) is its multiplicity space.

The group acts as

\[
U_g
=
\bigoplus_\lambda
I_{M_\lambda}\otimes D_\lambda(g).
\]

Schur's lemma then gives

\[
\boxed{
U(G)'
=
\bigoplus_\lambda
\operatorname{End}(M_\lambda)\otimes I_{V_\lambda}.
}
\]

So an exactly symmetric computation cannot arbitrarily mix representation sectors. It can only perform arbitrary computation inside the multiplicity spaces.

An approximate symmetry means approximately this block structure.

This gives a very concrete interpretation of the twirl:

\[
T
\longmapsto
\mathcal P_G(T)
\]

throws away precisely those components of \(T\) transforming nontrivially under the adjoint action of \(G\).

---

## 7. Discovering symmetries rather than assuming them

This is where I think the idea becomes especially useful for your computational interests.

Suppose you have inferred computational operators

\[
T_1,\ldots,T_m
\]

from data—for example transition operators in a Hankel/spectral realization.

Define the **commutator operator**

\[
\mathcal C:
X\mapsto
\begin{pmatrix}
[X,T_1]\\
\vdots\\
[X,T_m]
\end{pmatrix}.
\]

Then

\[
\ker\mathcal C
=
\{T_1,\ldots,T_m\}'
\]

is the exact commutant.

But now just take the SVD of \(\mathcal C\).

If

\[
\mathcal C X_k
=
\sigma_k Y_k,
\]

then:

- \(\sigma_k=0\): exact symmetry;
- small \(\sigma_k\): approximate symmetry;
- large \(\sigma_k\): strongly broken direction.

So you get a **spectrum of symmetry** rather than a binary yes/no answer.

If a low-singular-value \(X\) is Hermitian, then

\[
U(t)=e^{-itX}
\]

generates an approximately commuting one-parameter transformation.

This is, in my view, a very natural spectral theory of approximate symmetries of computation.

---

## 8. An important subtlety for Hankel-type reconstructed computations

There is one geometric issue that matters a lot.

A minimal linear realization reconstructed spectrally is generally defined only up to similarity:

\[
T_a\mapsto ST_aS^{-1}.
\]

But the naked Frobenius norm

\[
\|[X,T_a]\|_F
\]

is **not invariant under general similarity transforms**.

So to make “amount of symmetry” an intrinsic property of the computation, you need a metric on state space.

Let \(G>0\) define

\[
\langle x,y\rangle_G=x^\dagger Gy.
\]

Under

\[
x\mapsto Sx,
\]

the metric transforms as

\[
G\mapsto
G'=S^{-\dagger}GS^{-1}.
\]

Then define the induced operator norm

\[
\boxed{
\|A\|_{\mathrm{HS},G}^2
=
\operatorname{Tr}
\left(
G^{-1}A^\dagger GA
\right).
}
\]

This satisfies

\[
\|SAS^{-1}\|_{\mathrm{HS},G'}
=
\|A\|_{\mathrm{HS},G}.
\]

So the properly geometric approximate-commutant objective is something like

\[
\mathcal E(X)
=
\sum_a
\|[X,T_a]\|_{\mathrm{HS},G}^2.
\]

This is where Gram matrices, controllability/observability Gramians, balanced coordinates, or an empirically determined state metric become conceptually important.

**The algebra tells you what counts as symmetry; the metric tells you how much a symmetry is broken.**

That distinction is worth keeping.

---

## 9. When you really do want an “approximate group”

There is a different question. Suppose you infer transformations \(U_g\), but they only approximately obey

\[
U_gU_h=U_{gh}.
\]

Define a multiplicative defect such as

\[
\eta
=
\sup_{g,h}
\|U_gU_h-U_{gh}\|.
\]

Then you have an **approximate representation**.

The natural question is:

\[
\eta\ll1
\quad\stackrel{?}{\Longrightarrow}\quad
U_g
\text{ is close to some exact representation}.
\]

This is an instance of **Ulam stability** or stability of approximate representations. The answer depends on the group, the norm, the class of representations, and sometimes topology; there is no universal theorem saying that every almost-representation lies near an exact one.

There is a closely related lesson from almost-commuting matrices. For two bounded Hermitian matrices, Lin's theorem says that sufficiently small commutator implies proximity to genuinely commuting Hermitian matrices, uniformly in matrix dimension. But analogous statements can fail for unitary operators because of topological/K-theoretic obstructions.

This is an important warning:

\[
\text{small violation of algebraic relations}
\]

does **not automatically imply**

\[
\text{small perturbation of an exact algebraic structure}.
\]

For empirical computation, I would therefore usually regard the measured defects themselves as primary rather than insisting that every approximate structure be rounded to an exact group.

---

## 10. Approximate consequences for a computation

Suppose

\[
T=T_0+E,
\qquad
T_0=\mathcal P_G(T),
\qquad
\|E\|\leq\epsilon.
\]

Then

\[
T^n-T_0^n
=
\sum_{k=0}^{n-1}
T^{n-1-k}E T_0^k.
\]

If \(T\) and \(T_0\) are contractions,

\[
\boxed{
\|T^n-T_0^n\|
\leq n\epsilon.
}
\]

So a computation that is \(\epsilon\)-close to a symmetric computation behaves approximately symmetrically over finite horizons, with a directly controlled accumulation of error.

Likewise, small commutators give approximate selection rules; for Hermitian/normal systems small symmetry-breaking perturbations give the usual perturbative splitting of symmetry-protected eigenspaces.

Thus you can build a whole hierarchy:

\[
\text{small commutator}
\Rightarrow
\text{close to invariant sector}
\Rightarrow
\text{approximate selection rules/conservation laws}
\Rightarrow
\text{controlled approximate computational behavior}.
\]

---

## 11. What I would actually do for computation

A practical program would be:

1. **Represent the computation by an operator algebra** \(\mathcal A=\operatorname{alg}(T_1,\ldots,T_m)\), perhaps obtained from a Hankel/spectral reconstruction or from learned feature dynamics.
2. **Choose an intrinsic state-space metric** \(G\), rather than relying on an arbitrary coordinate Frobenius norm.
3. **Construct the commutator superoperator**
   \[
   \mathcal C(X)=([X,T_1],\ldots,[X,T_m]).
   \]
4. **Spectrally decompose \(\mathcal C^\dagger\mathcal C\).** Its zero modes are exact commutants; low modes are approximate symmetries.
5. Where a candidate group is known, use **twirling** to obtain the nearest exactly symmetric computation and explicitly separate symmetric from symmetry-breaking components.
6. Use the spectral gap of the symmetry Laplacian to translate small symmetry defect into bounds on approximate selection rules or computational behavior.
7. Only when you specifically need a genuine hidden group should you ask whether your approximate transformations can be “rounded” to an exact representation; that is the harder stability problem.

My main takeaway would be that **commutants are indeed the right language**, but the additional ingredient you were missing is a *metric and spectral theory on the space of algebraic relations*.

Instead of asking

\[
\text{“Does this system possess symmetry }G\text{?”}
\]

you can ask

\[
\boxed{
\text{“How much of this computation lies in the commutant, and what is the spectrum of its departures from it?”}
}
\]

That gives something continuous, geometrical, data-compatible, and directly usable for comparing computations.
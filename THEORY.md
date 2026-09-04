# Mathematical Theory of the DE–Gradient Hybrid Optimizer

This document develops the theoretical foundations for `DEGradientOptimizer`
(`de_gradient_optimizer.py`), a hybrid algorithm combining Differential
Evolution (DE) — a derivative-free, population-based global search method —
with Gradient Descent (GD) — a local, derivative-based descent method.

## 1. Problem Setting

We consider the box-constrained minimization problem

  minimize   f(x)
  subject to x in Ω = [l_1,u_1] × ... × [l_d,u_d] ⊂ R^d,

where f: R^d → R is bounded below on Ω. We do not assume convexity globally;
f may be multimodal. Locally, however, we assume standard smoothness
conditions near stationary points and along descent trajectories.

**Assumption A1 (L-smoothness).** f is differentiable and its gradient ∇f is
Lipschitz continuous with constant L > 0:
  ‖∇f(x) − ∇f(y)‖ ≤ L‖x − y‖ for all x, y ∈ Ω.

**Assumption A2 (Boundedness).** f is bounded below on Ω, i.e. f* = inf_{x∈Ω} f(x) > −∞.

## 2. Gradient Descent Component: Descent Lemma and Convergence

The optimizer's local refinement step performs

  x_{k+1} = P_Ω( x_k − η ∇f(x_k) ),

where P_Ω denotes projection onto the box Ω (implemented via clipping) and η
is the learning rate `grad_lr`.

**Lemma 1 (Descent Lemma).** Under A1, for any x, y ∈ R^d,

  f(y) ≤ f(x) + ∇f(x)^T (y − x) + (L/2)‖y − x‖².

*Proof.* Standard consequence of the fundamental theorem of calculus applied
to g(t) = f(x + t(y−x)) combined with the Lipschitz bound on ∇f; see Nesterov,
*Introductory Lectures on Convex Optimization*, Lemma 1.2.3. ∎

**Theorem 1 (Sufficient Decrease, unconstrained case).** Suppose A1 holds and
0 < η ≤ 1/L. Let x_{k+1} = x_k − η∇f(x_k) (interior iterates, no projection
active). Then

  f(x_{k+1}) ≤ f(x_k) − (η/2)‖∇f(x_k)‖².

*Proof.* Apply Lemma 1 with y = x_{k+1} = x_k − η∇f(x_k):

  f(x_{k+1}) ≤ f(x_k) − η‖∇f(x_k)‖² + (Lη²/2)‖∇f(x_k)‖²
            = f(x_k) − η(1 − Lη/2)‖∇f(x_k)‖².

Since η ≤ 1/L, we have 1 − Lη/2 ≥ 1/2, giving the claimed bound. ∎

**Corollary 1 (Convergence to stationarity).** Under A1–A2 and 0 < η ≤ 1/L,
running K unconstrained GD steps yields

  min_{0≤k<K} ‖∇f(x_k)‖² ≤ 2(f(x_0) − f*) / (ηK),

so ‖∇f(x_k)‖ → 0 as K → ∞; i.e., every accumulation point of the gradient
sub-sequence is a stationary point.

*Proof.* Sum the inequality of Theorem 1 telescopically from k=0 to K−1:

  (η/2) Σ_{k=0}^{K-1} ‖∇f(x_k)‖² ≤ f(x_0) − f(x_K) ≤ f(x_0) − f*.

Divide by ηK/2 and bound the average by the minimum term. ∎

**Remark (projected version).** When the box projection P_Ω is active,
Theorem 1 and Corollary 1 hold with ‖∇f(x_k)‖² replaced by the squared norm
of the projected gradient mapping G_η(x_k) = (x_k − P_Ω(x_k − η∇f(x_k)))/η,
a standard result for projected gradient descent on convex constraint sets
(Ω is a box, hence convex). This justifies using clipped GD steps inside the
optimizer without losing the O(1/K) stationarity guarantee.

## 3. Differential Evolution Component: Population Diversity and Global Exploration

DE maintains a population {x_1,...,x_NP} ⊂ Ω and generates trial vectors via
mutation

  v_i = x_i + F(x_best − x_i) + F(x_{r1} − x_{r2}),  r1 ≠ r2 ≠ i,

followed by binomial crossover with rate CR and greedy (elitist) selection:
the trial replaces x_i only if it does not increase fitness.

**Proposition 1 (Monotone non-degradation).** Let f_k = min_i f(x_i^{(k)}) be
the best fitness in generation k under DE's greedy selection rule (used
identically for the gradient-refinement step). Then the sequence (f_k) is
non-increasing: f_{k+1} ≤ f_k for all k.

*Proof.* Both the DE selection rule (trial replaces target iff
f(trial) ≤ f(target)) and the gradient-refinement replacement rule in
`DEGradientOptimizer._gradient_refine` combined with its acceptance check are
elitist: no accepted replacement can increase any individual's fitness, hence
the population minimum cannot increase. Since the best individual `best` is
only updated when a strictly smaller fitness is found, f_{k+1} ≤ f_k. ∎

This gives an immediate consequence: the optimizer's reported best fitness is
a monotone non-increasing function of the number of generations, which is
useful both as a sanity check and as the basis for the early-stopping
criterion on ‖f_{k+1} − f_k‖ < tol.

**Proposition 2 (Population diversity lower bound under mutation).** Assume
F > 0 and that at generation k the population is not fully converged, i.e.
there exist i with x_{r1} ≠ x_{r2} in the mutation base pair. Then the
expected pairwise diversity of the *mutant* population,

  D_k = E[ ‖v_i − v_j‖ ],   i ≠ j,

satisfies D_k ≥ F · E[‖x_{r1} − x_{r2}‖] − O(F · ‖x_best − x_i‖ variability),

i.e. the injected difference vector F(x_{r1}−x_{r2}) provides a diversity
floor proportional to F and the current spread of the population, preventing
premature collapse to a single point as long as F is bounded away from 0 and
the population has not fully converged (x_{r1} ≠ x_{r2} for some sampled
pairs). This is the standard qualitative diversity argument underlying DE's
global exploration ability (Storn & Price, 1997; Price, Storn & Lampinen,
*Differential Evolution*, 2005, Ch. 2).

## 4. Combined Algorithm: A Two-Phase Convergence Argument

The hybrid optimizer alternates:
  (i) a *global* DE phase applied to the whole population (Proposition 1),
  (ii) a *local* elitist GD refinement phase applied to the best
      `grad_fraction · NP` individuals (Theorem 1 / Corollary 1).

**Theorem 2 (Hybrid convergence).** Suppose A1–A2 hold in a neighborhood of
the eventual best individual, and 0 < η ≤ 1/L for the refinement step. Then:

 (a) The best-fitness sequence (f_k) is non-increasing and bounded below by
     f*, hence converges to some f_∞ ≥ f*  (monotone bounded sequence
     convergence, using Proposition 1 and A2).

 (b) If, from some generation k_0 onward, the best individual's trajectory
     under repeated gradient refinement stays in a region where A1 holds
     (e.g., a basin of attraction not altered further by DE because the
     gradient step already dominates), then by Corollary 1 the projected
     gradient norm along that sub-sequence of refinement steps tends to
     zero, so f_∞ is attained at a stationary point of f restricted to Ω.

*Proof.* (a) is immediate from Proposition 1 and the lower bound f* (A2):
every non-increasing sequence bounded below converges. (b) follows by
applying Corollary 1 to the sub-sequence of gradient-refinement iterates
applied to the best individual once DE no longer produces a strictly better
trial for it (which must eventually stabilize in a basin because (f_k) is
convergent by (a), so the increments f_k − f_{k+1} → 0). ∎

**Interpretation.** Part (a) is a *global* guarantee: the algorithm never
gets worse and always converges numerically. Part (b) is a *local*
guarantee inherited from gradient descent: once the population has
localized around a basin, the elitist gradient-refinement phase drives the
incumbent solution toward a KKT point of the box-constrained problem at the
classical O(1/K) rate. Together, these explain why the hybrid can match or
exceed pure DE's global reach while converging faster locally than DE alone,
and can escape poor local minima that trap pure gradient descent (via the
DE mutation/crossover phase acting on the rest of the population).

## 5. Complexity

Per generation, DE performs NP function evaluations (one trial per
individual). The gradient-refinement phase performs
`grad_fraction · NP · grad_steps` additional function evaluations (or
gradient evaluations, ×(2d) each if finite differences are used, since a
central difference requires 2 evaluations per dimension). Total per-generation
cost:

  NP + grad_fraction·NP·grad_steps·(1 + 2d·[no analytic gradient]).

This matches the intuition that supplying an analytic gradient (`grad`
argument) removes the O(d) finite-difference overhead per refinement step.

## 6. References

- R. Storn, K. Price. "Differential Evolution – A Simple and Efficient
  Heuristic for Global Optimization over Continuous Spaces." *Journal of
  Global Optimization*, 11(4), 1997.
- K. Price, R. Storn, J. Lampinen. *Differential Evolution: A Practical
  Approach to Global Optimization*. Springer, 2005.
- Y. Nesterov. *Introductory Lectures on Convex Optimization*. Springer, 2004.
- The repository's included reference paper `2211.17157v2.pdf` discusses a
  related swarm-based gradient optimizer design that motivated this hybrid.

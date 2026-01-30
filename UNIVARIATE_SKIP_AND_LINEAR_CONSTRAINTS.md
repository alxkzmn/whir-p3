# Univariate Skip vs. Linear Constraints

This note explains the relationship between the univariate-skip initial phase and explicit
linear functionals (such as "next-row" constraints). It also outlines what is required to make
skip-mode efficient in the presence of linear constraints.

## Background

### Relation to the univariate-skip idea

The univariate-skip technique replaces the first `k` Boolean variables with a multiplicative
subgroup domain `D` of size $|D| = 2^k$, effectively working over a product domain
$D \times H^{n-k}$ in the initial phase. This lets the prover "skip" `k` rounds by sending a
single univariate polynomial of degree roughly $d(|D|-1)$, which the verifier interpolates.

In practice, the initial phase can run in multiple modes:

- **With statement**: standard sumcheck against a statement.
- **With statement + skip**: univariate-skip optimization, folding `K` variables at once.
- **Without statement**: direct folding (no initial statement).

## Linear constraints vs. point constraints

Point constraints are of the form:

- **Point evaluation**: $p(z) = s$

Linear constraints are of the form:

- **Linear functional**: $\langle w, p(\cdot) \rangle = s$

The second kind cannot be represented as a single $\mathrm{eq}(z, X)$ weight polynomial.

## What univariate skip assumes

The univariate-skip optimization is designed around constraints that can be viewed as
pointwise evaluations over a table/domain. In that setting, the constraint function is
evaluated at many domain points (rows), and that structure is used to batch and skip.

## Why global linear functionals complicate skip-mode

"Next-row" constraints are naturally encoded as **global linear functionals** over the whole
evaluation table (a dot-product against a weight vector). They are not "evaluate-at-a-row"
constraints, so they do not automatically inherit the $D \times H^{n-k}$ structure that the
skip phase expects.

If univariate-skip is enabled while linear constraints are present, the verifier must evaluate
those constraints using the same skip-aware mapping as the prover to avoid mismatches.

## Current status

Skip-mode can be used with explicit linear constraints, provided the verifier evaluates those
constraints using the skip-aware mapping. A direct implementation evaluates linear constraint
weights over the skip domain (often via dense expansion for tensor-product constraints in the
skip round).

## What would be required to make this efficient

Dense expansion in the skip round is correct but can be expensive for large instances.
To improve efficiency:

- Add a skip-aware evaluator for tensor-product constraints that avoids densification.
- Evaluate row/column weights using skip folding directly, rather than building full $2^n$
  tables.

## Notes on correctness requirements

When explicit linear constraints are present, the verifier must include them when evaluating
the combined constraint polynomial. This means:

- linear constraints consume the continuation of the batching challenge powers, and
- the ordering of point vs. linear constraints must match between prover and verifier.

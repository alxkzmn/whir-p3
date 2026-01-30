# Design: Explicit Linear Constraints for Multilinear PCS Statements

This document describes how to support **explicit linear functional constraints** alongside
existing **point evaluation** constraints in a multilinear polynomial commitment setting.
The goal is to enable constraints that are naturally expressed as global linear functionals
over the full evaluation table (e.g., "next-row" openings) without requiring workarounds like
committing shifted/rotated columns.

## Problem statement

Historically, the statement layer could express constraints of the form:

- **Point evaluation**: $p(z) = s$

where batching is done via the multilinear equality polynomial $\mathrm{eq}(z, X)$ and
successive powers of a challenge $\gamma$.

However, some query types are not a single point evaluation. They are best viewed as a
**linear functional** over the entire evaluation table:

- **Linear functional**: $\langle w, p(\cdot) \rangle = s$

where $w$ is a full weight vector over the Boolean hypercube $\{0,1\}^n$. This cannot be
represented as a single $\mathrm{eq}(z, X)$ weight polynomial.

## High-level approach

Extend the statement layer so that the protocol can batch and verify a mixture of:

1. Point constraints (existing): weights are derived as $\mathrm{eq}(z_i, X)$
2. Explicit linear constraints (new): weights are provided directly as a vector over
   $\{0,1\}^n$
3. **Structured linear constraints (tensor-product)**: a compact representation of a linear
   functional that is a Kronecker product over a contiguous subrange of the concatenated
   polynomial

Both kinds participate in the same "combine weights + combine claimed sums" flow, using the
same batching challenge powers.

## Statement data model

The statement stores, in addition to point constraints, a list of linear constraints.
Each linear constraint corresponds to $\langle w_j, p(\cdot) \rangle = s_j$, where:

- $w_j$ is either:
  - a dense weight vector over $\{0,1\}^{n}$, or
  - a **tensor-product** weight vector over a contiguous range in the concatenated polynomial.
- $s_j$ is the expected value.

The statement should provide methods to:

- add a dense linear constraint,
- add a tensor-product constraint, and
- check whether any linear constraints are present.

### Ordering / batching invariant

Constraint weights are conceptually ordered as:

1. All point constraints in insertion order
2. All linear constraints in insertion order

This ordering matters because batching uses successive powers of the same challenge $\gamma$.
Both prover and verifier must consume powers in the same order to avoid mismatches.

## Verifier-side evaluation

The verifier evaluates the combined constraint polynomial $W(r)$ at verifier-chosen points.
With linear constraints introduced, the evaluator must include them as well.

The evaluator should:

- compute the contribution from point constraints as before, and
- compute the contribution from linear constraints by evaluating each explicit weight vector at
  the same evaluation point and weighting it by the **continuation** of the batching powers.

If there are `point_eq_count` point constraints, then the first linear constraint uses
$\gamma^{\text{point\_eq\_count}}$, the next uses $\gamma^{\text{point\_eq\_count}+1}$, etc.

## Interaction with univariate skip

Linear constraints can be evaluated in skip mode by using the same skip-aware mapping as the
prover. A direct implementation may expand tensor-product constraints to dense vectors in the
skip round, which is correct but can be expensive. See
`UNIVARIATE_SKIP_AND_LINEAR_CONSTRAINTS.md` for a deeper discussion.

## Tensor-product constraints for "next-row" queries

Instead of materializing a dense $2^n$ vector, "next-row" queries can be encoded as a
tensor-product constraint.

Let the concatenated polynomial range for a matrix be partitioned into rows of length
$2^{\log w}$ (where $\log w$ is the matrix log-width). For "next-row" queries, the weights
factor as:

$$
w(\text{row}, \text{col}) = w_\text{row}(\text{row}) \cdot w_\text{col}(\text{col})
$$

where:

- $w_\text{col}$ is the equality polynomial over the column variables, and
- $w_\text{row}$ is the rotated equality polynomial over the row variables.

We store these two vectors separately and apply them only over the matrix’s contiguous range
inside the concatenated polynomial. This preserves the same linear functional while avoiding
construction of a full-length dense vector.

### Why this matches the intended semantics

The dense representation produces a vector where each row chunk has the same column weights,
scaled by the row’s rotated weight. The tensor-product representation encodes exactly that
Kronecker structure, and the verifier evaluates it by:

$$
\langle w, p \rangle = \sum_{\text{row}} w_\text{row}(\text{row}) \cdot
\sum_{\text{col}} w_\text{col}(\text{col})\, p(\text{row},\text{col})
$$

The concatenated polynomial’s **high bits** (selecting the range) are fixed by a range start,
and the verifier multiplies by the corresponding equality factor for those bits.

## Comparison with dense vectors

**Dense representation:**

- Build a length-$2^{\log b}$ weight vector for every "next-row" query.
- Cost: $O(2^{\log b})$ memory traffic and time per query.
- Verifier must evaluate a full multilinear polynomial defined by the dense vector.

**Tensor-product representation:**

- Store only `row_weights` and `col_weights` plus range metadata.
- Cost: $O(2^{\log w} + 2^{\log h})$ storage (row/col) and no dense materialization.
- Verifier evaluates a product of two smaller multilinear polynomials and a fixed-range factor.

This preserves correctness while drastically reducing verifier work for "next-row" queries.

## Testing

Regression coverage should include:

- Point-only statements still combine/verify as before.
- Linear constraints round-trip through statement combination logic.
- Packed vs. unpacked flows remain consistent.

## Future work

Potential follow-ups:

- Reduce the cost of building/storing full weight vectors for some linear functionals
  (e.g., more structured representations).
- Extend the skip initial phase to incorporate linear constraints more efficiently, restoring
  the optimization for proofs that include "next-row" constraints.

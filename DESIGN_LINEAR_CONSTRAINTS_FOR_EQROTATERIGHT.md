# Design: Explicit Linear Constraints (for `EqRotateRight`-style queries)

This document describes the changes made in `whir-p3` to support **explicit linear functional constraints** in addition to the existing **point evaluation** constraints.

The immediate driver in this workspace is enabling HyperPlonk’s “next-row” openings (expressed upstream as `p3-ml-pcs::MlQuery::EqRotateRight`) **without** a workaround like committing shifted/rotated columns.

## Problem statement

Historically, `whir-p3`’s `EqStatement` could express constraints of the form:

- **Point evaluation**: \(p(z) = s\)

where batching is done via the multilinear equality polynomial \(\mathrm{eq}(z, X)\) and powers of a challenge \(\gamma\).

However, some query types (notably `EqRotateRight`) are not a single point evaluation. They are best viewed as a **linear functional** over the entire evaluation table:

- **Linear functional**: \(\langle w, p(\cdot) \rangle = s\)

where \(w\) is a full weight vector over the Boolean hypercube \(\{0,1\}^n\).

This cannot be represented as a single \(\mathrm{eq}(z, X)\) weight polynomial.

## High-level approach

We extend the statement layer so that the protocol can batch and verify a mixture of:

1. Point constraints (existing): weights are derived as \(\mathrm{eq}(z_i, X)\)
2. Explicit linear constraints (new): weights are provided directly as a vector over \(\{0,1\}^n\)
3. **Structured linear constraints (tensor-product)**: a compact representation of a linear functional
  that is a Kronecker product over a contiguous subrange of the concatenated polynomial

Both kinds participate in the same “combine weights + combine claimed sums” flow, using the same batching challenge powers.

## Changes in `EqStatement`

File: `src/whir/constraints/statement/eq.rs`

### New fields

`EqStatement<F>` now stores, in addition to `points`/`evaluations`:

- `linear_weights: Vec<LinearConstraint<F>>`
- `linear_evaluations: Vec<F>`

Each entry corresponds to a constraint of the form \(\langle w_j, p(\cdot) \rangle = s_j\) where:

- `w_j` is either:
  - a dense weight vector over \(\{0,1\}^{num\_variables}\), or
  - a **tensor-product** weight vector over a contiguous range in the concatenated polynomial.
- `s_j` is the expected value

### New API

- `add_linear_constraint(weights: EvaluationsList<F>, eval: F)`
- `add_tensor_product_constraint(range_start, log_range_len, row_weights, col_weights, eval)`
- `has_linear_constraints() -> bool`

### Ordering / batching invariant

Constraint weights are conceptually ordered as:

1. All point constraints in insertion order
2. All linear constraints in insertion order

This ordering matters because batching uses successive powers of the same challenge \(\gamma\). Any code that consumes `challenge.powers()` must match the same order on both prover and verifier.

## Verifier-side evaluation fix

File: `src/whir/constraints/evaluator.rs`

`ConstraintPolyEvaluator` evaluates the combined constraint polynomial \(W(r)\) at verifier-chosen points.

With linear constraints introduced, the evaluator must include them as well.

The implementation now:

- Computes the contribution from point constraints as before.
- Computes the contribution from linear constraints by evaluating each explicit weight vector at the same evaluation point and weighting it by the **continuation** of the batching powers:

  - If there are `point_eq_count` point constraints, then the first linear constraint uses \(\gamma^{point\_eq\_count}\), the next uses \(\gamma^{point\_eq\_count+1}\), etc.

This closes the soundness gap that would otherwise appear as an immediate sumcheck mismatch (e.g. failing at round 0).

## Interaction with univariate skip / SVO

There is an important limitation: the current univariate-skip optimized initial phase assumes the statement can be expressed in the “pointwise” style.

- Explicit linear functionals are currently **not** supported in the skip path.

We enforce this in two ways:

1. The adapter layer (`p3-whir`, outside this crate) forces classic `WithStatement` if it detects any linear constraints.
2. The constraint system validation for the skip case rejects linear constraints.

See `UNIVARIATE_SKIP_AND_LINEAR_CONSTRAINTS.md` for a longer discussion and a mapping to the linked univariate-skip paper.

## Tensor-product constraints for `EqRotateRight`

The adapter now uses a tensor-product constraint for `MlQuery::EqRotateRight` instead of
materializing a dense \(2^n\) vector.

Let the concatenated polynomial range for a matrix be partitioned into rows of length
\(2^{\log w}\) (where \(\log w\) is the matrix log-width). For `EqRotateRight`, the weights factor as:

$$
w(\text{row}, \text{col}) = w_\text{row}(\text{row}) \cdot w_\text{col}(\text{col})
$$

where:

- \(w_\text{col}\) is the equality polynomial over the column variables (`eq_r`), and
- \(w_\text{row}\) is the rotated equality polynomial from `EqRotateRight`.

We store these two vectors separately and apply them only over the matrix’s contiguous range
inside the concatenated polynomial. This preserves the same linear functional while avoiding
construction of a full-length dense vector.

### Why this matches the old semantics

The old approach produced a dense vector `weight` where each row chunk had the same column
weights (`eq_r`) scaled by the row’s rotated weight. The tensor-product representation encodes
exactly that Kronecker structure, and the verifier evaluates it by:

$$
\langle w, p \rangle = \sum_{\text{row}} w_\text{row}(\text{row}) \cdot \sum_{\text{col}} w_\text{col}(\text{col})\, p(\text{row},\text{col})
$$

The concatenated polynomial’s **high bits** (selecting the range) are fixed by
`range_start`, and the verifier multiplies by the corresponding equality factor for those bits.

## Comparison with the old dense-vector approach

**Old (dense):**

- Build a length-\(2^{\log b}\) `weight` vector for every `EqRotateRight` query.
- Cost: \(O(2^{\log b})\) memory traffic and time per query.
- Verifier must evaluate a full multilinear polynomial defined by the dense vector.

**New (tensor-product):**

- Store only `row_weights` and `col_weights` plus range metadata.
- Cost: \(O(2^{\log w} + 2^{\log h})\) storage (row/col) and no dense materialization.
- Verifier evaluates a product of two smaller multilinear polynomials and a fixed-range factor.

This preserves correctness while drastically reducing verifier work for `EqRotateRight`.

## Integration note (outside `whir-p3`)

The `p3-whir` adapter translates `MlQuery::EqRotateRight` into a tensor-product constraint and
passes it into `EqStatement::add_tensor_product_constraint`.

This is intentionally kept in the adapter:

- `whir-p3` remains generic: it supports explicit linear constraints without knowing the upstream query semantics.
- The adapter owns the policy of how to express a given `MlQuery`.

## Testing

The `whir-p3` test suite contains regression coverage for the new behavior, including checks that:

- Point-only statements still combine/verify as before.
- Linear constraints round-trip through statement combine logic.
- Packed vs unpacked flows remain consistent.

(See the `whir::constraints::statement::eq` tests in `src/whir/constraints/statement/eq.rs`.)

## Future work

Potential follow-ups (intentionally not implemented here):

- Reduce the cost of building/storing full weight vectors for some linear functionals (e.g. more structured representations).
- Extend the skip/SVO initial phase to incorporate linear constraints correctly, restoring the optimization for proofs that include `EqRotateRight`.

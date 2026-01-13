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

Both kinds participate in the same “combine weights + combine claimed sums” flow, using the same batching challenge powers.

## Changes in `EqStatement`

File: `src/whir/constraints/statement/eq.rs`

### New fields

`EqStatement<F>` now stores, in addition to `points`/`evaluations`:

- `linear_weights: Vec<EvaluationsList<F>>`
- `linear_evaluations: Vec<F>`

Each entry corresponds to a constraint of the form \(\langle w_j, p(\cdot) \rangle = s_j\) where:

- `w_j` is the full weight vector over \(\{0,1\}^{num_variables}\)
- `s_j` is the expected value

### New API

- `add_linear_constraint(weights: EvaluationsList<F>, eval: F)`
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

## Integration note (outside `whir-p3`)

The `p3-whir` adapter translates `MlQuery::EqRotateRight` into an explicit weight vector over the concatenated polynomial’s hypercube and passes it into `EqStatement::add_linear_constraint`.

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

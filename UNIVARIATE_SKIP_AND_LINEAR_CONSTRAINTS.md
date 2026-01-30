# Univariate Skip vs. Linear Constraints (e.g. `EqRotateRight`)

This note documents the relationship between the univariate-skip initial phase and explicit
linear functionals (such as `p3-ml-pcs::MlQuery::EqRotateRight`).

It also clarifies what this means for HyperPlonk and outlines what would be required to make
skip-mode work with linear constraints.

## Background

### Relation to the linked paper

The “univariate skip” idea used here corresponds closely to Section **5.1 (The Univariate Skip)**
of [_“Some Improvements for the PIOP for ZeroCheck”_](https://eprint.iacr.org/2024/108).

In the paper’s terminology, the prover replaces the first `k` Boolean variables with a multiplicative
subgroup domain `D` of size $|D| = 2^k$, effectively working over a product domain
$D \times H^{n-k}$ for the first phase. This lets the prover “skip” `k` rounds by sending a single
univariate polynomial of degree roughly $d(|D|-1)$ and having the verifier interpolate it.

In `whir-p3`, this is reflected as an _initial-phase variant_ (the `WithStatementSkip` branch)
that folds `K_SKIP_SUMCHECK` variables at once.

WHIR can run its _initial phase_ in multiple modes (see `InitialPhase` in
`src/whir/proof.rs`):

- `WithStatement { .. }`: classic/standard sumcheck with a statement.
- `WithStatementSkip(..)`: a “univariate skip” optimization, which folds `K_SKIP_SUMCHECK` variables at once.
- `WithoutStatement { .. }`: direct folding (no initial statement).

HyperPlonk (in this workspace) needs both “local row” and “next row” openings.
In the `MlPcs` API, “next row” is represented by `MlQuery::EqRotateRight`, which is **not** a point
query; it is a **linear functional** over the full evaluation table.

Because of that, WHIR’s PCS adapter encodes `EqRotateRight` as an _explicit weight vector_
constraint:

- Point evaluation: `p(z) = s`
- Linear functional: `⟨w, p(·)⟩ = s`

The second kind cannot be represented as a single `eq(z, X)` weight polynomial.

## What “univariate skip” currently assumes

The univariate-skip optimization is implemented as a special initial-phase verifier/prover path.
In practice (today), the skip path assumes the initial statement is of the “point evaluation” form
(or at least a form that can be reduced in the same way).

Importantly (and matching the paper’s setting), the skip optimization is designed around
**pointwise constraints over rows/points of a table/domain**: you evaluate a constraint function
at many domain points (rows) and use that structure to batch/skip.

## Why `EqRotateRight` breaks skip-mode today

`EqRotateRight` is encoded as an explicit `EqStatement` linear constraint:

- `EqStatement::add_linear_constraint(weights, eval)`

These linear constraints do work in **standard** `WithStatement` mode because they are batched into
sumcheck as part of the normal “statement → combined weight polynomial” flow.

However, the skip-mode path must incorporate generic `linear_weights` into its specialized skip
transcript and checks.

This matches the paper’s intuition: the skip trick relies on viewing the constraint as a function
evaluated _per row/point_ over an enlarged domain $D \times H^{n-k}$. In contrast,
`EqRotateRight` is a **global linear functional** over the whole evaluation table
(a dot-product against a weight vector). It is not “evaluate-at-a-row” semantics, so it does not
automatically inherit the $D \times H^{n-k}$ structure the skip phase expects.

If univariate-skip is enabled while linear constraints are present, the verifier must evaluate
those constraints using the same skip-aware mapping as the prover to avoid mismatches.


## Current status

Skip-mode can be used with explicit linear constraints, provided the
verifier evaluates those constraints using the skip-aware mapping. This is implemented by
evaluating linear constraint weights over the skip domain (currently via a dense expansion for
tensor-product constraints in the skip round).

## What would be required to make this efficient

The current implementation evaluates linear constraints in the skip round by expanding tensor-
product constraints to a dense vector. This is correct but can be expensive for large instances.
To improve efficiency:

- Add a skip-aware evaluator for tensor-product constraints that avoids densification.
- Evaluate row/column weights using skip folding directly, rather than building full $2^n$ tables.

## Notes on current correctness fixes

When adding explicit linear constraints, the verifier must evaluate the constraint polynomial
including them.

In this workspace we fixed that by updating `ConstraintPolyEvaluator` to include the
contribution of `EqStatement::linear_weights` (weighted by the same batching challenge powers).
See `src/whir/constraints/evaluator.rs`.

## Status

- Standard (`WithStatement`) + explicit linear constraints: supported.
- Univariate skip (`WithStatementSkip`) + explicit linear constraints: supported, with the caveat
  that tensor-product constraints are currently evaluated via dense expansion in the skip round.

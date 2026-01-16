# Univariate Skip vs. Linear Constraints (e.g. `EqRotateRight`)

This note documents a current limitation in the WHIR integration used by this workspace:
**the univariate-skip initial phase is not currently compatible with explicit linear functionals**
(such as `p3-ml-pcs::MlQuery::EqRotateRight`).

It also clarifies what this means for HyperPlonk and outlines what would be required to make
skip-mode work with linear constraints.

## Background

### Relation to the linked paper

The “univariate skip” idea used here corresponds closely to Section **5.1 (The Univariate Skip)**
of _“Some Improvements for the PIOP for ZeroCheck”_ (Angus Gruen, Polygon Zero)
(see the file `univariate skip paper.md` in this repository).

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

In this workspace we enforce this assumption by validating that _skip-mode_ does not contain
unsupported constraint kinds:

- `Constraint::validate_for_skip_case()` rejects select constraints and (now) rejects explicit linear constraints.
  See `src/whir/constraints/mod.rs`.

Importantly (and matching the paper’s setting), the skip optimization is designed around
**pointwise constraints over rows/points of a table/domain**: you evaluate a constraint function
at many domain points (rows) and use that structure to batch/skip.

## Why `EqRotateRight` breaks skip-mode today

`EqRotateRight` is encoded as an explicit `EqStatement` linear constraint:

- `EqStatement::add_linear_constraint(weights, eval)`

These linear constraints do work in **standard** `WithStatement` mode because they are batched into
sumcheck as part of the normal “statement → combined weight polynomial” flow.

However, the current **skip-mode** initial phase does not incorporate generic `linear_weights` into
its specialized skip transcript and checks.

This matches the paper’s intuition: the skip trick relies on viewing the constraint as a function
evaluated _per row/point_ over an enlarged domain $D \times H^{n-k}$. In contrast,
`EqRotateRight` is a **global linear functional** over the whole evaluation table
(a dot-product against a weight vector). It is not “evaluate-at-a-row” semantics, so it does not
automatically inherit the $D \times H^{n-k}$ structure the skip phase expects.

If we allowed univariate-skip while also allowing linear constraints, we would risk a mismatch
between what the prover proves and what the verifier checks (i.e. unsoundness) unless the skip
path is extended accordingly.

This is why the PCS adapter in `whir/src/pcs.rs` currently forces
`InitialPhase::WithStatement { .. }` when building proofs that may include `EqRotateRight`.

## Does this mean HyperPlonk will never be compatible with univariate skip?

No.

It means:

- **As implemented today**, HyperPlonk + WHIR (with `EqRotateRight` implemented via explicit
  linear functionals) uses standard `WithStatement` for correctness.

HyperPlonk could still use univariate-skip under either of these conditions:

1. **HyperPlonk stops needing `EqRotateRight`** (e.g. it is rewritten to only issue point queries), or
2. **WHIR’s univariate-skip initial phase is extended to support explicit linear constraints**.

So this is a _current implementation limitation_, not a fundamental impossibility.

## What would be required to support skip-mode with linear constraints

High-level requirements:

- Extend the skip-mode initial phase so that it correctly accounts for `EqStatement::linear_weights`
  (and their expected evaluations) in the claimed sum and in the prover/verifier transcript.

Concretely, you would need to:

- Update the skip-mode sumcheck construction so that the combined “constraint weight polynomial”
  includes both:
  - point-eval weights `eq(z_i, ·)` and
  - explicit weights `w_j(·)`.
- Ensure prover and verifier derive the _same_ combined polynomial and the same claimed sums under
  the skip folding step.
- Define how linear constraints interact with the “skip interpolation” step (the part that samples
  `r_skip` and interpolates over a coset/subgroup).

Conceptually, you need a way to express the _linear functional constraints_ in a form that the
skip phase’s univariate polynomial (the analogue of the paper’s $v_0(X)$) can commit to and that
the verifier can check consistently.

Pragmatically, it’s likely easiest to:

- first implement correctness for the skip-mode path in the same style as the classic path, and
- only then recover the performance wins by optimizing.

## Notes on current correctness fixes

When adding explicit linear constraints, the verifier must evaluate the constraint polynomial
including them.

In this workspace we fixed that by updating `ConstraintPolyEvaluator` to include the
contribution of `EqStatement::linear_weights` (weighted by the same batching challenge powers).
See `src/whir/constraints/evaluator.rs`.

## Status

- Standard (`WithStatement`) + explicit linear constraints: supported.
- Univariate skip (`WithStatementSkip`) + explicit linear constraints: not supported (by design) until
  the skip-mode logic is extended.

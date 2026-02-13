use alloc::{vec, vec::Vec};

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_interpolation::interpolate_subgroup;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::{
    parameters::FoldingFactor,
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::constraints::{Constraint, statement::eq::LinearConstraint},
};

fn eval_evals_with_skip<EF>(
    evals: &EvaluationsList<EF>,
    r_all: &MultilinearPoint<EF>,
    k_skip: usize,
) -> EF
where
    EF: TwoAdicField,
{
    let n = r_all.num_variables() + k_skip - 1;
    assert_eq!(
        evals.num_variables(),
        n,
        "Evaluation list must match full domain size for skip evaluation"
    );

    let num_remaining_vars = n - k_skip;
    let width = 1 << num_remaining_vars;
    let mat = RowMajorMatrix::new(evals.as_slice().to_vec(), width);

    let r_skip = *r_all
        .last_variable()
        .expect("skip challenge must be present");
    let r_rest = MultilinearPoint::new(r_all.as_slice()[..num_remaining_vars].to_vec());

    let folded_row = interpolate_subgroup(&mat, r_skip);
    EvaluationsList::new(folded_row).evaluate_hypercube_base(&r_rest)
}

fn eq_d<EF: TwoAdicField>(x: EF, y: EF, subgroup_size: usize) -> EF {
    let order = EF::from_usize(subgroup_size);
    let order_inv = order
        .try_inverse()
        .expect("subgroup size must be invertible in the field");
    let y_inv = y.inverse();
    let base = x * y_inv;

    let mut total_sum = EF::ZERO;
    let mut power = EF::ONE;
    for _ in 0..subgroup_size {
        total_sum += power;
        power *= base;
    }

    total_sum * order_inv
}

fn lagrange_weights_for_skip<EF: TwoAdicField>(k_skip: usize, r_skip: EF) -> Vec<EF> {
    let subgroup_size = 1usize << k_skip;
    let subgroup_gen = EF::two_adic_generator(k_skip);

    let mut weights = vec![EF::ZERO; subgroup_size];
    let mut x = EF::ONE;
    for i in 0..subgroup_size {
        let bit_rev_i = i.reverse_bits() >> (usize::BITS - k_skip as u32);
        weights[bit_rev_i] = eq_d(x, r_skip, subgroup_size);
        x *= subgroup_gen;
    }

    weights
}

fn eval_tensor_product_with_skip<EF: TwoAdicField + Field>(
    range_start: usize,
    log_range_len: usize,
    row_weights: &EvaluationsList<EF>,
    col_weights: &EvaluationsList<EF>,
    eval_point: &MultilinearPoint<EF>,
    k_skip: usize,
    num_variables: usize,
) -> EF {
    let range_len = 1usize << log_range_len;
    let row_len = col_weights.num_evals();
    let log_row_len = log2_strict_usize(row_len);
    let rows = row_weights.num_evals();
    debug_assert_eq!(row_len * rows, range_len);

    let num_remaining_vars = num_variables - k_skip;
    let width = 1usize << num_remaining_vars;

    let r_skip = *eval_point
        .last_variable()
        .expect("skip challenge must be present");
    let r_rest = MultilinearPoint::new(eval_point.as_slice()[..num_remaining_vars].to_vec());

    let row_lagrange = lagrange_weights_for_skip(k_skip, r_skip);
    let col_eq = EvaluationsList::new_from_point(r_rest.as_slice(), EF::ONE);

    let mut total = EF::ZERO;
    let range_end = range_start + range_len;
    for idx in range_start..range_end {
        let row = idx / width;
        let col = idx % width;
        let local = idx - range_start;
        let row_local = local >> log_row_len;
        let col_local = local & (row_len - 1);
        let weight = row_weights.0[row_local] * col_weights.0[col_local];
        total += weight * row_lagrange[row] * col_eq.0[col];
    }

    total
}

fn eval_eq_with_skip<EF: TwoAdicField>(
    constraint_point: &MultilinearPoint<EF>,
    eval_point: &MultilinearPoint<EF>,
    k_skip: usize,
) -> EF {
    assert_eq!(
        constraint_point.num_variables(),
        eval_point.num_variables(),
        "Skip-eq points must have matching dimensions"
    );
    assert!(
        constraint_point.num_variables() >= 1,
        "Skip-eq points must contain at least one coordinate"
    );

    // Constraint points are [z_skip, z_rest...], while eval points are [r_rest..., r_skip].
    let z_skip = constraint_point[0];
    let z_rest = MultilinearPoint::new(constraint_point.as_slice()[1..].to_vec());
    let r_skip = *eval_point
        .last_variable()
        .expect("skip challenge must be present");
    let r_rest = MultilinearPoint::new(eval_point.as_slice()[..eval_point.num_variables() - 1].to_vec());

    let subgroup_eq = eq_d(r_skip, z_skip, 1usize << k_skip);
    let hypercube_eq = z_rest.eq_poly(&r_rest);
    subgroup_eq * hypercube_eq
}

/// Evaluate a single round's constraint.
fn eval_round<F: Field, EF: ExtensionField<F> + TwoAdicField>(
    round: usize,
    constraint: &Constraint<F, EF>,
    original_point: &MultilinearPoint<EF>,
    context: &PointContext<EF>,
) -> EF {
    let (eval_point, use_skip_eval, k_skip) = match (round, context) {
        (
            0,
            PointContext::Skip {
                rotated, k_skip, ..
            },
        ) => (rotated.clone(), true, *k_skip),
        (0, PointContext::NonSkip) => (original_point.reversed(), false, 0),
        (
            i,
            PointContext::Skip {
                rotated,
                k_skip,
                prover_challenge_offsets,
            },
        ) => {
            let start = if i == 1 {
                0
            } else {
                prover_challenge_offsets[i - 1]
            };
            let challenges = rotated.get_subpoint_over_range(0..rotated.num_variables() - 1);
            (
                challenges.get_subpoint_over_range(start..challenges.num_variables()),
                false,
                *k_skip,
            )
        }
        (_, PointContext::NonSkip) => {
            let slice = original_point.get_subpoint_over_range(0..constraint.num_variables());
            (slice.reversed(), false, 0)
        }
    };

    let eq_contribution = constraint
        .iter_eqs()
        .map(|(pt, coeff)| {
            let val = if use_skip_eval {
                eval_eq_with_skip(pt, &eval_point, k_skip)
            } else {
                pt.eq_poly(&eval_point)
            };
            val * coeff
        })
        .sum::<EF>();

    let linear_eq_contribution = if constraint.eq_statement.linear_weights.is_empty() {
        EF::ZERO
    } else {
        let point_eq_count = constraint.eq_statement.points.len();
        constraint
            .eq_statement
            .linear_weights
            .iter()
            .zip(constraint.challenge.powers().skip(point_eq_count))
            .map(|(weights, coeff)| {
                if use_skip_eval {
                    match weights {
                        LinearConstraint::Dense(weights) => {
                            eval_evals_with_skip(weights, &eval_point, k_skip) * coeff
                        }
                        LinearConstraint::TensorProduct {
                            range_start,
                            log_range_len,
                            row_weights,
                            col_weights,
                        } => {
                            eval_tensor_product_with_skip(
                                *range_start,
                                *log_range_len,
                                row_weights,
                                col_weights,
                                &eval_point,
                                k_skip,
                                constraint.num_variables(),
                            ) * coeff
                        }
                    }
                } else {
                    match weights {
                        LinearConstraint::Dense(weights) => {
                            weights.evaluate_hypercube_ext::<F>(&eval_point) * coeff
                        }
                        LinearConstraint::TensorProduct {
                            range_start,
                            log_range_len,
                            row_weights,
                            col_weights,
                        } => {
                            let num_vars = constraint.eq_statement.num_variables();
                            let row_len = col_weights.num_evals();
                            let log_row_len = log2_strict_usize(row_len);
                            let log_rows = *log_range_len - log_row_len;
                            let high_bits = num_vars - *log_range_len;
                            let point_slice = eval_point.as_slice();
                            let (high_slice, rest) = point_slice.split_at(high_bits);
                            let (row_slice, col_slice) = rest.split_at(log_rows);

                            let mut fixed_bits = (*log_range_len..num_vars)
                                .map(|i| EF::from_bool(((*range_start >> i) & 1) == 1))
                                .collect::<Vec<_>>();
                            fixed_bits.reverse();
                            let fixed_eq = MultilinearPoint::new(fixed_bits)
                                .eq_poly(&MultilinearPoint::new(high_slice.to_vec()));

                            let row_eval = row_weights.evaluate_hypercube_ext::<F>(
                                &MultilinearPoint::new(row_slice.to_vec()),
                            );
                            let col_eval = col_weights.evaluate_hypercube_ext::<F>(
                                &MultilinearPoint::new(col_slice.to_vec()),
                            );

                            fixed_eq * row_eval * col_eval * coeff
                        }
                    }
                }
            })
            .sum::<EF>()
    };

    let sel_contribution = constraint
        .iter_sels()
        .map(|(&var, coeff)| {
            let expanded =
                MultilinearPoint::expand_from_univariate(var, constraint.num_variables());
            coeff * expanded.select_poly(&eval_point)
        })
        .sum::<EF>();

    eq_contribution + linear_eq_contribution + sel_contribution
}

/// Lightweight evaluator for the combined constraint polynomial W(r).
#[derive(Clone, Debug)]
pub struct ConstraintPolyEvaluator {
    /// The folding factor.
    pub folding_factor: FoldingFactor,
    /// Optional skip step indicating whether the univariate skip optimization is active.
    pub univariate_skip: Option<usize>,
}

impl ConstraintPolyEvaluator {
    /// Creates a new `ConstraintPolyEvaluator` with skip disabled.
    #[must_use]
    pub const fn new(folding_factor: FoldingFactor) -> Self {
        Self {
            folding_factor,
            univariate_skip: None,
        }
    }

    /// Creates a new `ConstraintPolyEvaluator` with univariate skip enabled.
    #[must_use]
    pub const fn new_with_univariate_skip(folding_factor: FoldingFactor, k_skip: usize) -> Self {
        Self {
            folding_factor,
            univariate_skip: Some(k_skip),
        }
    }

    /// Evaluate the combined constraint polynomial W(r).
    #[must_use]
    pub fn eval_constraints_poly<F: Field, EF: ExtensionField<F> + TwoAdicField>(
        &self,
        constraints: &[Constraint<F, EF>],
        point: &MultilinearPoint<EF>,
    ) -> EF {
        if constraints.is_empty() {
            return EF::ZERO;
        }

        let context = if let Some(k_skip) = self.univariate_skip {
            self.prepare_skip_context(point, constraints.len().saturating_sub(1), k_skip)
        } else {
            PointContext::NonSkip
        };

        constraints
            .iter()
            .enumerate()
            .map(|(i, constraint)| eval_round(i, constraint, point, &context))
            .sum()
    }

    fn prepare_skip_context<EF: ExtensionField<impl Field> + TwoAdicField>(
        &self,
        point: &MultilinearPoint<EF>,
        num_prover_rounds: usize,
        k_skip: usize,
    ) -> PointContext<EF> {
        let mut rotated = point.as_slice()[1..].to_vec();
        rotated.push(point.as_slice()[0]);
        let rotated = MultilinearPoint::new(rotated);

        let mut offsets = vec![0];
        for round in 0..num_prover_rounds {
            offsets.push(offsets[round] + self.folding_factor.at_round(round + 1));
        }

        PointContext::Skip {
            rotated,
            prover_challenge_offsets: offsets,
            k_skip,
        }
    }
}

enum PointContext<EF> {
    NonSkip,
    Skip {
        rotated: MultilinearPoint<EF>,
        prover_challenge_offsets: Vec<usize>,
        k_skip: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec, vec::Vec};

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use proptest::prelude::*;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        parameters::FoldingFactor,
        poly::evals::EvaluationsList,
        whir::constraints::statement::{EqStatement, SelectStatement},
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_eval_constraints_poly() {
        let num_vars = 20;
        let folding_factor = FoldingFactor::Constant(5);
        let num_eq_constraints_per_round = &[2usize, 3, 1];
        let num_sel_constraints_per_round = &[31usize, 41, 51];

        let mut rng = SmallRng::seed_from_u64(0);
        let mut num_vars_at_round = num_vars;
        let mut constraints = vec![];

        for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
            .iter()
            .zip(num_sel_constraints_per_round.iter())
            .enumerate()
        {
            let gamma = rng.random();
            let mut eq_statement = EqStatement::initialize(num_vars_at_round);
            (0..num_eq).for_each(|_| {
                eq_statement.add_evaluated_constraint(
                    MultilinearPoint::rand(&mut rng, num_vars_at_round),
                    rng.random(),
                );
            });

            let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_at_round);
            (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
            constraints.push(Constraint::new(gamma, eq_statement, sel_statement));

            num_vars_at_round -= folding_factor.at_round(round_idx);
        }

        let final_point = MultilinearPoint::rand(&mut rng, num_vars);
        let evaluator = ConstraintPolyEvaluator::new(folding_factor);
        let result_from_eval_poly = evaluator.eval_constraints_poly(&constraints, &final_point);

        let expected_result = constraints
            .iter()
            .map(|constraint| {
                let num_vars = constraint.num_variables();
                let mut combined = EvaluationsList::zero(num_vars);
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);
                let point = final_point.get_subpoint_over_range(0..num_vars).reversed();
                combined.evaluate_hypercube_ext::<F>(&point)
            })
            .sum::<EF>();

        assert_eq!(result_from_eval_poly, expected_result);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly(
            (num_vars, folding_factor_val) in (10..=20usize)
                .prop_flat_map(|n| (Just(n), 2..=(n / 2)))
        ) {
            let mut num_vars_current = num_vars;
            let folding_factor = FoldingFactor::Constant(folding_factor_val);
            let mut folding_factors_vec = vec![];
            while num_vars_current > 0 {
                let num_to_fold = core::cmp::min(folding_factor_val, num_vars_current);
                if num_to_fold == 0 {
                    break;
                }
                folding_factors_vec.push(num_to_fold);
                num_vars_current -= num_to_fold;
            }
            let num_rounds = folding_factors_vec.len();

            let mut rng = SmallRng::seed_from_u64(0);
            let num_eq_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();
            let num_sel_constraints_per_round: Vec<usize> = (0..num_rounds)
                .map(|_| rng.random_range(0..=2))
                .collect();

            let mut num_vars_current = num_vars;
            let mut constraints = vec![];

            for (round_idx, (&num_eq, &num_sel)) in num_eq_constraints_per_round
                .iter()
                .zip(num_sel_constraints_per_round.iter())
                .enumerate()
            {
                let gamma = rng.random();
                let mut eq_statement = EqStatement::initialize(num_vars_current);
                (0..num_eq).for_each(|_| {
                    eq_statement.add_evaluated_constraint(
                        MultilinearPoint::rand(&mut rng, num_vars_current),
                        rng.random(),
                    );
                });

                let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars_current);
                (0..num_sel).for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));
                constraints.push(Constraint::new(gamma, eq_statement, sel_statement));

                num_vars_current -= folding_factors_vec[round_idx];
            }

            let final_point = MultilinearPoint::rand(&mut rng, num_vars);
            let evaluator = ConstraintPolyEvaluator::new(folding_factor);
            let result_from_eval_poly = evaluator.eval_constraints_poly(&constraints, &final_point);

            let mut num_vars_at_round = num_vars;
            let expected_result = constraints
                .iter()
                .enumerate()
                .map(|(round_idx, constraint)| {
                    let point = final_point
                        .get_subpoint_over_range(0..num_vars_at_round)
                        .reversed();
                    let mut combined = EvaluationsList::zero(constraint.num_variables());
                    let mut eval = EF::ZERO;
                    constraint.combine(&mut combined, &mut eval);
                    num_vars_at_round -= folding_factors_vec[round_idx];
                    combined.evaluate_hypercube_ext::<F>(&point)
                })
                .sum::<EF>();

            prop_assert_eq!(result_from_eval_poly, expected_result);
        }
    }
}

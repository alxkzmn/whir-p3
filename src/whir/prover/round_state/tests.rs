use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_matrix::dense::DenseMatrix;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use crate::{
    fiat_shamir::domain_separator::DomainSeparator,
    parameters::{FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        WhirConfig,
        committer::{Witness, writer::CommitmentWriter},
        constraints::statement::EqStatement,
        parameters::InitialPhaseConfig,
        proof::WhirProof,
        prover::{Prover, round_state::RoundState},
    },
};

type F = BabyBear;
type EF4 = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

/// Field/Poseidon-based transcript + Merkle configuration.
///
/// This module is always compiled; when `feature="keccak"` is enabled you’ll get **both**
/// the field tests and the keccak tests below.
mod field {

    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    /// Poseidon-based Merkle digests are `[F; 8]` with the current sponge parameters.
    const DIGEST_ELEMS: usize = 8;

    /// Create a WHIR protocol configuration for test scenarios.
    ///
    /// This utility function builds a `WhirConfig` using the provided parameters:
    /// - `num_variables`: Number of variables in the multilinear polynomial.
    /// - `initial_phase_config`: Configuration for the initial phase.
    /// - `folding_factor`: Number of variables to fold per round.
    /// - `pow_bits`: Difficulty of the proof-of-work challenge used in Fiat-Shamir.
    ///
    /// The returned config can be used to initialize a prover and set up domain commitments
    /// for round state construction in WHIR tests.   
    fn make_test_config(
        num_variables: usize,
        initial_phase_config: InitialPhaseConfig,
        folding_factor: usize,
        pow_bits: usize,
    ) -> WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger> {
        let merkle_hash = MyHash::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1)));
        let merkle_compress =
            MyCompress::new(Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(2)));

        // Define the core protocol parameters for WHIR, customizing behavior based
        // on whether to start with an initial sumcheck and how to fold the polynomial.
        let protocol_params = ProtocolParameters {
            initial_phase_config,
            security_level: 80,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(folding_factor),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        // Combine the multivariate and protocol parameters into a full WHIR config
        WhirConfig::new(num_variables, protocol_params)
    }

    /// Prepare the Fiat-Shamir domain, prover state, and Merkle commitment for a test polynomial.
    ///
    /// This helper sets up the necessary transcript (`DomainSeparator`) and
    /// commits to a polynomial using a `CommitmentWriter`. It returns:
    /// - the initialized domain separator,
    /// - the `ProverState` transcript context for Fiat-Shamir interaction,
    /// - and a `Witness` object containing the committed polynomial and Merkle data.
    ///
    /// This is used as a boilerplate step before running the first WHIR round.
    #[allow(clippy::type_complexity)]
    fn setup_domain_and_commitment(
        params: &WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger>,
        poly: EvaluationsList<F>,
    ) -> (
        WhirProof<F, EF4, DIGEST_ELEMS>,
        MyChallenger,
        Witness<EF4, F, DenseMatrix<F>, DIGEST_ELEMS>,
    ) {
        // Build ProtocolParameters from WhirConfig fields
        let protocol_params = ProtocolParameters {
            initial_phase_config: params.initial_phase_config,
            security_level: params.security_level,
            pow_bits: params.starting_folding_pow_bits,
            folding_factor: params.folding_factor,
            merkle_hash: params.merkle_hash.clone(),
            merkle_compress: params.merkle_compress.clone(),
            soundness_type: params.soundness_type,
            starting_log_inv_rate: params.starting_log_inv_rate,
            rs_domain_initial_reduction_factor: 1,
        };
        // Create a new Fiat-Shamir domain separator.
        let mut domsep = DomainSeparator::new(vec![]);
        // Observe the public statement into the transcript for binding.
        domsep.commit_statement::<_, _, _, DIGEST_ELEMS>(params);
        // Reserve transcript space for WHIR proof messages.
        domsep.add_whir_proof::<_, _, _, DIGEST_ELEMS>(params);

        // Convert the domain separator into a mutable prover-side transcript.
        let mut rng = SmallRng::seed_from_u64(1);
        let mut prover_challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domsep.observe_domain_separator(&mut prover_challenger);

        // Create a committer using the protocol configuration (Merkle parameters, hashers, etc.).
        let committer = CommitmentWriter::new(params);

        let mut proof = WhirProof::from_protocol_parameters(&protocol_params, poly.num_variables());

        // Perform DFT-based commitment to the polynomial, producing a witness
        // which includes the Merkle tree and polynomial values.
        let witness = committer
            .commit(
                &Radix2DFTSmallBatch::<F>::default(),
                &mut proof,
                &mut prover_challenger,
                poly,
            )
            .unwrap();

        // Return all initialized components needed for round state setup.
        (proof, prover_challenger, witness)
    }

    #[test]
    fn test_no_initial_statement_no_sumcheck() {
        // Number of variables in the multilinear polynomial
        let num_variables = 2;

        // Create a WHIR protocol config with:
        // - no initial sumcheck,
        // - folding factor 2,
        // - no PoW grinding.
        let config = make_test_config(num_variables, InitialPhaseConfig::WithoutStatement, 2, 0);
        // Define a polynomial
        let poly = EvaluationsList::new(vec![F::from_u64(3); 1 << num_variables]);

        // Initialize:
        // - domain separator for Fiat-Shamir transcript,
        // - prover state,
        // - witness containing Merkle tree for `poly`.
        let (mut proof, mut challenger, witness) = setup_domain_and_commitment(&config, poly);
        // Create an empty public statement (no constraints)
        let statement = EqStatement::<EF4>::initialize(num_variables);

        // Initialize the round state using the setup configuration and witness
        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger,
            statement,
            witness,
        )
        .unwrap();

        // Folding factor was 2, so we expect 2 sampled folding randomness values
        assert_eq!(state.folding_randomness.num_variables(), 2);

        // Since this is the first round, no Merkle data for folded rounds should exist
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_initial_statement_with_folding_factor_3() {
        // Set the number of variables in the multilinear polynomial
        let num_variables = 3;

        // Create a WHIR configuration with:
        // - initial statement enabled (sumcheck will run),
        // - folding factor = 3 (fold all variables in the first round),
        // - PoW disabled.
        let config = make_test_config(
            num_variables,
            InitialPhaseConfig::WithStatementClassic,
            3,
            0,
        );

        // Define the multilinear polynomial:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2
        //              + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let e1 = F::from_u64(1);
        let e2 = F::from_u64(2);
        let e3 = F::from_u64(3);
        let e4 = F::from_u64(4);
        let e5 = F::from_u64(5);
        let e6 = F::from_u64(6);
        let e7 = F::from_u64(7);
        let e8 = F::from_u64(8);

        let poly = EvaluationsList::new(vec![
            e1,
            e1 + e2,
            e1 + e3,
            e1 + e2 + e3 + e4,
            e1 + e5,
            e1 + e2 + e5 + e6,
            e1 + e3 + e5 + e7,
            e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
        ]);

        // Manual redefinition of the same polynomial as a function for evaluation
        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * e2
                + x1 * e3
                + x1 * x2 * e4
                + x0 * e5
                + x0 * x2 * e6
                + x0 * x1 * e7
                + x0 * x1 * x2 * e8
                + e1
        };

        // Add a single equality constraint to the statement: f(1,1,1) = expected value
        let mut statement = EqStatement::<EF4>::initialize(num_variables);
        statement.add_evaluated_constraint(
            MultilinearPoint::new(vec![EF4::ONE, EF4::ONE, EF4::ONE]),
            f(EF4::ONE, EF4::ONE, EF4::ONE),
        );

        // Set up the domain separator, prover state, and witness for this configuration
        let (mut proof, mut challenger_rf, witness) = setup_domain_and_commitment(&config, poly);

        // Run the first round state initialization (this will trigger sumcheck)
        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger_rf,
            statement,
            witness,
        )
        .unwrap();

        // Extract the constructed sumcheck prover and folding randomness
        let sumcheck = &state.sumcheck_prover;
        let sumcheck_randomness = state.folding_randomness.clone();

        // With a folding factor of 3, all variables are collapsed in 1 round, so we expect only 1 evaluation left
        assert_eq!(sumcheck.poly.num_evals(), 1);

        // The value of f at the folding point should match the evaluation
        let eval_at_point = sumcheck.evals().as_slice()[0];
        let expected = f(
            sumcheck_randomness[0],
            sumcheck_randomness[1],
            sumcheck_randomness[2],
        );
        assert_eq!(eval_at_point, expected);

        // Check that dot product of evaluations and weights matches the final sum
        let dot_product: EF4 = sumcheck.poly.dot_product();
        assert_eq!(dot_product, sumcheck.sum);

        // The `folding_randomness` should store values in forward order (X0, X1, X2)
        assert_eq!(
            state.folding_randomness.as_slice(),
            vec![
                sumcheck_randomness[0],
                sumcheck_randomness[1],
                sumcheck_randomness[2]
            ]
        );

        // No folded Merkle tree data should exist at this point
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_zero_poly_multiple_constraints() {
        // Use a polynomial with 3 variables
        let num_variables = 3;

        // Build a WHIR config with an initial statement, folding factor 1, and no PoW
        let config = make_test_config(
            num_variables,
            InitialPhaseConfig::WithStatementClassic,
            1,
            0,
        );

        // Define a zero polynomial: f(X) = 0 for all X
        let poly = EvaluationsList::new(vec![F::ZERO; 1 << num_variables]);

        // Generate domain separator, prover state, and Merkle commitment witness for the poly
        let (mut proof, mut challenger_rf, witness) = setup_domain_and_commitment(&config, poly);

        // Create a new statement with multiple constraints
        let mut statement = EqStatement::<EF4>::initialize(num_variables);

        // Add one equality constraint per Boolean input: f(x) = 0 for all x ∈ {0,1}³
        for i in 0..1 << num_variables {
            let point = (0..num_variables)
                .map(|b| EF4::from_u64(((i >> b) & 1) as u64))
                .collect();
            statement.add_evaluated_constraint(MultilinearPoint::new(point), EF4::ZERO);
        }

        // Initialize the first round of the WHIR protocol with the zero polynomial and constraints
        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger_rf,
            statement,
            witness,
        )
        .unwrap();

        // Extract the sumcheck prover and folding randomness
        let sumcheck = &state.sumcheck_prover;
        let sumcheck_randomness = state.folding_randomness.clone();

        for (f, w) in sumcheck.evals().iter().zip(&sumcheck.weights()) {
            // Each evaluation should be 0
            assert_eq!(*f, EF4::ZERO);
            // Their contribution to the weighted sum should also be 0
            assert_eq!(*f * *w, EF4::ZERO);
        }
        // Final claimed sum is 0
        assert_eq!(sumcheck.sum, EF4::ZERO);

        // Folding randomness should have length equal to the folding factor (1)
        assert_eq!(sumcheck_randomness.num_variables(), 1);

        // Confirm that folding randomness matches exactly
        assert_eq!(
            state.folding_randomness,
            MultilinearPoint::new(vec![sumcheck_randomness[0]])
        );

        // No Merkle commitment data for folded rounds yet
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_initialize_round_state_with_initial_statement() {
        // Use a polynomial in 3 variables
        let num_variables = 3;

        // Set PoW grinding difficulty (used in Fiat-Shamir)
        let pow_bits = 4;

        // Build a WHIR configuration with:
        // - initial statement enabled,
        // - folding factor of 1 (fold one variable in the first round),
        // - PoW bits enabled.
        let config = make_test_config(
            num_variables,
            InitialPhaseConfig::WithStatementClassic,
            1,
            pow_bits,
        );

        // Define a multilinear polynomial:
        // f(X0, X1, X2) = 1 + 2*X2 + 3*X1 + 4*X1*X2 + 5*X0 + 6*X0*X2 + 7*X0*X1 + 8*X0*X1*X2
        let e1 = F::from_u64(1);
        let e2 = F::from_u64(2);
        let e3 = F::from_u64(3);
        let e4 = F::from_u64(4);
        let e5 = F::from_u64(5);
        let e6 = F::from_u64(6);
        let e7 = F::from_u64(7);
        let e8 = F::from_u64(8);

        let poly = EvaluationsList::new(vec![
            e1,
            e1 + e2,
            e1 + e3,
            e1 + e2 + e3 + e4,
            e1 + e5,
            e1 + e2 + e5 + e6,
            e1 + e3 + e5 + e7,
            e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
        ]);

        // Equivalent function for evaluating the polynomial manually
        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * e2
                + x1 * e3
                + x1 * x2 * e4
                + x0 * e5
                + x0 * x2 * e6
                + x0 * x1 * e7
                + x0 * x1 * x2 * e8
                + e1
        };

        // Construct a statement with one evaluation constraint at the point (1, 0, 1)
        let mut statement = EqStatement::<EF4>::initialize(num_variables);
        statement.add_evaluated_constraint(
            MultilinearPoint::new(vec![EF4::ONE, EF4::ZERO, EF4::ONE]),
            f(EF4::ONE, EF4::ZERO, EF4::ONE),
        );

        // Set up Fiat-Shamir domain and produce commitment + witness
        // Generate domain separator, prover state, and Merkle commitment witness for the poly
        let (mut proof, mut challenger_rf, witness) = setup_domain_and_commitment(&config, poly);

        // Run the first round initialization
        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger_rf,
            statement,
            witness,
        )
        .expect("RoundState initialization failed");

        // Unwrap the sumcheck prover and get the sampled folding randomness
        let sumcheck = &state.sumcheck_prover;
        let sumcheck_randomness = &state.folding_randomness;

        // Evaluate f at (32636, 9876, r0) and match it with the sumcheck's recovered evaluation
        let evals_f = &sumcheck.evals();
        assert_eq!(
            evals_f.evaluate_hypercube_ext::<F>(&MultilinearPoint::new(vec![
                EF4::from_u64(32636),
                EF4::from_u64(9876)
            ])),
            f(
                sumcheck_randomness[0],
                EF4::from_u64(32636),
                EF4::from_u64(9876),
            )
        );

        // Manually verify that ⟨f, w⟩ = claimed sum
        assert_eq!(sumcheck.poly.dot_product(), sumcheck.sum);

        // No Merkle tree data has been created for folded rounds yet
        assert!(state.merkle_prover_data.is_none());

        // The folding randomness must match what was sampled by the sumcheck
        assert_eq!(
            state.folding_randomness,
            MultilinearPoint::new(vec![sumcheck_randomness[0]])
        );
    }
}

/// Keccak(bytes32)-based transcript + Merkle configuration.
#[cfg(feature = "keccak")]
mod keccak {

    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_keccak::Keccak256Hash;

    use super::*;
    use crate::keccak_mmcs::{KeccakNodeCompress, KeccakU32BeLeafHasher};

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;
    type MyHash = KeccakU32BeLeafHasher;
    type MyCompress = KeccakNodeCompress;
    type MyChallenger = SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>>;

    /// Keccak Merkle digests are bytes32, i.e. `[u8; 32]`.
    const DIGEST_ELEMS: usize = 32;

    fn make_test_config(
        num_variables: usize,
        initial_phase_config: InitialPhaseConfig,
        folding_factor: usize,
        pow_bits: usize,
    ) -> WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger> {
        let merkle_hash = MyHash::default();
        let merkle_compress = MyCompress::default();

        let protocol_params = ProtocolParameters {
            initial_phase_config,
            security_level: 80,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(folding_factor),
            merkle_hash,
            merkle_compress,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };

        WhirConfig::new(num_variables, protocol_params)
    }

    #[allow(clippy::type_complexity)]
    fn setup_domain_and_commitment(
        params: &WhirConfig<EF4, F, MyHash, MyCompress, MyChallenger>,
        poly: EvaluationsList<F>,
    ) -> (
        WhirProof<F, EF4, DIGEST_ELEMS, u8>,
        MyChallenger,
        Witness<EF4, F, DenseMatrix<F>, DIGEST_ELEMS, u8>,
    ) {
        let protocol_params = ProtocolParameters {
            initial_phase_config: params.initial_phase_config,
            security_level: params.security_level,
            pow_bits: params.starting_folding_pow_bits,
            folding_factor: params.folding_factor,
            merkle_hash: params.merkle_hash.clone(),
            merkle_compress: params.merkle_compress.clone(),
            soundness_type: params.soundness_type,
            starting_log_inv_rate: params.starting_log_inv_rate,
            rs_domain_initial_reduction_factor: 1,
        };

        let mut domsep = DomainSeparator::new(vec![]);
        domsep.commit_statement::<_, _, _, DIGEST_ELEMS>(params);
        domsep.add_whir_proof::<_, _, _, DIGEST_ELEMS>(params);

        let mut prover_challenger = MyChallenger::from_hasher(vec![], Keccak256Hash);
        domsep.observe_domain_separator(&mut prover_challenger);

        let committer = CommitmentWriter::new(params);
        let mut proof = WhirProof::<F, EF4, DIGEST_ELEMS, u8>::from_protocol_parameters(
            &protocol_params,
            poly.num_variables(),
        );

        let witness = committer
            .commit(
                &Radix2DFTSmallBatch::<F>::default(),
                &mut proof,
                &mut prover_challenger,
                poly,
            )
            .unwrap();

        (proof, prover_challenger, witness)
    }

    #[test]
    fn test_no_initial_statement_no_sumcheck() {
        let num_variables = 2;
        let config = make_test_config(num_variables, InitialPhaseConfig::WithoutStatement, 2, 0);
        let poly = EvaluationsList::new(vec![F::from_u64(3); 1 << num_variables]);

        let (mut proof, mut challenger, witness) = setup_domain_and_commitment(&config, poly);
        let statement = EqStatement::<EF4>::initialize(num_variables);

        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger,
            statement,
            witness,
        )
        .unwrap();

        assert_eq!(state.folding_randomness.num_variables(), 2);
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_initial_statement_with_folding_factor_3() {
        let num_variables = 3;
        let config = make_test_config(
            num_variables,
            InitialPhaseConfig::WithStatementClassic,
            3,
            0,
        );

        let e1 = F::from_u64(1);
        let e2 = F::from_u64(2);
        let e3 = F::from_u64(3);
        let e4 = F::from_u64(4);
        let e5 = F::from_u64(5);
        let e6 = F::from_u64(6);
        let e7 = F::from_u64(7);
        let e8 = F::from_u64(8);

        let poly = EvaluationsList::new(vec![
            e1,
            e1 + e2,
            e1 + e3,
            e1 + e2 + e3 + e4,
            e1 + e5,
            e1 + e2 + e5 + e6,
            e1 + e3 + e5 + e7,
            e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
        ]);

        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * e2
                + x1 * e3
                + x1 * x2 * e4
                + x0 * e5
                + x0 * x2 * e6
                + x0 * x1 * e7
                + x0 * x1 * x2 * e8
                + e1
        };

        let mut statement = EqStatement::<EF4>::initialize(num_variables);
        statement.add_evaluated_constraint(
            MultilinearPoint::new(vec![EF4::ONE, EF4::ONE, EF4::ONE]),
            f(EF4::ONE, EF4::ONE, EF4::ONE),
        );

        let (mut proof, mut challenger, witness) = setup_domain_and_commitment(&config, poly);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger,
            statement,
            witness,
        )
        .unwrap();

        let sumcheck = &state.sumcheck_prover;
        let r = state.folding_randomness.clone();

        assert_eq!(sumcheck.num_evals(), 1);
        assert_eq!(sumcheck.evals().as_slice()[0], f(r[0], r[1], r[2]));
        assert_eq!(sumcheck.poly.dot_product(), sumcheck.sum);
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_zero_poly_multiple_constraints() {
        let num_variables = 3;
        let config = make_test_config(
            num_variables,
            InitialPhaseConfig::WithStatementClassic,
            1,
            0,
        );

        let poly = EvaluationsList::new(vec![F::ZERO; 1 << num_variables]);
        let (mut proof, mut challenger, witness) = setup_domain_and_commitment(&config, poly);

        let mut statement = EqStatement::<EF4>::initialize(num_variables);
        for i in 0..1 << num_variables {
            let point = (0..num_variables)
                .map(|b| EF4::from_u64(((i >> b) & 1) as u64))
                .collect();
            statement.add_evaluated_constraint(MultilinearPoint::new(point), EF4::ZERO);
        }

        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger,
            statement,
            witness,
        )
        .unwrap();

        let sumcheck = &state.sumcheck_prover;
        for (f, w) in sumcheck.evals().iter().zip(&sumcheck.weights()) {
            assert_eq!(*f, EF4::ZERO);
            assert_eq!(*f * *w, EF4::ZERO);
        }
        assert_eq!(sumcheck.sum, EF4::ZERO);
        assert_eq!(state.folding_randomness.num_variables(), 1);
        assert!(state.merkle_prover_data.is_none());
    }

    #[test]
    fn test_initialize_round_state_with_initial_statement() {
        let num_variables = 3;
        let pow_bits = 4;
        let config = make_test_config(
            num_variables,
            InitialPhaseConfig::WithStatementClassic,
            1,
            pow_bits,
        );

        let e1 = F::from_u64(1);
        let e2 = F::from_u64(2);
        let e3 = F::from_u64(3);
        let e4 = F::from_u64(4);
        let e5 = F::from_u64(5);
        let e6 = F::from_u64(6);
        let e7 = F::from_u64(7);
        let e8 = F::from_u64(8);

        let poly = EvaluationsList::new(vec![
            e1,
            e1 + e2,
            e1 + e3,
            e1 + e2 + e3 + e4,
            e1 + e5,
            e1 + e2 + e5 + e6,
            e1 + e3 + e5 + e7,
            e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8,
        ]);

        let f = |x0: EF4, x1: EF4, x2: EF4| {
            x2 * e2
                + x1 * e3
                + x1 * x2 * e4
                + x0 * e5
                + x0 * x2 * e6
                + x0 * x1 * e7
                + x0 * x1 * x2 * e8
                + e1
        };

        let mut statement = EqStatement::<EF4>::initialize(num_variables);
        statement.add_evaluated_constraint(
            MultilinearPoint::new(vec![EF4::ONE, EF4::ZERO, EF4::ONE]),
            f(EF4::ONE, EF4::ZERO, EF4::ONE),
        );

        let (mut proof, mut challenger, witness) = setup_domain_and_commitment(&config, poly);
        let dft = Radix2DFTSmallBatch::<F>::default();
        let state = RoundState::initialize_first_round_state(
            &dft,
            &Prover(&config),
            &mut proof,
            &mut challenger,
            statement,
            witness,
        )
        .unwrap();

        let sumcheck = &state.sumcheck_prover;
        let r = &state.folding_randomness;

        assert_eq!(sumcheck.poly.dot_product(), sumcheck.sum);
        assert!(state.merkle_prover_data.is_none());

        // Spot-check the same deterministic evaluation identity used by the field test.
        assert_eq!(
            sumcheck
                .evals()
                .evaluate_hypercube_ext::<F>(&MultilinearPoint::new(vec![
                    EF4::from_u64(32636),
                    EF4::from_u64(9876)
                ])),
            f(r[0], EF4::from_u64(32636), EF4::from_u64(9876))
        );
    }
}

#[cfg(feature = "keccak")]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "keccak")]
use p3_challenger::{HashChallenger, SerializingChallenger32};
#[cfg(feature = "keccak")]
use p3_dft::Radix2DFTSmallBatch;
#[cfg(feature = "keccak")]
use p3_field::extension::BinomialExtensionField;
#[cfg(feature = "keccak")]
use p3_keccak::Keccak256Hash;
#[cfg(feature = "keccak")]
use p3_koala_bear::KoalaBear;
#[cfg(feature = "keccak")]
use rand::{Rng, SeedableRng, rngs::SmallRng};
#[cfg(feature = "keccak")]
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator,
    keccak_mmcs::{KeccakNodeCompress, KeccakU32BeLeafHasher},
    parameters::{DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, errors::SecurityAssumption},
    poly::{evals::EvaluationsList, multilinear::MultilinearPoint},
    whir::{
        committer::writer::CommitmentWriter,
        constraints::statement::EqStatement,
        parameters::{InitialPhaseConfig, WhirConfig},
        proof::WhirProof,
        prover::Prover,
    },
};

#[cfg(feature = "keccak")]
type F = KoalaBear;
#[cfg(feature = "keccak")]
type EF = BinomialExtensionField<F, 4>;

#[cfg(feature = "keccak")]
type MerkleHashKeccak = KeccakU32BeLeafHasher;
#[cfg(feature = "keccak")]
type MerkleCompressKeccak = KeccakNodeCompress;
#[cfg(feature = "keccak")]
type MyChallengerKeccak = SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>>;
#[cfg(feature = "keccak")]
const DIGEST_ELEMS: usize = 32;

#[cfg(feature = "keccak")]
#[allow(clippy::type_complexity)]
fn prepare_inputs_keccak() -> (
    WhirConfig<EF, F, MerkleHashKeccak, MerkleCompressKeccak, MyChallengerKeccak>,
    ProtocolParameters<MerkleHashKeccak, MerkleCompressKeccak>,
    usize,
    Radix2DFTSmallBatch<F>,
    EvaluationsList<F>,
    EqStatement<EF>,
    DomainSeparator<EF, F>,
) {
    // Protocol parameter configuration

    // Target cryptographic security in bits.
    let security_level = 100;

    // Number of Boolean variables in the multilinear polynomial. Polynomial has 2^24 coefficients.
    let num_variables = 24;

    // Number of PoW bits required, computed based on the domain size and rate.
    let pow_bits = DEFAULT_MAX_POW;

    // Folding factor `k`: number of variables folded per round in the sumcheck.
    let folding_factor = FoldingFactor::Constant(4);

    // Low-degree extension (LDE) blowup factor: inverse of `rate`.
    let starting_rate = 1;

    // RS code initial domain size reduction factor (controls the LDE domain size).
    let rs_domain_initial_reduction_factor = 3;

    // Create multivariate polynomial and hash setup

    // Define the hash functions for Merkle tree and compression (Keccak uses defaults).
    let merkle_hash = MerkleHashKeccak::default();
    let merkle_compress = MerkleCompressKeccak::default();

    // Type of soundness assumption used in the IOP model.
    let soundness_type = SecurityAssumption::CapacityBound;

    // Assemble the protocol-level parameters.
    let whir_params = ProtocolParameters {
        initial_phase_config: InitialPhaseConfig::WithStatementClassic,
        security_level,
        pow_bits,
        folding_factor,
        merkle_hash,
        merkle_compress,
        soundness_type,
        starting_log_inv_rate: starting_rate,
        rs_domain_initial_reduction_factor,
    };

    // Combine multivariate and protocol parameters into a unified WHIR config.
    let params = WhirConfig::new(num_variables, whir_params.clone());

    // Sample random multilinear polynomial

    // Total number of coefficients = 2^num_variables.
    let num_coeffs = 1 << num_variables;

    // Use a fixed-seed RNG to ensure deterministic benchmark inputs.
    let mut rng = SmallRng::seed_from_u64(0);

    // Sample a random multilinear polynomial over `F`, represented by its evaluations.
    let polynomial = EvaluationsList::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());

    // Build a simple constraint system with one point

    // Sample a random Boolean point in {0,1}^num_variables.
    let point = MultilinearPoint::rand(&mut rng, num_variables);

    // Create a new WHIR `Statement` with one constraint.
    let mut statement = EqStatement::<EF>::initialize(num_variables);
    statement.add_unevaluated_constraint_hypercube(point, &polynomial);

    // Fiat-Shamir setup

    // Create a domain separator for transcript hashing.
    let mut domainsep = DomainSeparator::new(vec![]);

    // Commit protocol parameters and proof type to the domain separator.
    domainsep.commit_statement::<_, _, _, 32>(&params);
    domainsep.add_whir_proof::<_, _, _, 32>(&params);

    // DFT backend setup

    // Construct a Radix-2 FFT backend that supports small batch DFTs over `F`.
    let dft = Radix2DFTSmallBatch::<F>::new(1 << params.max_fft_size());

    // Return all preprocessed components needed to run commit/prove/verify benchmarks.
    (
        params,
        whir_params,
        num_variables,
        dft,
        polynomial,
        statement,
        domainsep,
    )
}

#[cfg(feature = "keccak")]
fn benchmark_commit_and_prove_keccak(c: &mut Criterion) {
    let (params, whir_params, num_variables, dft, polynomial, statement, domainsep) =
        prepare_inputs_keccak();

    c.bench_function("commit_keccak", |b| {
        b.iter(|| {
            let mut challenger_clone = MyChallengerKeccak::from_hasher(vec![], Keccak256Hash);
            domainsep.observe_domain_separator(&mut challenger_clone);
            let mut proof = WhirProof::<F, EF, DIGEST_ELEMS, u8>::from_protocol_parameters(
                &whir_params,
                num_variables,
            );
            let committer = CommitmentWriter::new(&params);
            let _witness = committer
                .commit(&dft, &mut proof, &mut challenger_clone, polynomial.clone())
                .unwrap();
        });
    });

    c.bench_function("prove_keccak", |b| {
        b.iter(|| {
            let mut challenger_clone = MyChallengerKeccak::from_hasher(vec![], Keccak256Hash);
            domainsep.observe_domain_separator(&mut challenger_clone);
            let mut proof = WhirProof::<F, EF, DIGEST_ELEMS, u8>::from_protocol_parameters(
                &whir_params,
                num_variables,
            );
            let committer = CommitmentWriter::new(&params);
            let witness = committer
                .commit(&dft, &mut proof, &mut challenger_clone, polynomial.clone())
                .unwrap();

            let prover = Prover(&params);
            prover
                .prove(
                    &dft,
                    &mut proof,
                    &mut challenger_clone,
                    statement.clone(),
                    witness,
                )
                .unwrap();
        });
    });
}

#[cfg(feature = "keccak")]
criterion_group!(benches, benchmark_commit_and_prove_keccak);

#[cfg(feature = "keccak")]
criterion_main!(benches);

#[cfg(not(feature = "keccak"))]
fn main() {
    panic!(
        "Keccak benchmarks require the 'keccak' feature to be enabled. Build with: cargo bench --features keccak --bench whir_keccak"
    );
}

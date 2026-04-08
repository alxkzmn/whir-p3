use alloc::vec::Vec;

use hashbrown::{HashMap, hash_map::Entry};
use p3_field::{ExtensionField, Field};
use p3_symmetric::CryptographicHasher;
use serde::{Deserialize, Serialize};

use crate::metrics::{add_leaf_hash_call, add_node_hash_call};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "W: Serialize, [W; DIGEST_ELEMS]: Serialize",
    deserialize = "W: Deserialize<'de>, [W; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct MerkleMultiProof<W, const DIGEST_ELEMS: usize> {
    pub decommitments: Vec<[W; DIGEST_ELEMS]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MultiproofError {
    EmptyIndices,
    LengthMismatch {
        indices: usize,
        openings: usize,
    },
    IndicesNotStrictlyIncreasing {
        prev: usize,
        next: usize,
    },
    InconsistentPathLength {
        expected: usize,
        got: usize,
    },
    InconsistentSiblingHash {
        level: usize,
        node: usize,
    },
    MissingSiblingHash {
        level: usize,
        node: usize,
    },
    InsufficientDecommitments {
        expected_at_least: usize,
        got: usize,
    },
    TrailingDecommitments {
        consumed: usize,
        total: usize,
    },
    InvalidFinalLayer {
        layer_size: usize,
        index: usize,
    },
    RootMismatch {
        query: usize,
    },
}

fn ensure_sorted_unique(indices: &[usize]) -> Result<(), MultiproofError> {
    if indices.is_empty() {
        return Err(MultiproofError::EmptyIndices);
    }
    for pair in indices.windows(2) {
        if pair[0] >= pair[1] {
            return Err(MultiproofError::IndicesNotStrictlyIncreasing {
                prev: pair[0],
                next: pair[1],
            });
        }
    }
    Ok(())
}

pub fn build_multiproof_from_paths<W: Clone + Eq, const DIGEST_ELEMS: usize>(
    indices: &[usize],
    opening_paths: Vec<Vec<[W; DIGEST_ELEMS]>>,
) -> Result<MerkleMultiProof<W, DIGEST_ELEMS>, MultiproofError> {
    ensure_sorted_unique(indices)?;
    if indices.len() != opening_paths.len() {
        return Err(MultiproofError::LengthMismatch {
            indices: indices.len(),
            openings: opening_paths.len(),
        });
    }

    let depth = opening_paths.first().map_or(0, Vec::len);
    for path in &opening_paths {
        if path.len() != depth {
            return Err(MultiproofError::InconsistentPathLength {
                expected: depth,
                got: path.len(),
            });
        }
    }

    let mut sibling_lookup: HashMap<(usize, usize), [W; DIGEST_ELEMS]> = HashMap::new();
    for (&leaf_index, path) in indices.iter().zip(opening_paths.into_iter()) {
        let mut node_index = leaf_index;
        for (level, sibling_hash) in path.into_iter().enumerate() {
            let sibling_index = node_index ^ 1;
            match sibling_lookup.entry((level, sibling_index)) {
                Entry::Vacant(slot) => {
                    slot.insert(sibling_hash);
                }
                Entry::Occupied(existing) => {
                    if existing.get() != &sibling_hash {
                        return Err(MultiproofError::InconsistentSiblingHash {
                            level,
                            node: sibling_index,
                        });
                    }
                }
            }
            node_index >>= 1;
        }
    }

    let mut frontier = indices.to_vec();
    let mut decommitments = Vec::new();

    for level in 0..depth {
        let mut next_frontier = Vec::with_capacity(frontier.len().div_ceil(2));
        let mut cursor = 0;
        while cursor < frontier.len() {
            let node = frontier[cursor];
            if node & 1 == 0 && cursor + 1 < frontier.len() && frontier[cursor + 1] == node + 1 {
                cursor += 2;
            } else {
                let sibling = node ^ 1;
                let Some(hash) = sibling_lookup.get(&(level, sibling)) else {
                    return Err(MultiproofError::MissingSiblingHash {
                        level,
                        node: sibling,
                    });
                };
                decommitments.push(hash.clone());
                cursor += 1;
            }
            next_frontier.push(node >> 1);
        }
        next_frontier.dedup();
        frontier = next_frontier;
    }

    Ok(MerkleMultiProof { decommitments })
}

pub fn build_linearized_multiproof_from_paths<W: Clone, const DIGEST_ELEMS: usize>(
    indices: &[usize],
    opening_paths: Vec<Vec<[W; DIGEST_ELEMS]>>,
) -> Result<MerkleMultiProof<W, DIGEST_ELEMS>, MultiproofError> {
    ensure_sorted_unique(indices)?;
    if indices.len() != opening_paths.len() {
        return Err(MultiproofError::LengthMismatch {
            indices: indices.len(),
            openings: opening_paths.len(),
        });
    }

    let depth = opening_paths.first().map_or(0, Vec::len);
    for path in &opening_paths {
        if path.len() != depth {
            return Err(MultiproofError::InconsistentPathLength {
                expected: depth,
                got: path.len(),
            });
        }
    }

    Ok(MerkleMultiProof {
        decommitments: opening_paths.into_iter().flatten().collect(),
    })
}

pub fn compute_root_from_multiproof<W: Copy + Eq, C, const DIGEST_ELEMS: usize>(
    indices: &[usize],
    leaf_hashes: &[[W; DIGEST_ELEMS]],
    depth: usize,
    decommitments: &[[W; DIGEST_ELEMS]],
    mut compress: C,
) -> Result<[W; DIGEST_ELEMS], MultiproofError>
where
    C: FnMut([[W; DIGEST_ELEMS]; 2]) -> [W; DIGEST_ELEMS],
{
    ensure_sorted_unique(indices)?;
    if indices.len() != leaf_hashes.len() {
        return Err(MultiproofError::LengthMismatch {
            indices: indices.len(),
            openings: leaf_hashes.len(),
        });
    }

    let mut frontier: Vec<(usize, [W; DIGEST_ELEMS])> = indices
        .iter()
        .copied()
        .zip(leaf_hashes.iter().copied())
        .collect();

    let mut decommitment_cursor = 0usize;

    for _ in 0..depth {
        let mut next_frontier = Vec::with_capacity(frontier.len().div_ceil(2));
        let mut cursor = 0usize;

        while cursor < frontier.len() {
            let (node, hash) = frontier[cursor];

            let parent_hash = if node & 1 == 0
                && cursor + 1 < frontier.len()
                && frontier[cursor + 1].0 == node + 1
            {
                let sibling_hash = frontier[cursor + 1].1;
                cursor += 2;
                add_node_hash_call();
                compress([hash, sibling_hash])
            } else {
                let Some(sibling_hash) = decommitments.get(decommitment_cursor).copied() else {
                    return Err(MultiproofError::InsufficientDecommitments {
                        expected_at_least: decommitment_cursor + 1,
                        got: decommitments.len(),
                    });
                };
                decommitment_cursor += 1;
                cursor += 1;

                add_node_hash_call();
                if node & 1 == 0 {
                    compress([hash, sibling_hash])
                } else {
                    compress([sibling_hash, hash])
                }
            };

            next_frontier.push((node >> 1, parent_hash));
        }

        next_frontier.dedup_by(|left, right| left.0 == right.0);
        frontier = next_frontier;
    }

    if decommitment_cursor != decommitments.len() {
        return Err(MultiproofError::TrailingDecommitments {
            consumed: decommitment_cursor,
            total: decommitments.len(),
        });
    }

    if frontier.len() != 1 || frontier[0].0 != 0 {
        return Err(MultiproofError::InvalidFinalLayer {
            layer_size: frontier.len(),
            index: frontier.first().map_or(usize::MAX, |x| x.0),
        });
    }

    Ok(frontier[0].1)
}

pub fn compute_root_from_linearized_multiproof<W: Copy + Eq, C, const DIGEST_ELEMS: usize>(
    indices: &[usize],
    leaf_hashes: &[[W; DIGEST_ELEMS]],
    depth: usize,
    decommitments: &[[W; DIGEST_ELEMS]],
    mut compress: C,
) -> Result<[W; DIGEST_ELEMS], MultiproofError>
where
    C: FnMut([[W; DIGEST_ELEMS]; 2]) -> [W; DIGEST_ELEMS],
{
    ensure_sorted_unique(indices)?;
    if indices.len() != leaf_hashes.len() {
        return Err(MultiproofError::LengthMismatch {
            indices: indices.len(),
            openings: leaf_hashes.len(),
        });
    }

    let expected_decommitments = indices.len().saturating_mul(depth);
    if decommitments.len() < expected_decommitments {
        return Err(MultiproofError::InsufficientDecommitments {
            expected_at_least: expected_decommitments,
            got: decommitments.len(),
        });
    }
    if decommitments.len() > expected_decommitments {
        return Err(MultiproofError::TrailingDecommitments {
            consumed: expected_decommitments,
            total: decommitments.len(),
        });
    }

    let mut reconstructed_root = None;
    for (query, (&index, &leaf_hash)) in indices.iter().zip(leaf_hashes.iter()).enumerate() {
        let mut node = index;
        let mut digest = leaf_hash;
        let path_start = query * depth;
        let path_end = path_start + depth;

        for &sibling in &decommitments[path_start..path_end] {
            add_node_hash_call();
            digest = if node & 1 == 0 {
                compress([digest, sibling])
            } else {
                compress([sibling, digest])
            };
            node >>= 1;
        }

        if let Some(expected_root) = reconstructed_root {
            if digest != expected_root {
                return Err(MultiproofError::RootMismatch { query });
            }
        } else {
            reconstructed_root = Some(digest);
        }
    }

    reconstructed_root.ok_or(MultiproofError::EmptyIndices)
}

/// Compute a base-field leaf hash in the same format used by WHIR commitments.
pub fn hash_leaf_base<F, W, H, const DIGEST_ELEMS: usize>(
    hasher: &H,
    values: &[F],
) -> [W; DIGEST_ELEMS]
where
    H: CryptographicHasher<F, [W; DIGEST_ELEMS]>,
    F: Copy,
{
    add_leaf_hash_call();
    hasher.hash_iter(values.iter().copied())
}

/// Compute an extension-field leaf hash by flattening basis coefficients.
pub fn hash_leaf_extension<F, EF, W, H, const DIGEST_ELEMS: usize>(
    hasher: &H,
    values: &[EF],
) -> [W; DIGEST_ELEMS]
where
    H: CryptographicHasher<F, [W; DIGEST_ELEMS]>,
    F: Field,
    EF: ExtensionField<F>,
{
    add_leaf_hash_call();
    hasher.hash_iter(
        values
            .iter()
            .flat_map(|el| el.as_basis_coefficients_slice().iter().copied()),
    )
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::Mmcs;
    use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
    use p3_matrix::{Dimensions, dense::RowMajorMatrix};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{
        CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
    };
    use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::SliceRandom};

    use super::{
        MultiproofError, build_linearized_multiproof_from_paths, build_multiproof_from_paths,
        compute_root_from_linearized_multiproof, compute_root_from_multiproof, hash_leaf_base,
        hash_leaf_extension,
    };

    type F = BabyBear;
    type EF = p3_field::extension::BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 8>;

    fn make_mmcs(seed: u64) -> (MyMmcs, Hash, Compress) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = Hash::new(perm.clone());
        let compress = Compress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());
        (mmcs, hash, compress)
    }

    fn random_matrix(seed: u64, height: usize, width: usize) -> RowMajorMatrix<F> {
        let mut rng = SmallRng::seed_from_u64(seed);
        RowMajorMatrix::new(
            (0..height * width)
                .map(|_| F::from_u32(rng.random()))
                .collect(),
            width,
        )
    }

    fn leaf_hashes(hash: &Hash, opened_rows: &[Vec<F>]) -> Vec<[F; 8]> {
        opened_rows
            .iter()
            .map(|row| hash_leaf_base::<F, F, Hash, 8>(hash, row))
            .collect()
    }

    #[test]
    fn single_query_matches_single_path() {
        let height = 16;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(1);
        let matrix = random_matrix(2, height, width);
        let (commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let index = 3usize;
        let opening = mmcs.open_batch(index, &prover_data);
        let opened_row = opening.opened_values[0].clone();
        let path = opening.opening_proof.clone();

        let multiproof = build_multiproof_from_paths(&[index], vec![path.clone()]).unwrap();
        assert_eq!(multiproof.decommitments, path);

        let root = compute_root_from_multiproof(
            &[index],
            &leaf_hashes(&hash, &[opened_row]),
            multiproof.decommitments.len(),
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        )
        .unwrap();
        let expected_root: [F; 8] = commit.into();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn adjacent_queries_share_nodes() {
        let height = 16;
        let width = 4;
        let (mmcs, _hash, _compress) = make_mmcs(11);
        let matrix = random_matrix(12, height, width);
        let (_commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [4usize, 5usize];
        let mut paths = Vec::new();
        let mut total_single = 0usize;
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            let path = opening.opening_proof;
            total_single += path.len();
            paths.push(path);
        }

        let multiproof = build_multiproof_from_paths(&indices, paths).unwrap();
        assert!(multiproof.decommitments.len() < total_single);
    }

    #[test]
    fn full_opening_has_no_decommitments() {
        let height = 8;
        let width = 2;
        let (mmcs, _hash, _compress) = make_mmcs(21);
        let matrix = random_matrix(22, height, width);
        let (_commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices: Vec<_> = (0..height).collect();
        let mut paths = Vec::with_capacity(indices.len());
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            paths.push(opening.opening_proof);
        }

        let multiproof = build_multiproof_from_paths(&indices, paths).unwrap();
        assert!(multiproof.decommitments.is_empty());
    }

    #[test]
    fn tampered_decommitment_fails() {
        let height = 16;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(31);
        let matrix = random_matrix(32, height, width);
        let (commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [2usize, 7usize, 9usize];
        let mut opened_rows = Vec::new();
        let mut paths = Vec::new();
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            opened_rows.push(opening.opened_values[0].clone());
            let path = opening.opening_proof;
            paths.push(path);
        }

        let mut multiproof = build_multiproof_from_paths(&indices, paths).unwrap();
        assert!(!multiproof.decommitments.is_empty());
        multiproof.decommitments[0][0] += F::ONE;

        let root = compute_root_from_multiproof(
            &indices,
            &leaf_hashes(&hash, &opened_rows),
            4,
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        )
        .unwrap();
        let expected_root: [F; 8] = commit.into();
        assert_ne!(root, expected_root);
    }

    #[test]
    fn randomized_cross_check_against_single_openings() {
        let mut rng = SmallRng::seed_from_u64(41);
        for round in 0..20 {
            let log_height = rng.random_range(3..=7);
            let height = 1usize << log_height;
            let width = rng.random_range(1..=6);
            let (mmcs, hash, compress) = make_mmcs(1_000 + round);
            let matrix = random_matrix(2_000 + round, height, width);
            let (commit, prover_data) = mmcs.commit_matrix(matrix.clone());

            let sample_size = rng.random_range(1..=height);
            let mut indices: Vec<_> = (0..height).collect();
            indices.shuffle(&mut rng);
            indices.truncate(sample_size);
            indices.sort_unstable();
            indices.dedup();

            let mut opened_rows = Vec::new();
            let mut paths: Vec<Vec<[F; 8]>> = Vec::new();
            for &index in &indices {
                let opening = mmcs.open_batch(index, &prover_data);
                mmcs.verify_batch(
                    &commit,
                    &[Dimensions { height, width }],
                    index,
                    (&opening).into(),
                )
                .unwrap();
                opened_rows.push(opening.opened_values[0].clone());
                paths.push(opening.opening_proof);
            }

            let depth = paths[0].len();
            let multiproof = build_multiproof_from_paths(&indices, paths).unwrap();
            let root = compute_root_from_multiproof(
                &indices,
                &leaf_hashes(&hash, &opened_rows),
                depth,
                &multiproof.decommitments,
                |pair| compress.compress(pair),
            )
            .unwrap();
            let expected_root: [F; 8] = commit.into();
            assert_eq!(root, expected_root);
        }
    }

    #[test]
    fn trailing_decommitments_are_rejected() {
        let height = 16;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(91);
        let matrix = random_matrix(92, height, width);
        let (_commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [1usize, 6usize];
        let mut opened_rows = Vec::new();
        let mut paths = Vec::new();
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            opened_rows.push(opening.opened_values[0].clone());
            paths.push(opening.opening_proof);
        }

        let mut multiproof = build_multiproof_from_paths(&indices, paths).unwrap();
        multiproof.decommitments.push([F::ZERO; 8]);

        let err = compute_root_from_multiproof(
            &indices,
            &leaf_hashes(&hash, &opened_rows),
            4,
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        )
        .unwrap_err();
        assert!(matches!(err, MultiproofError::TrailingDecommitments { .. }));
    }

    #[test]
    fn mismatched_index_and_leaf_lengths_are_rejected() {
        let (_mmcs, _hash, compress) = make_mmcs(101);
        let indices = [0usize, 1usize];
        let leaf_hashes = vec![[F::ZERO; 8]];
        let err =
            compute_root_from_multiproof(&indices, &leaf_hashes, 1, &[[F::ZERO; 8]], |pair| {
                compress.compress(pair)
            })
            .unwrap_err();
        assert!(matches!(err, MultiproofError::LengthMismatch { .. }));
    }

    #[test]
    fn extension_leaf_hash_helper_matches_manual_flattening() {
        let (_mmcs, hash, _compress) = make_mmcs(111);
        let values = vec![EF::new([
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ])];
        let helper = hash_leaf_extension::<F, EF, F, Hash, 8>(&hash, &values);
        let manual = hash.hash_iter(
            values
                .iter()
                .flat_map(|el: &EF| el.as_basis_coefficients_slice().iter().copied()),
        );
        assert_eq!(helper, manual);
    }

    // --- Linearized multiproof tests ---

    #[test]
    fn linearized_single_query_roundtrip() {
        let height = 16;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(201);
        let matrix = random_matrix(202, height, width);
        let (commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let index = 5usize;
        let opening = mmcs.open_batch(index, &prover_data);
        let opened_row = opening.opened_values[0].clone();
        let path = opening.opening_proof.clone();
        let depth = path.len();

        let multiproof =
            build_linearized_multiproof_from_paths(&[index], vec![path.clone()]).unwrap();
        // Single query: linearized decommitments == raw path
        assert_eq!(multiproof.decommitments, path);

        let root = compute_root_from_linearized_multiproof(
            &[index],
            &leaf_hashes(&hash, &[opened_row]),
            depth,
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        )
        .unwrap();
        let expected_root: [F; 8] = commit.into();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn linearized_multiple_queries_roundtrip() {
        let height = 32;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(211);
        let matrix = random_matrix(212, height, width);
        let (commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [1usize, 7, 14, 25];
        let mut opened_rows = Vec::new();
        let mut paths = Vec::new();
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            opened_rows.push(opening.opened_values[0].clone());
            paths.push(opening.opening_proof);
        }
        let depth = paths[0].len();

        let multiproof = build_linearized_multiproof_from_paths(&indices, paths).unwrap();
        // Linearized: nq * depth decommitments (no dedup)
        assert_eq!(multiproof.decommitments.len(), indices.len() * depth);

        let root = compute_root_from_linearized_multiproof(
            &indices,
            &leaf_hashes(&hash, &opened_rows),
            depth,
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        )
        .unwrap();
        let expected_root: [F; 8] = commit.into();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn linearized_adjacent_queries_no_dedup() {
        let height = 16;
        let width = 4;
        let (mmcs, _hash, _compress) = make_mmcs(221);
        let matrix = random_matrix(222, height, width);
        let (_commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [4usize, 5usize];
        let mut paths = Vec::new();
        let mut total_single = 0usize;
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            total_single += opening.opening_proof.len();
            paths.push(opening.opening_proof);
        }

        let multiproof = build_linearized_multiproof_from_paths(&indices, paths).unwrap();
        // Linearized keeps all sibling hashes — no dedup unlike frontier
        assert_eq!(multiproof.decommitments.len(), total_single);
    }

    #[test]
    fn linearized_tampered_decommitment_fails() {
        let height = 16;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(231);
        let matrix = random_matrix(232, height, width);
        let (commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [2usize, 9usize];
        let mut opened_rows = Vec::new();
        let mut paths = Vec::new();
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            opened_rows.push(opening.opened_values[0].clone());
            paths.push(opening.opening_proof);
        }
        let depth = paths[0].len();

        let mut multiproof = build_linearized_multiproof_from_paths(&indices, paths).unwrap();
        // Tamper with first query's first sibling
        multiproof.decommitments[0][0] += F::ONE;

        let result = compute_root_from_linearized_multiproof(
            &indices,
            &leaf_hashes(&hash, &opened_rows),
            depth,
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        );
        let expected_root: [F; 8] = commit.into();
        // Tampering the first query makes it compute a wrong root.
        // The second query then either produces a RootMismatch error
        // (cross-check against first query's wrong root), or if there's
        // only one query, we get the wrong root back silently.
        match result {
            Err(MultiproofError::RootMismatch { .. }) => {} // detected via cross-check
            Ok(root) => assert_ne!(root, expected_root),    // wrong root returned
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn linearized_tampered_second_query_returns_root_mismatch() {
        let height = 16;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(241);
        let matrix = random_matrix(242, height, width);
        let (_commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [3usize, 11usize];
        let mut opened_rows = Vec::new();
        let mut paths = Vec::new();
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            opened_rows.push(opening.opened_values[0].clone());
            paths.push(opening.opening_proof);
        }
        let depth = paths[0].len();

        let mut multiproof = build_linearized_multiproof_from_paths(&indices, paths).unwrap();
        // Tamper with second query's first sibling (offset = depth)
        multiproof.decommitments[depth][0] += F::ONE;

        let err = compute_root_from_linearized_multiproof(
            &indices,
            &leaf_hashes(&hash, &opened_rows),
            depth,
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        )
        .unwrap_err();
        assert!(matches!(err, MultiproofError::RootMismatch { query: 1 }));
    }

    #[test]
    fn linearized_trailing_decommitments_rejected() {
        let height = 16;
        let width = 4;
        let (mmcs, hash, compress) = make_mmcs(251);
        let matrix = random_matrix(252, height, width);
        let (_commit, prover_data) = mmcs.commit_matrix(matrix.clone());

        let indices = [1usize, 6usize];
        let mut opened_rows = Vec::new();
        let mut paths = Vec::new();
        for &index in &indices {
            let opening = mmcs.open_batch(index, &prover_data);
            opened_rows.push(opening.opened_values[0].clone());
            paths.push(opening.opening_proof);
        }
        let depth = paths[0].len();

        let mut multiproof = build_linearized_multiproof_from_paths(&indices, paths).unwrap();
        multiproof.decommitments.push([F::ZERO; 8]);

        let err = compute_root_from_linearized_multiproof(
            &indices,
            &leaf_hashes(&hash, &opened_rows),
            depth,
            &multiproof.decommitments,
            |pair| compress.compress(pair),
        )
        .unwrap_err();
        assert!(matches!(err, MultiproofError::TrailingDecommitments { .. }));
    }

    #[test]
    fn linearized_insufficient_decommitments_rejected() {
        let (_mmcs, _hash, compress) = make_mmcs(261);
        let indices = [0usize, 1usize];
        let lh = vec![[F::ZERO; 8]; 2];

        let err = compute_root_from_linearized_multiproof(
            &indices,
            &lh,
            4,                  // depth=4 → expects 2*4=8 decommitments
            &[[F::ZERO; 8]; 3], // only 3
            |pair| compress.compress(pair),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            MultiproofError::InsufficientDecommitments { .. }
        ));
    }

    #[test]
    fn linearized_mismatched_lengths_rejected() {
        let (_mmcs, _hash, compress) = make_mmcs(271);
        let indices = [0usize, 1usize];
        let lh = vec![[F::ZERO; 8]]; // 1 leaf hash, 2 indices

        let err =
            compute_root_from_linearized_multiproof(&indices, &lh, 1, &[[F::ZERO; 8]; 2], |pair| {
                compress.compress(pair)
            })
            .unwrap_err();
        assert!(matches!(err, MultiproofError::LengthMismatch { .. }));
    }

    #[test]
    fn linearized_inconsistent_path_lengths_rejected() {
        let indices = [0usize, 1usize];
        let paths = vec![
            vec![[F::ZERO; 8]; 4], // depth 4
            vec![[F::ZERO; 8]; 3], // depth 3 — mismatch
        ];
        let err = build_linearized_multiproof_from_paths::<F, 8>(&indices, paths).unwrap_err();
        assert!(matches!(
            err,
            MultiproofError::InconsistentPathLength { .. }
        ));
    }

    #[test]
    fn linearized_empty_indices_returns_error() {
        let (_mmcs, _hash, compress) = make_mmcs(281);
        let err = compute_root_from_linearized_multiproof::<F, _, 8>(&[], &[], 4, &[], |pair| {
            compress.compress(pair)
        })
        .unwrap_err();
        assert!(matches!(err, MultiproofError::EmptyIndices));
    }

    #[test]
    fn linearized_randomized_cross_check() {
        let mut rng = SmallRng::seed_from_u64(291);
        for round in 0..20 {
            let log_height = rng.random_range(3..=7);
            let height = 1usize << log_height;
            let width = rng.random_range(1..=6);
            let (mmcs, hash, compress) = make_mmcs(3_000 + round);
            let matrix = random_matrix(4_000 + round, height, width);
            let (commit, prover_data) = mmcs.commit_matrix(matrix.clone());

            let sample_size = rng.random_range(1..=height);
            let mut indices: Vec<_> = (0..height).collect();
            indices.shuffle(&mut rng);
            indices.truncate(sample_size);
            indices.sort_unstable();
            indices.dedup();

            let mut opened_rows = Vec::new();
            let mut paths = Vec::new();
            for &index in &indices {
                let opening = mmcs.open_batch(index, &prover_data);
                mmcs.verify_batch(
                    &commit,
                    &[Dimensions { height, width }],
                    index,
                    (&opening).into(),
                )
                .unwrap();
                opened_rows.push(opening.opened_values[0].clone());
                paths.push(opening.opening_proof);
            }

            let depth = paths[0].len();
            let multiproof = build_linearized_multiproof_from_paths(&indices, paths).unwrap();
            assert_eq!(multiproof.decommitments.len(), indices.len() * depth);

            let root = compute_root_from_linearized_multiproof(
                &indices,
                &leaf_hashes(&hash, &opened_rows),
                depth,
                &multiproof.decommitments,
                |pair| compress.compress(pair),
            )
            .unwrap();
            let expected_root: [F; 8] = commit.into();
            assert_eq!(
                root,
                expected_root,
                "round {round}: height={height} width={width} nq={}",
                indices.len()
            );
        }
    }
}

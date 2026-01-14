use alloc::{vec, vec::Vec};

use p3_field::{PackedValue, PrimeField32, PrimeField64};
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

/// EVM-friendly Merkle digest type.
pub type Digest32 = [u8; 32];

/// `keccak256(0x00 || row_bytes)` where each base field element is encoded as **4-byte BE u32**.
///
/// This is designed to be reproduced efficiently in Solidity.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeccakU32BeLeafHasher;

impl<P> CryptographicHasher<P, Digest32> for KeccakU32BeLeafHasher
where
    P: PackedValue,
    P::Value: PrimeField32,
{
    fn hash_iter<I>(&self, input: I) -> Digest32
    where
        I: IntoIterator<Item = P>,
    {
        // Preimage = 0x00 || (u32_be(coeff0) || u32_be(coeff1) || ...)
        // where each packed input contributes all its scalar lanes in order.
        let packed_items: Vec<P> = input.into_iter().collect();
        let mut lanes = 0usize;
        for packed in &packed_items {
            lanes += packed.as_slice().len();
        }

        let mut preimage = vec![0u8; 1 + 4 * lanes];
        preimage[0] = 0x00;

        let mut offset = 1;
        for packed in packed_items {
            for &x in packed.as_slice() {
                let w = x.as_canonical_u32();
                preimage[offset..offset + 4].copy_from_slice(&w.to_be_bytes());
                offset += 4;
            }
        }
        Keccak256Hash.hash_iter(preimage)
    }
}

/// `keccak256(0x01 || left32 || right32)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeccakNodeCompress;

impl PseudoCompressionFunction<Digest32, 2> for KeccakNodeCompress {
    fn compress(&self, input: [Digest32; 2]) -> Digest32 {
        let left = &input[0];
        let right = &input[1];
        let prefix = [0x01u8];
        Keccak256Hash.hash_iter_slices([&prefix[..], &left[..], &right[..]])
    }
}

/// Encode a base field element for the EVM transcript/verifier (4-byte big-endian u32).
#[inline]
pub fn encode_base_u32_be<F: PrimeField32>(x: F) -> [u8; 4] {
    x.as_canonical_u32().to_be_bytes()
}

/// Encode a prime field element for EVM when it fits in u32 (4-byte big-endian u32).
///
/// Panics if the element doesn't fit in u32.
#[inline]
pub fn encode_prime_u32_be<F: PrimeField64>(x: F) -> [u8; 4] {
    let v = x.to_unique_u64();
    assert!(u32::try_from(v).is_ok());
    (v as u32).to_be_bytes()
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn leaf_hash_is_domain_separated_and_uses_be_u32() {
        let h = KeccakU32BeLeafHasher;
        let input = [BabyBear::ONE, BabyBear::TWO];

        let w1 = BabyBear::ONE.as_canonical_u32();
        let w2 = BabyBear::TWO.as_canonical_u32();
        assert_eq!(w1, 1);
        assert_eq!(w2, 2);

        let expected_bytes = core::iter::once(0x00u8)
            .chain(w1.to_be_bytes())
            .chain(w2.to_be_bytes());

        assert_eq!(h.hash_iter(input), Keccak256Hash.hash_iter(expected_bytes));
    }

    #[test]
    fn node_compress_is_domain_separated() {
        let c = KeccakNodeCompress;
        let left = [1u8; 32];
        let right = [2u8; 32];

        let prefix = [0x01u8];
        let expected = Keccak256Hash.hash_iter_slices([&prefix[..], &left[..], &right[..]]);
        assert_eq!(c.compress([left, right]), expected);
    }
}

use p3_field::{Field, PackedValue};
use p3_challenger::CanObserve;
use p3_symmetric::Hash;

/// Map a Merkle digest **word type** `W` to its packed representation for a given field `F`.
///
/// - For the original (field-based) Merkle tree, `W = F` and `Packed = F::Packing`.
/// - For the EVM/Keccak path, `W = u8` and `Packed = u8`.
pub trait DigestWord<F: Field>: 'static + Copy + Send + Sync + Default + Eq {
    type Packed: PackedValue<Value = Self>;
}

impl<F: Field> DigestWord<F> for F {
    type Packed = F::Packing;
}

/// Select the packed leaf type `P` for the Merkle/MMCS layer, based on the digest word type `W`.
///
/// - Default (field digests): use SIMD packing (`P = F::Packing`) for performance.
/// - Keccak bytes32 digests: use scalar leaves (`P = F`) to satisfy MerkleTree's requirement
///   that `P::WIDTH == PW::WIDTH` without introducing packed-byte digest types.
pub trait LeafPacking<F: Field>: DigestWord<F> {
    type LeafPacked: PackedValue<Value = F>;
}

impl<F: Field> LeafPacking<F> for F {
    type LeafPacked = F::Packing;
}

/// How to observe a Merkle root for a given `(F, W, DIGEST_ELEMS)` choice.
///
/// We keep the **challenger observation type** as an associated type, so different `(F, W)` pairs
/// can choose what they observe without relying on specialization.
pub trait ObserveMerkleRoot<F: Field, W, const DIGEST_ELEMS: usize> {
    type Obs;

    fn observe_root<Challenger>(challenger: &mut Challenger, root: Hash<F, W, DIGEST_ELEMS>)
    where
        Challenger: CanObserve<Self::Obs>;
}

impl<F: Field, const DIGEST_ELEMS: usize> ObserveMerkleRoot<F, F, DIGEST_ELEMS> for () {
    type Obs = F;

    fn observe_root<Challenger>(challenger: &mut Challenger, root: Hash<F, F, DIGEST_ELEMS>)
    where
        Challenger: CanObserve<Self::Obs>,
    {
        challenger.observe_slice(root.as_ref());
    }
}

#[cfg(feature = "keccak")]
impl DigestWord<p3_baby_bear::BabyBear> for u8 {
    type Packed = u8;
}

#[cfg(feature = "keccak")]
impl DigestWord<p3_koala_bear::KoalaBear> for u8 {
    type Packed = u8;
}

#[cfg(feature = "keccak")]
impl LeafPacking<p3_baby_bear::BabyBear> for u8 {
    type LeafPacked = p3_baby_bear::BabyBear;
}

#[cfg(feature = "keccak")]
impl LeafPacking<p3_koala_bear::KoalaBear> for u8 {
    type LeafPacked = p3_koala_bear::KoalaBear;
}

#[cfg(feature = "keccak")]
impl<const DIGEST_ELEMS: usize> ObserveMerkleRoot<p3_baby_bear::BabyBear, u8, DIGEST_ELEMS> for () {
    type Obs = Hash<p3_baby_bear::BabyBear, u8, DIGEST_ELEMS>;

    fn observe_root<Challenger>(challenger: &mut Challenger, root: Self::Obs)
    where
        Challenger: CanObserve<Self::Obs>,
    {
        challenger.observe(root);
    }
}

#[cfg(feature = "keccak")]
impl<const DIGEST_ELEMS: usize> ObserveMerkleRoot<p3_koala_bear::KoalaBear, u8, DIGEST_ELEMS>
    for ()
{
    type Obs = Hash<p3_koala_bear::KoalaBear, u8, DIGEST_ELEMS>;

    fn observe_root<Challenger>(challenger: &mut Challenger, root: Self::Obs)
    where
        Challenger: CanObserve<Self::Obs>,
    {
        challenger.observe(root);
    }
}



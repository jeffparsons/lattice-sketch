use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash};
use std::marker::PhantomData;

use crate::{U24, U40, U48, U56, assert_mergeable, bucket_indices};

mod sealed {
    use crate::Lattice;

    pub trait PackedLattice: Lattice + Copy {
        fn byte_count(bucket_count: usize) -> usize;
        fn get(bytes: &[u8], index: usize) -> Self;
        fn set(bytes: &mut [u8], index: usize, value: Self);
    }
}

use sealed::PackedLattice;

impl PackedLattice for bool {
    #[inline]
    fn byte_count(bucket_count: usize) -> usize {
        bucket_count.div_ceil(8)
    }

    #[inline]
    fn get(bytes: &[u8], index: usize) -> Self {
        bytes[index / 8] & (1 << (index % 8)) != 0
    }

    #[inline]
    fn set(bytes: &mut [u8], index: usize, value: Self) {
        let bit = 1 << (index % 8);
        if value {
            bytes[index / 8] |= bit;
        } else {
            bytes[index / 8] &= !bit;
        }
    }
}

macro_rules! impl_packed_lattice_for_narrow_integers {
    ($($name:ident => $bytes:literal bytes in $repr:ty);* $(;)?) => {$(
        impl PackedLattice for $name {
            #[inline]
            fn byte_count(bucket_count: usize) -> usize {
                // Padding, so the last bucket can still be read as a whole `$repr`.
                bucket_count * $bytes + (size_of::<$repr>() - $bytes)
            }

            #[inline]
            fn get(bytes: &[u8], index: usize) -> Self {
                let word = <$repr>::from_le_bytes(
                    bytes[index * $bytes..][..size_of::<$repr>()].try_into().unwrap(),
                );
                $name::try_from(word & ((1 << ($bytes * 8)) - 1)).expect("masked to fit")
            }

            #[inline]
            fn set(bytes: &mut [u8], index: usize, value: Self) {
                let slot: &mut [u8; size_of::<$repr>()] =
                    (&mut bytes[index * $bytes..][..size_of::<$repr>()]).try_into().unwrap();
                let mask: $repr = (1 << ($bytes * 8)) - 1;
                let word = <$repr>::from_le_bytes(*slot) & !mask | <$repr>::from(value);
                *slot = word.to_le_bytes();
            }
        }
    )*};
}

impl_packed_lattice_for_narrow_integers! {
    U24 => 3 bytes in u32;
    U40 => 5 bytes in u64;
    U48 => 6 bytes in u64;
    U56 => 7 bytes in u64;
}

/// A sketch that stores narrow value types more densely than [`Sketch`](crate::Sketch) does.
///
/// `Sketch<K, L>` stores each bucket as a whole `L`, so a `bool` bucket takes a byte and a
/// [`U24`] bucket four. `PackedSketch` stores `bool` buckets at one bit each and [`U24`], [`U40`],
/// [`U48`] and [`U56`] buckets at three, five, six and seven bytes, so for the same memory it has
/// more buckets and therefore gives tighter bounds. Those are the only value types it supports.
/// Each operation costs a little more than on `Sketch`, because of the unpacking.
pub struct PackedSketch<K, L: PackedLattice, S = BuildHasherDefault<DefaultHasher>> {
    bytes: Vec<u8>,
    bucket_count: usize,
    buckets_per_key: usize,
    hash_builder: S,
    _key: PhantomData<fn(&K) -> L>,
}

impl<K: Hash, L: PackedLattice> PackedSketch<K, L> {
    /// Makes a new sketch with `bucket_count` buckets, each holding `initial`, in which each key
    /// maps to `buckets_per_key` of them.
    ///
    /// `initial` is the answer every query returns until something is inserted. In general this
    /// should be the lowest value you ever intend to insert.
    ///
    /// # Panics
    ///
    /// Panics if `bucket_count` or `buckets_per_key` is zero.
    pub fn new(bucket_count: usize, buckets_per_key: usize, initial: L) -> Self {
        Self::with_hasher(bucket_count, buckets_per_key, initial, Default::default())
    }
}

impl<K: Hash, L: PackedLattice, S: BuildHasher> PackedSketch<K, L, S> {
    /// Makes a new sketch like [`new`](Self::new), but hashing keys with `hash_builder` instead
    /// of the default hasher.
    ///
    /// Two sketches can only be [merged](Self::merge) if they were built with equal hash
    /// builders.
    ///
    /// # Panics
    ///
    /// Panics if `bucket_count` or `buckets_per_key` is zero.
    pub fn with_hasher(
        bucket_count: usize,
        buckets_per_key: usize,
        initial: L,
        hash_builder: S,
    ) -> Self {
        assert!(bucket_count > 0, "bucket_count must be at least 1");
        assert!(buckets_per_key > 0, "buckets_per_key must be at least 1");
        let mut bytes = vec![0; L::byte_count(bucket_count)];
        // TODO: fill whole words at once instead of setting each bucket.
        for index in 0..bucket_count {
            L::set(&mut bytes, index, initial);
        }
        PackedSketch {
            bytes,
            bucket_count,
            buckets_per_key,
            hash_builder,
            _key: PhantomData,
        }
    }

    /// Inserts `value` for `key`.
    ///
    /// Each of the key's buckets is raised to the [join](crate::Lattice::join) of its current
    /// value and `value`, so subsequent queries for `key` return a value at least `value`.
    #[inline]
    pub fn insert(&mut self, key: &K, value: &L) {
        for index in self.indices(key) {
            let bucket = L::get(&self.bytes, index);
            L::set(&mut self.bytes, index, bucket.join(value));
        }
    }

    /// Merges `other` into `self`, so that `self` afterwards answers every query with an upper
    /// bound on everything inserted into either sketch.
    ///
    /// Each bucket becomes the [join](crate::Lattice::join) of the two sketches' corresponding
    /// buckets. The result is identical to having made every insert into one sketch, in any
    /// order.
    ///
    /// # Panics
    ///
    /// Panics if the two sketches differ in bucket count, buckets per key, or hash builder.
    pub fn merge(&mut self, other: &Self)
    where
        S: PartialEq,
    {
        assert_mergeable(
            self.bucket_count,
            other.bucket_count,
            self.buckets_per_key,
            other.buckets_per_key,
            &self.hash_builder,
            &other.hash_builder,
        );
        // TODO: join whole words at once (OR for `bool`) instead of each bucket.
        for index in 0..self.bucket_count {
            let bucket = L::get(&self.bytes, index);
            let other_bucket = L::get(&other.bytes, index);
            L::set(&mut self.bytes, index, bucket.join(&other_bucket));
        }
    }

    /// Returns an upper bound on every value ever inserted for `key`.
    ///
    /// The bound is the [meet](crate::Lattice::meet) of the key's buckets. It is often the
    /// tightest possible — the [join](crate::Lattice::join) of everything inserted for `key` —
    /// but can be looser when other keys share buckets. Before anything is inserted every query
    /// returns `initial`.
    #[inline]
    pub fn query(&self, key: &K) -> L {
        let mut values = self.indices(key).map(|index| L::get(&self.bytes, index));
        let first = values.next().expect("buckets_per_key is at least 1");
        values.fold(first, |bound, value| bound.meet(&value))
    }

    #[inline]
    fn indices(&self, key: &K) -> impl Iterator<Item = usize> + use<K, L, S> {
        bucket_indices(
            self.hash_builder.hash_one(key),
            self.bucket_count,
            self.buckets_per_key,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sketch;

    #[test]
    fn bool_answers_match_plain_sketch() {
        for bucket_count in [1, 64, 100] {
            for initial in [false, true] {
                let mut plain = Sketch::new(bucket_count, 3, initial);
                let mut packed = PackedSketch::new(bucket_count, 3, initial);
                for key in 0..20u64 {
                    let value = key % 3 == 0;
                    plain.insert(&key, &value);
                    packed.insert(&key, &value);
                }
                for key in 0..400u64 {
                    assert_eq!(packed.query(&key), plain.query(&key));
                }
            }
        }
    }

    #[test]
    fn narrow_integer_answers_match_plain_sketch() {
        macro_rules! check {
            ($($name:ident: $bits:literal bits in $repr:ty);*) => {$(
                let value_for = |key: u64| {
                    let spread = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - $bits);
                    $name::try_from(spread as $repr).unwrap()
                };
                for bucket_count in [1, 64, 100] {
                    for initial in [0u16, 1000] {
                        let mut plain = Sketch::new(bucket_count, 3, $name::from(initial));
                        let mut packed = PackedSketch::new(bucket_count, 3, $name::from(initial));
                        for key in 0..200u64 {
                            plain.insert(&key, &value_for(key));
                            packed.insert(&key, &value_for(key));
                        }
                        for key in 0..400u64 {
                            assert_eq!(packed.query(&key), plain.query(&key));
                        }
                    }
                }
            )*};
        }
        check!(U24: 24 bits in u32; U40: 40 bits in u64; U48: 48 bits in u64; U56: 56 bits in u64);
    }

    #[test]
    fn merging_equals_inserting_everything_into_one_sketch() {
        let mut left = PackedSketch::new(100, 3, false);
        let mut right = PackedSketch::new(100, 3, false);
        let mut both = PackedSketch::new(100, 3, false);
        for key in 0..15u64 {
            left.insert(&key, &true);
            both.insert(&key, &true);
        }
        for key in 10..25u64 {
            right.insert(&key, &true);
            both.insert(&key, &true);
        }
        left.merge(&right);
        assert_eq!(left.bytes, both.bytes);
    }

    #[test]
    fn merging_floors_never_inserted_keys_at_the_join_of_both_initial_values() {
        let mut left = PackedSketch::new(100, 3, false);
        let right = PackedSketch::new(100, 3, true);
        left.merge(&right);
        for key in ["absent", "missing", "nowhere"] {
            assert!(left.query(&key));
        }
    }
}

use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash};
use std::marker::PhantomData;

use crate::{assert_mergeable, bucket_indices};

mod sealed {
    use crate::Lattice;

    pub trait PackedLattice: Lattice + Copy {
        fn word_count(bucket_count: usize) -> usize;
        fn get(words: &[u64], index: usize) -> Self;
        fn set(words: &mut [u64], index: usize, value: Self);
    }
}

use sealed::PackedLattice;

impl PackedLattice for bool {
    fn word_count(bucket_count: usize) -> usize {
        bucket_count.div_ceil(64)
    }

    fn get(words: &[u64], index: usize) -> Self {
        words[index / 64] & (1 << (index % 64)) != 0
    }

    fn set(words: &mut [u64], index: usize, value: Self) {
        let bit = 1 << (index % 64);
        if value {
            words[index / 64] |= bit;
        } else {
            words[index / 64] &= !bit;
        }
    }
}

pub struct PackedSketch<K, L: PackedLattice, S = BuildHasherDefault<DefaultHasher>> {
    words: Vec<u64>,
    bucket_count: usize,
    buckets_per_key: usize,
    hash_builder: S,
    _key: PhantomData<fn(&K) -> L>,
}

impl<K: Hash, L: PackedLattice> PackedSketch<K, L> {
    pub fn new(bucket_count: usize, buckets_per_key: usize, initial: L) -> Self {
        Self::with_hasher(bucket_count, buckets_per_key, initial, Default::default())
    }
}

impl<K: Hash, L: PackedLattice, S: BuildHasher> PackedSketch<K, L, S> {
    pub fn with_hasher(
        bucket_count: usize,
        buckets_per_key: usize,
        initial: L,
        hash_builder: S,
    ) -> Self {
        assert!(bucket_count > 0, "bucket_count must be at least 1");
        assert!(buckets_per_key > 0, "buckets_per_key must be at least 1");
        let mut words = vec![0; L::word_count(bucket_count)];
        // TODO: fill whole words at once instead of setting each bucket.
        for index in 0..bucket_count {
            L::set(&mut words, index, initial);
        }
        PackedSketch {
            words,
            bucket_count,
            buckets_per_key,
            hash_builder,
            _key: PhantomData,
        }
    }

    pub fn insert(&mut self, key: &K, value: &L) {
        for index in self.indices(key) {
            let bucket = L::get(&self.words, index);
            L::set(&mut self.words, index, bucket.join(value));
        }
    }

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
            let bucket = L::get(&self.words, index);
            let other_bucket = L::get(&other.words, index);
            L::set(&mut self.words, index, bucket.join(&other_bucket));
        }
    }

    pub fn query(&self, key: &K) -> L {
        let mut values = self.indices(key).map(|index| L::get(&self.words, index));
        let first = values.next().expect("buckets_per_key is at least 1");
        values.fold(first, |bound, value| bound.meet(&value))
    }

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
        assert_eq!(left.words, both.words);
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

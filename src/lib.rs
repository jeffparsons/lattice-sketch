use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash};
use std::marker::PhantomData;

#[cfg(target_has_atomic = "8")]
mod atomic;
mod narrow_integers;
mod packed;

#[cfg(target_has_atomic = "8")]
pub use atomic::AtomicSketch;
pub use narrow_integers::{OutOfRangeError, U24};
pub use packed::PackedSketch;

/// A lattice.
pub trait Lattice {
    /// The least upper bound of `self` and `other`.
    fn join(&self, other: &Self) -> Self;

    /// The greatest lower bound of `self` and `other`.
    fn meet(&self, other: &Self) -> Self;
}

macro_rules! impl_lattice_for_integers {
    ($($integer:ty),*) => {$(
        impl Lattice for $integer {
            fn join(&self, other: &Self) -> Self {
                (*self).max(*other)
            }

            fn meet(&self, other: &Self) -> Self {
                (*self).min(*other)
            }
        }
    )*};
}

impl_lattice_for_integers!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, U24
);

impl Lattice for bool {
    fn join(&self, other: &Self) -> Self {
        *self || *other
    }

    fn meet(&self, other: &Self) -> Self {
        *self && *other
    }
}

/// A lattice sketch.
///
/// Approximates inserted values from above.
///
/// Also called a _compact approximator_ by Boldi and Vigna (2003).[^boldi-vigna]
///
/// [^boldi-vigna]: Paolo Boldi and Sebastiano Vigna, "Compact Approximation of Lattice
///     Functions with Applications to Large-Alphabet Text Search", 2003.
///     <https://arxiv.org/abs/cs/0306046>
pub struct Sketch<K, L, S = BuildHasherDefault<DefaultHasher>> {
    buckets: Vec<L>,
    buckets_per_key: usize,
    hash_builder: S,
    _key: PhantomData<fn(&K)>,
}

impl<K: Hash, L: Lattice + Clone> Sketch<K, L> {
    pub fn new(bucket_count: usize, buckets_per_key: usize, initial: L) -> Self {
        Self::with_hasher(bucket_count, buckets_per_key, initial, Default::default())
    }
}

impl<K: Hash, L: Lattice + Clone, S: BuildHasher> Sketch<K, L, S> {
    pub fn with_hasher(
        bucket_count: usize,
        buckets_per_key: usize,
        initial: L,
        hash_builder: S,
    ) -> Self {
        assert!(bucket_count > 0, "bucket_count must be at least 1");
        assert!(buckets_per_key > 0, "buckets_per_key must be at least 1");
        Sketch {
            buckets: vec![initial; bucket_count],
            buckets_per_key,
            hash_builder,
            _key: PhantomData,
        }
    }

    pub fn insert(&mut self, key: &K, value: &L) {
        for index in self.indices(key) {
            self.buckets[index] = self.buckets[index].join(value);
        }
    }

    pub fn merge(&mut self, other: &Self)
    where
        S: PartialEq,
    {
        assert_mergeable(
            self.buckets.len(),
            other.buckets.len(),
            self.buckets_per_key,
            other.buckets_per_key,
            &self.hash_builder,
            &other.hash_builder,
        );
        for (bucket, other_bucket) in self.buckets.iter_mut().zip(&other.buckets) {
            *bucket = bucket.join(other_bucket);
        }
    }

    pub fn query(&self, key: &K) -> L {
        let mut values = self.indices(key).map(|index| &self.buckets[index]);
        let first = values
            .next()
            .expect("buckets_per_key is at least 1")
            .clone();
        values.fold(first, |bound, value| bound.meet(value))
    }

    fn indices(&self, key: &K) -> impl Iterator<Item = usize> + use<K, L, S> {
        bucket_indices(
            self.hash_builder.hash_one(key),
            self.buckets.len(),
            self.buckets_per_key,
        )
    }
}

fn assert_mergeable<S: PartialEq>(
    bucket_count: usize,
    other_bucket_count: usize,
    buckets_per_key: usize,
    other_buckets_per_key: usize,
    hash_builder: &S,
    other_hash_builder: &S,
) {
    assert_eq!(bucket_count, other_bucket_count, "bucket counts must match");
    assert_eq!(
        buckets_per_key, other_buckets_per_key,
        "buckets_per_key must match"
    );
    assert!(hash_builder == other_hash_builder, "hashers must be equal");
}

fn bucket_indices(
    key_hash: u64,
    bucket_count: usize,
    buckets_per_key: usize,
) -> impl Iterator<Item = usize> {
    // TODO: this currently uses double hashing (Kirsch & Mitzenmacher)
    // instead of computing `buckets_per_key` independent hashes.
    // Decide whether this matters or if we should document it.
    //
    // This has to be settled before we can offer backward
    // compatibility across versions of the crate.
    let bucket_count = bucket_count as u64;
    let step = if bucket_count == 1 {
        0
    } else {
        1 + mix(key_hash) % (bucket_count - 1)
    };
    let mut index = key_hash % bucket_count;
    (0..buckets_per_key).map(move |_| {
        let current = index;
        index = (index + step) % bucket_count;
        current as usize
    })
}

// splitmix64's finaliser, so `step` is unrelated to `key_hash % bucket_count`.
fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_join_to_max_and_meet_to_min() {
        macro_rules! check {
            ($low:expr, $high:expr; $($integer:ty),*) => {$(
                let (low, high): ($integer, $integer) = ($low, $high);
                assert_eq!(low.join(&high), high);
                assert_eq!(high.join(&low), high);
                assert_eq!(low.meet(&high), low);
                assert_eq!(high.meet(&low), low);
            )*};
        }
        check!(3, 5; u8, u16, u32, u64, u128, usize);
        check!(-3, 2; i8, i16, i32, i64, i128, isize);
    }

    #[test]
    fn bool_joins_with_or_and_meets_with_and() {
        for left in [false, true] {
            for right in [false, true] {
                assert_eq!(left.join(&right), left || right);
                assert_eq!(left.meet(&right), left && right);
            }
        }
    }

    #[test]
    fn a_key_never_lands_entirely_in_one_bucket() {
        for bucket_count in [2, 3, 7, 64] {
            let sketch = Sketch::new(bucket_count, 3, 0u64);
            for key in 0..1000u64 {
                let indices: Vec<usize> = sketch.indices(&key).collect();
                assert!(indices.iter().any(|&index| index != indices[0]));
            }
        }
    }

    #[test]
    fn inserted_keys_are_never_under_reported() {
        // Far more keys than buckets, so collisions force over-reporting.
        let mut sketch = Sketch::new(64, 3, 0u64);
        let mut truth = std::collections::HashMap::new();
        for round in 0..3 {
            for key in 0..200u64 {
                // Arbitrary scramble, so a key's largest value might arrive in any round.
                // (Factors are the 1,000th prime and 10,000th prime.)
                let value = (key * 7919 + round * 104_729) % 1000;
                sketch.insert(&key, &value);
                let best = truth.entry(key).or_insert(value);
                *best = (*best).max(value);
            }
        }
        let mut over_reported = 0;
        for (key, value) in truth {
            let reported = sketch.query(&key);
            assert!(reported >= value);
            if reported > value {
                over_reported += 1;
            }
        }
        assert!(
            over_reported > 0,
            "test is too sparse to exercise collisions"
        );
    }

    #[test]
    fn a_custom_hasher_is_used_consistently() {
        let mut sketch = Sketch::with_hasher(64, 3, 0u64, std::hash::RandomState::new());
        sketch.insert(&"key", &7);
        assert!(sketch.query(&"key") >= 7);
    }

    #[test]
    fn keys_never_inserted_report_at_least_the_initial_value() {
        let mut sketch = Sketch::new(64, 3, 10u64);
        assert_eq!(sketch.query(&"absent"), 10);
        for key in ["a", "b", "c"] {
            sketch.insert(&key, &50);
        }
        for key in ["absent", "missing", "nowhere"] {
            assert!(sketch.query(&key) >= 10);
        }
    }

    #[test]
    fn merging_equals_inserting_everything_into_one_sketch() {
        let mut left = Sketch::new(64, 3, 0u64);
        let mut right = Sketch::new(64, 3, 0u64);
        let mut both = Sketch::new(64, 3, 0u64);
        for key in 0..150u64 {
            let value = (key * 7919) % 1000;
            left.insert(&key, &value);
            both.insert(&key, &value);
        }
        for key in 50..200u64 {
            let value = (key * 7919 + 104_729) % 1000;
            right.insert(&key, &value);
            both.insert(&key, &value);
        }
        left.merge(&right);
        assert_eq!(left.buckets, both.buckets);
    }

    #[test]
    fn merging_floors_never_inserted_keys_at_the_join_of_both_initial_values() {
        let mut left = Sketch::new(64, 3, 10u64);
        let right = Sketch::new(64, 3, 20u64);
        left.merge(&right);
        for key in ["absent", "missing", "nowhere"] {
            assert_eq!(left.query(&key), 20);
        }
    }

    #[test]
    #[should_panic(expected = "bucket counts must match")]
    fn merging_different_bucket_counts_panics() {
        let mut sketch = Sketch::<u64, u64>::new(64, 3, 0);
        sketch.merge(&Sketch::new(32, 3, 0));
    }

    #[test]
    #[should_panic(expected = "buckets_per_key must match")]
    fn merging_different_buckets_per_key_panics() {
        let mut sketch = Sketch::<u64, u64>::new(64, 3, 0);
        sketch.merge(&Sketch::new(64, 4, 0));
    }

    #[test]
    #[should_panic(expected = "hashers must be equal")]
    fn merging_different_hashers_panics() {
        #[derive(PartialEq)]
        struct SeededHasher(u64);

        impl BuildHasher for SeededHasher {
            type Hasher = DefaultHasher;

            fn build_hasher(&self) -> DefaultHasher {
                let mut hasher = DefaultHasher::new();
                std::hash::Hasher::write_u64(&mut hasher, self.0);
                hasher
            }
        }

        let mut sketch = Sketch::<u64, u64, _>::with_hasher(64, 3, 0, SeededHasher(1));
        sketch.merge(&Sketch::with_hasher(64, 3, 0, SeededHasher(2)));
    }
}

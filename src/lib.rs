use std::hash::{DefaultHasher, Hash, Hasher};
use std::marker::PhantomData;

/// A lattice.
pub trait Lattice {
    /// The least upper bound of `self` and `other`.
    fn join(&self, other: &Self) -> Self;

    /// The greatest lower bound of `self` and `other`.
    fn meet(&self, other: &Self) -> Self;
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
pub struct Sketch<K, L> {
    buckets: Vec<L>,
    buckets_per_key: usize,
    _key: PhantomData<fn(&K)>,
}

impl<K: Hash, L: Lattice + Clone> Sketch<K, L> {
    pub fn new(bucket_count: usize, buckets_per_key: usize, initial: L) -> Self {
        assert!(bucket_count > 0, "bucket_count must be at least 1");
        assert!(buckets_per_key > 0, "buckets_per_key must be at least 1");
        Sketch {
            buckets: vec![initial; bucket_count],
            buckets_per_key,
            _key: PhantomData,
        }
    }

    pub fn insert(&mut self, key: &K, value: &L) {
        for index in self.indices(key) {
            self.buckets[index] = self.buckets[index].join(value);
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

    fn indices(&self, key: &K) -> impl Iterator<Item = usize> + use<K, L> {
        // TODO: this currently uses double hashing (Kirsch & Mitzenmacher)
        // instead of computing `buckets_per_key` independent hashes.
        // Decide whether this matters or if we should document it.
        //
        // This has to be settled before we can offer backward
        // compatibility across versions of the crate.
        // (And we need to make the hasher configurable.)
        let start = hash(key);
        let step = hash(&start);
        let bucket_count = self.buckets.len() as u64;
        (0..self.buckets_per_key as u64).map(move |probe| {
            (start.wrapping_add(probe.wrapping_mul(step)) % bucket_count) as usize
        })
    }
}

fn hash<T: Hash + ?Sized>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `u64` under its usual order, which forms a lattice: join is `max`, meet is `min`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Max(u64);

    impl Lattice for Max {
        fn join(&self, other: &Self) -> Self {
            Max(self.0.max(other.0))
        }

        fn meet(&self, other: &Self) -> Self {
            Max(self.0.min(other.0))
        }
    }

    #[test]
    fn inserted_keys_are_never_under_reported() {
        // Far more keys than buckets, so collisions force over-reporting.
        let mut sketch = Sketch::new(64, 3, Max(0));
        let mut truth = std::collections::HashMap::new();
        for round in 0..3 {
            for key in 0..200u64 {
                // Arbitrary scramble, so a key's largest value might arrive in any round.
                // (Factors are the 1,000th prime and 10,000th prime.)
                let value = Max((key * 7919 + round * 104_729) % 1000);
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
    fn keys_never_inserted_report_at_least_the_initial_value() {
        let mut sketch = Sketch::new(64, 3, Max(10));
        assert_eq!(sketch.query(&"absent"), Max(10));
        for key in ["a", "b", "c"] {
            sketch.insert(&key, &Max(50));
        }
        for key in ["absent", "missing", "nowhere"] {
            assert!(sketch.query(&key) >= Max(10));
        }
    }
}

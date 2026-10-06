use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::bucket_indices;

mod sealed {
    use crate::Lattice;

    pub trait AtomicLattice: Lattice + Copy + PartialEq {
        type Atomic;

        fn new_atomic(value: Self) -> Self::Atomic;
        fn get_mut(atomic: &mut Self::Atomic) -> &mut Self;
        fn load(atomic: &Self::Atomic) -> Self;
        fn fetch_join(atomic: &Self::Atomic, value: Self);
    }
}

use sealed::AtomicLattice;

macro_rules! impl_atomic_lattice_for_integers {
    ($($integer:ty => $atomic:ident, $width:literal);* $(;)?) => {$(
        #[cfg(target_has_atomic = $width)]
        impl AtomicLattice for $integer {
            type Atomic = std::sync::atomic::$atomic;

            fn new_atomic(value: Self) -> Self::Atomic {
                Self::Atomic::new(value)
            }

            fn get_mut(atomic: &mut Self::Atomic) -> &mut Self {
                atomic.get_mut()
            }

            fn load(atomic: &Self::Atomic) -> Self {
                atomic.load(Ordering::Relaxed)
            }

            fn fetch_join(atomic: &Self::Atomic, value: Self) {
                atomic.fetch_max(value, Ordering::Relaxed);
            }
        }
    )*};
}

impl_atomic_lattice_for_integers!(
    u8 => AtomicU8, "8";
    u16 => AtomicU16, "16";
    u32 => AtomicU32, "32";
    u64 => AtomicU64, "64";
    usize => AtomicUsize, "ptr";
    i8 => AtomicI8, "8";
    i16 => AtomicI16, "16";
    i32 => AtomicI32, "32";
    i64 => AtomicI64, "64";
    isize => AtomicIsize, "ptr";
);

impl AtomicLattice for bool {
    type Atomic = AtomicBool;

    fn new_atomic(value: Self) -> AtomicBool {
        AtomicBool::new(value)
    }

    fn get_mut(atomic: &mut AtomicBool) -> &mut Self {
        atomic.get_mut()
    }

    fn load(atomic: &AtomicBool) -> Self {
        atomic.load(Ordering::Relaxed)
    }

    fn fetch_join(atomic: &AtomicBool, value: Self) {
        atomic.fetch_or(value, Ordering::Relaxed);
    }
}

pub struct AtomicSketch<K, L: AtomicLattice, S = BuildHasherDefault<DefaultHasher>> {
    buckets: Vec<L::Atomic>,
    buckets_per_key: usize,
    hash_builder: S,
    _key: PhantomData<fn(&K)>,
}

impl<K: Hash, L: AtomicLattice> AtomicSketch<K, L> {
    pub fn new(bucket_count: usize, buckets_per_key: usize, initial: L) -> Self {
        Self::with_hasher(bucket_count, buckets_per_key, initial, Default::default())
    }
}

impl<K: Hash, L: AtomicLattice, S: BuildHasher> AtomicSketch<K, L, S> {
    pub fn with_hasher(
        bucket_count: usize,
        buckets_per_key: usize,
        initial: L,
        hash_builder: S,
    ) -> Self {
        assert!(bucket_count > 0, "bucket_count must be at least 1");
        assert!(buckets_per_key > 0, "buckets_per_key must be at least 1");
        AtomicSketch {
            buckets: (0..bucket_count).map(|_| L::new_atomic(initial)).collect(),
            buckets_per_key,
            hash_builder,
            _key: PhantomData,
        }
    }

    pub fn insert(&mut self, key: &K, value: &L) {
        for index in self.indices(key) {
            let bucket = L::get_mut(&mut self.buckets[index]);
            *bucket = bucket.join(value);
        }
    }

    pub fn insert_shared(&self, key: &K, value: &L) {
        for index in self.indices(key) {
            let bucket = &self.buckets[index];
            let current = L::load(bucket);
            if current.join(value) != current {
                L::fetch_join(bucket, *value);
            }
        }
    }

    pub fn query(&self, key: &K) -> L {
        let mut values = self.indices(key).map(|index| L::load(&self.buckets[index]));
        let first = values.next().expect("buckets_per_key is at least 1");
        values.fold(first, |bound, value| bound.meet(&value))
    }

    fn indices(&self, key: &K) -> impl Iterator<Item = usize> + use<K, L, S> {
        bucket_indices(
            self.hash_builder.hash_one(key),
            self.buckets.len(),
            self.buckets_per_key,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sketch;
    use std::collections::HashMap;

    #[test]
    fn inserted_keys_are_never_under_reported() {
        for shared in [false, true] {
            // Far more keys than buckets, so collisions force over-reporting.
            let mut sketch = AtomicSketch::new(64, 3, 0u64);
            let mut truth = HashMap::new();
            for round in 0..3 {
                for key in 0..200u64 {
                    let value = (key * 7919 + round * 104_729) % 1000;
                    if shared {
                        sketch.insert_shared(&key, &value);
                    } else {
                        sketch.insert(&key, &value);
                    }
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
    }

    #[test]
    fn keys_never_inserted_report_at_least_the_initial_value() {
        let sketch = AtomicSketch::new(64, 3, 10u64);
        assert_eq!(sketch.query(&"absent"), 10);
        for key in ["a", "b", "c"] {
            sketch.insert_shared(&key, &50);
        }
        for key in ["absent", "missing", "nowhere"] {
            assert!(sketch.query(&key) >= 10);
        }
    }

    #[test]
    fn answers_match_plain_sketch() {
        let mut plain = Sketch::new(64, 3, 0u64);
        let mut exclusive = AtomicSketch::new(64, 3, 0u64);
        let shared = AtomicSketch::new(64, 3, 0u64);
        for key in 0..200u64 {
            let value = (key * 7919) % 1000;
            plain.insert(&key, &value);
            exclusive.insert(&key, &value);
            shared.insert_shared(&key, &value);
        }
        for key in 0..400u64 {
            assert_eq!(exclusive.query(&key), plain.query(&key));
            assert_eq!(shared.query(&key), plain.query(&key));
        }
    }

    #[test]
    fn integer_answers_match_plain_sketch() {
        macro_rules! check {
            ($($integer:ty),*) => {$(
                let mut plain = Sketch::new(64, 3, <$integer>::MIN);
                let mut exclusive = AtomicSketch::new(64, 3, <$integer>::MIN);
                let shared = AtomicSketch::new(64, 3, <$integer>::MIN);
                for key in 0..200u64 {
                    let value = ((key * 7919 % 200) as i16 - 100) as $integer;
                    plain.insert(&key, &value);
                    exclusive.insert(&key, &value);
                    shared.insert_shared(&key, &value);
                }
                for key in 0..400u64 {
                    assert_eq!(exclusive.query(&key), plain.query(&key));
                    assert_eq!(shared.query(&key), plain.query(&key));
                }
            )*};
        }
        check!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
    }

    #[test]
    fn bool_answers_match_plain_sketch() {
        let mut plain = Sketch::new(64, 3, false);
        let mut exclusive = AtomicSketch::new(64, 3, false);
        let shared = AtomicSketch::new(64, 3, false);
        for key in 0..20u64 {
            let value = key % 3 == 0;
            plain.insert(&key, &value);
            exclusive.insert(&key, &value);
            shared.insert_shared(&key, &value);
        }
        for key in 0..400u64 {
            assert_eq!(exclusive.query(&key), plain.query(&key));
            assert_eq!(shared.query(&key), plain.query(&key));
        }
    }

    #[test]
    fn inserts_from_several_threads_are_never_under_reported() {
        let sketch = AtomicSketch::new(64, 3, 0u64);
        let value_for = |thread: u64, key: u64| (key * 7919 + thread * 104_729) % 1000;
        std::thread::scope(|scope| {
            for thread in 0..4 {
                let sketch = &sketch;
                scope.spawn(move || {
                    for key in 0..200u64 {
                        sketch.insert_shared(&key, &value_for(thread, key));
                    }
                });
            }
        });
        for key in 0..200u64 {
            let truth = (0..4).map(|thread| value_for(thread, key)).max().unwrap();
            assert!(sketch.query(&key) >= truth);
        }
    }
}

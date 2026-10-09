use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hash};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{assert_mergeable, bucket_indices};

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

            #[inline]
            fn new_atomic(value: Self) -> Self::Atomic {
                Self::Atomic::new(value)
            }

            #[inline]
            fn get_mut(atomic: &mut Self::Atomic) -> &mut Self {
                atomic.get_mut()
            }

            #[inline]
            fn load(atomic: &Self::Atomic) -> Self {
                atomic.load(Ordering::Relaxed)
            }

            #[inline]
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

    #[inline]
    fn new_atomic(value: Self) -> AtomicBool {
        AtomicBool::new(value)
    }

    #[inline]
    fn get_mut(atomic: &mut AtomicBool) -> &mut Self {
        atomic.get_mut()
    }

    #[inline]
    fn load(atomic: &AtomicBool) -> Self {
        atomic.load(Ordering::Relaxed)
    }

    #[inline]
    fn fetch_join(atomic: &AtomicBool, value: Self) {
        atomic.fetch_or(value, Ordering::Relaxed);
    }
}

/// A sketch that can be updated by concurrent writers without locking.
///
/// [`insert_shared`](Self::insert_shared) and [`merge_shared`](Self::merge_shared) take `&self`
/// and may be called from any number of threads at once. Queries need synchronisation to
/// guarantee visibility of updates from other threads; see [`query`](Self::query).
///
/// The value type must be `bool` or an integer type with an atomic counterpart on the target,
/// so `u128`, `i128` and the narrow integer types are excluded.
///
/// Each bucket is a [`std::sync::atomic`] type. [`insert`](Self::insert) and [`merge`](Self::merge)
/// take `&mut self` and use plain accesses to the destination buckets.
///
/// # Example
///
/// Record events from two threads, then query after both have finished:
///
/// ```
/// # #[cfg(target_has_atomic = "64")]
/// # {
/// use lattice_sketch::AtomicSketch;
/// use std::thread;
///
/// let latest_event = AtomicSketch::new(1024, 3, 0u64);
/// thread::scope(|scope| {
///     scope.spawn(|| latest_event.insert_shared(&"sensor-1", &1_700_000_060));
///     scope.spawn(|| latest_event.insert_shared(&"sensor-2", &1_700_000_120));
/// });
///
/// assert!(latest_event.query(&"sensor-1") >= 1_700_000_060);
/// assert!(latest_event.query(&"sensor-2") >= 1_700_000_120);
/// # }
/// ```
///
/// The scope joins both threads before the queries, so both inserts are included.
/// If your application can tolerate queries missing updates from other threads, you can query
/// without explicit synchronisation. In that case, a result below a cutoff does not rule out
/// an event inserted by another thread.
pub struct AtomicSketch<K, L: AtomicLattice, S = BuildHasherDefault<DefaultHasher>> {
    buckets: Vec<L::Atomic>,
    buckets_per_key: usize,
    hash_builder: S,
    _key: PhantomData<fn(&K)>,
}

impl<K: Hash, L: AtomicLattice> AtomicSketch<K, L> {
    /// Creates `bucket_count` buckets, each holding `initial`. Each insert and query makes
    /// `buckets_per_key` bucket accesses.
    ///
    /// Every query on a newly created sketch returns `initial`. It remains a lower bound on
    /// every answer afterwards. Choose a lower bound on all values you intend to insert: for
    /// example, `false` for booleans or the minimum value for integers. Starting with `10` and
    /// inserting `3` still gives an answer of at least `10`, even without collisions.
    ///
    /// # Panics
    ///
    /// Panics if `bucket_count` or `buckets_per_key` is zero.
    pub fn new(bucket_count: usize, buckets_per_key: usize, initial: L) -> Self {
        Self::with_hasher(bucket_count, buckets_per_key, initial, Default::default())
    }
}

impl<K: Hash, L: AtomicLattice, S: BuildHasher> AtomicSketch<K, L, S> {
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
        AtomicSketch {
            buckets: (0..bucket_count).map(|_| L::new_atomic(initial)).collect(),
            buckets_per_key,
            hash_builder,
            _key: PhantomData,
        }
    }

    /// Inserts `value` for `key`.
    ///
    /// Each of the key's buckets is raised to the [join](crate::Lattice::join) of its current value and
    /// `value`, so subsequent queries for `key` return a value at least `value`. Requires
    /// exclusive access; see [`insert_shared`](Self::insert_shared) to insert through a shared
    /// reference.
    #[inline]
    pub fn insert(&mut self, key: &K, value: &L) {
        for index in self.indices(key) {
            let bucket = L::get_mut(&mut self.buckets[index]);
            *bucket = bucket.join(value);
        }
    }

    /// Inserts `value` for `key` through a shared reference.
    ///
    /// Inserts completed before a query begins on the same thread are included. To include an
    /// insert made on another thread, synchronise with that thread after the insert and before
    /// the query, such as by receiving a channel message, acquiring a lock, or joining the thread.
    /// This must establish a happens-before relationship from the insert's completion to the query.
    ///
    /// Bucket updates use atomic read-modify-write operations when needed, with no
    /// synchronisation across buckets. A concurrent [`query`](Self::query) may therefore observe
    /// some of an insert's bucket updates but not others. Buckets only move up, so a query for
    /// `key` returns an upper bound on each value inserted for that key whose updates it
    /// observed in full. An insert that is only partly visible may exceed the query's result.
    ///
    /// Bucket updates use [`Relaxed`](Ordering::Relaxed) ordering.
    #[inline]
    pub fn insert_shared(&self, key: &K, value: &L) {
        for index in self.indices(key) {
            let bucket = &self.buckets[index];
            let current = L::load(bucket);
            if current.join(value) != current {
                L::fetch_join(bucket, *value);
            }
        }
    }

    /// Merges the bucket values observed in `other` into `self`.
    ///
    /// Each bucket becomes the [join](crate::Lattice::join) of its current value and the
    /// corresponding bucket read from `other`. Exclusive access is required only to `self`;
    /// `other` may be updated concurrently, so its buckets may reflect different stages of an
    /// insert or merge. Reads from `other` use [`Relaxed`](Ordering::Relaxed) ordering.
    ///
    /// If all updates to `other` happen before this merge, the result is identical to having
    /// made every insert from both sketches into one sketch, in any order, starting with the
    /// join of their initial values. Synchronise with writers to `other` to guarantee this;
    /// unsynchronised or concurrent updates may be missed or only partly included.
    ///
    /// See [`merge_shared`](Self::merge_shared) to merge through a shared reference.
    ///
    /// # Panics
    ///
    /// Panics if the two sketches differ in bucket count, buckets per key, or hash builder.
    pub fn merge(&mut self, other: &Self)
    where
        S: PartialEq,
    {
        self.assert_mergeable(other);
        for (bucket, other_bucket) in self.buckets.iter_mut().zip(&other.buckets) {
            let bucket = L::get_mut(bucket);
            *bucket = bucket.join(&L::load(other_bucket));
        }
    }

    /// Merges `other` into `self` through a shared reference.
    ///
    /// Equivalent to [`merge`](Self::merge), but each bucket is updated atomically, with the same
    /// visibility guarantees as [`insert_shared`](Self::insert_shared). Both sketches may be
    /// concurrently updated. Reads from `other` have the same visibility caveats as in
    /// `merge`; the merge does not take a consistent snapshot of its buckets.
    ///
    /// # Panics
    ///
    /// Panics if the two sketches differ in bucket count, buckets per key, or hash builder.
    pub fn merge_shared(&self, other: &Self)
    where
        S: PartialEq,
    {
        self.assert_mergeable(other);
        for (bucket, other_bucket) in self.buckets.iter().zip(&other.buckets) {
            let value = L::load(other_bucket);
            let current = L::load(bucket);
            if current.join(&value) != current {
                L::fetch_join(bucket, value);
            }
        }
    }

    /// Returns an upper bound on values inserted for `key` whose inserts happen before this
    /// query. Inserts on the same thread are included; inserts on another thread require
    /// synchronisation, such as a channel, a lock, or joining that thread.
    ///
    /// The bound is the [meet](crate::Lattice::meet) of the key's buckets. Collisions with other
    /// keys and the choice of `initial` can make the bound looser. Every answer is at least
    /// `initial`. A merge whose updates happen before this query raises that floor to the
    /// join of the sketches' initial values.
    ///
    /// Every query on a newly created sketch returns `initial`. A key never inserted can
    /// return more than `initial` because of collisions with other keys.
    ///
    /// Buckets are read with [`Relaxed`](Ordering::Relaxed) ordering. Concurrent inserts and
    /// merges may be only partially visible; the result need not include unsynchronised
    /// updates from another thread. See [`insert_shared`](Self::insert_shared) for details.
    #[inline]
    pub fn query(&self, key: &K) -> L {
        let mut values = self.indices(key).map(|index| L::load(&self.buckets[index]));
        let first = values.next().expect("buckets_per_key is at least 1");
        values.fold(first, |bound, value| bound.meet(&value))
    }

    fn assert_mergeable(&self, other: &Self)
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
    }

    #[inline]
    fn indices(&self, key: &K) -> impl Iterator<Item = usize> + use<K, L, S> {
        bucket_indices(
            self.hash_builder.hash_one(key),
            self.buckets.len(),
            self.buckets_per_key,
        )
    }
}

#[cfg(all(test, target_has_atomic = "64"))]
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
    fn merging_equals_inserting_everything_into_one_sketch() {
        for shared in [false, true] {
            let mut left = AtomicSketch::new(64, 3, 0u64);
            let right = AtomicSketch::new(64, 3, 0u64);
            let both = AtomicSketch::new(64, 3, 0u64);
            for key in 0..150u64 {
                let value = (key * 7919) % 1000;
                left.insert_shared(&key, &value);
                both.insert_shared(&key, &value);
            }
            for key in 50..200u64 {
                let value = (key * 7919 + 104_729) % 1000;
                right.insert_shared(&key, &value);
                both.insert_shared(&key, &value);
            }
            if shared {
                left.merge_shared(&right);
            } else {
                left.merge(&right);
            }
            let loaded = |sketch: &AtomicSketch<u64, u64>| -> Vec<u64> {
                sketch.buckets.iter().map(u64::load).collect()
            };
            assert_eq!(loaded(&left), loaded(&both));
        }
    }

    #[test]
    fn merging_floors_never_inserted_keys_at_the_join_of_both_initial_values() {
        for shared in [false, true] {
            let mut left = AtomicSketch::new(64, 3, 10u64);
            let right = AtomicSketch::new(64, 3, 20u64);
            if shared {
                left.merge_shared(&right);
            } else {
                left.merge(&right);
            }
            for key in ["absent", "missing", "nowhere"] {
                assert_eq!(left.query(&key), 20);
            }
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

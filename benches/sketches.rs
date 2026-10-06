use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lattice_sketch::{AtomicSketch, PackedSketch, Sketch, U24};

const BUCKETS_PER_KEY: usize = 4;
// Inserted before timing so every memory page gets written; reads from untouched pages would look cached.
const WARM_UP_KEYS: u64 = 1 << 17;

const BUDGETS: [(&str, usize); 2] = [("64 KiB", 64 << 10), ("64 MiB", 64 << 20)];

fn query_key(counter: u64) -> u64 {
    counter.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

macro_rules! bench_sketch {
    ($criterion:expr, $name:literal, $new:expr, $value:expr, [$($insert:ident),+]) => {{
        $(
            let mut group = $criterion.benchmark_group(concat!($name, "/", stringify!($insert)));
            for (label, budget) in BUDGETS {
                #[allow(unused_mut)]
                let mut sketch = $new(budget);
                for key in 0..WARM_UP_KEYS {
                    sketch.$insert(&key, &$value(key));
                }
                let mut counter = WARM_UP_KEYS;
                group.bench_function(BenchmarkId::from_parameter(label), |bencher| {
                    bencher.iter(|| {
                        counter += 1;
                        sketch.$insert(&black_box(counter), &$value(counter));
                    })
                });
            }
            group.finish();
        )+

        let mut group = $criterion.benchmark_group(concat!($name, "/query"));
        for (label, budget) in BUDGETS {
            let mut sketch = $new(budget);
            for key in 0..WARM_UP_KEYS {
                sketch.insert(&key, &$value(key));
            }
            let mut counter = 0;
            group.bench_function(BenchmarkId::from_parameter(label), |bencher| {
                bencher.iter(|| {
                    counter += 1;
                    sketch.query(&black_box(query_key(counter)))
                })
            });
        }
        group.finish();
    }};
}

fn sketches(criterion: &mut Criterion) {
    bench_sketch!(
        criterion,
        "Sketch<u32>",
        |budget: usize| Sketch::<u64, u32>::new(budget / 4, BUCKETS_PER_KEY, 0),
        |key: u64| key as u32,
        [insert]
    );
    bench_sketch!(
        criterion,
        "PackedSketch<bool>",
        |budget: usize| PackedSketch::<u64, bool>::new(budget * 8, BUCKETS_PER_KEY, false),
        |key: u64| key.is_multiple_of(2),
        [insert]
    );
    bench_sketch!(
        criterion,
        "PackedSketch<U24>",
        |budget: usize| PackedSketch::<u64, U24>::new(budget / 3, BUCKETS_PER_KEY, U24::from(0u8)),
        |key: u64| U24::from(key as u16),
        [insert]
    );
    bench_sketch!(
        criterion,
        "AtomicSketch<u32>",
        |budget: usize| AtomicSketch::<u64, u32>::new(budget / 4, BUCKETS_PER_KEY, 0),
        |key: u64| key as u32,
        [insert, insert_shared]
    );
}

criterion_group!(benches, sketches);
criterion_main!(benches);

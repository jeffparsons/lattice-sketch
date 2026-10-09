# lattice-sketch

[![Crate](https://img.shields.io/crates/v/lattice-sketch.svg)](https://crates.io/crates/lattice-sketch)
[![Docs](https://docs.rs/lattice-sketch/badge.svg)](https://docs.rs/lattice-sketch)
[![Build status](https://github.com/jeffparsons/lattice-sketch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jeffparsons/lattice-sketch/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.99%2B-blue.svg)](https://github.com/jeffparsons/lattice-sketch)

Sketches that approximate inserted values from above, generalising the Bloom filter.

Track an upper bound on the latest event time recorded for each device:

```rust
use lattice_sketch::Sketch;

// Event times are seconds since the Unix epoch, keyed by device ID.
let mut latest_event = Sketch::new(1024, 3, 0u64);
assert_eq!(latest_event.query(&"sensor-1"), 0);

latest_event.insert(&"sensor-1", &1_700_000_060);
latest_event.insert(&"sensor-1", &1_700_000_000); // An older event arrives later.
latest_event.insert(&"sensor-2", &1_700_000_120);
assert!(latest_event.query(&"sensor-1") >= 1_700_000_060);
assert!(latest_event.query(&"sensor-2") >= 1_700_000_120);
```

Insertion joins values rather than replacing them: for integers, it retains their maximum,
so the late arrival of an older event does not lower the answer. Collisions can make a device
appear more recently active than its recorded events justify, even if it has no recorded
events. The initial value remains a lower bound on every answer, so choose it below all values
you intend to insert.

The first constructor argument is the bucket count; more buckets generally reduce collisions.
The second is the number of bucket accesses per key. Increasing it costs more work and can
improve queries, but also updates more buckets per insert. Tune both for your workload and
memory budget.

`Sketch` supports any lattice value type. `AtomicSketch` supports concurrent updates, with
synchronisation needed to guarantee visibility of updates from other threads. `PackedSketch`
stores booleans and narrow integers more densely. See the
[crate documentation](https://docs.rs/lattice-sketch) for details.

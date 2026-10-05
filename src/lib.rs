/// A lattice.
pub trait Lattice {
    /// The least upper bound of `self` and `other`.
    fn join(&self, other: &Self) -> Self;

    /// The greatest lower bound of `self` and `other`.
    fn meet(&self, other: &Self) -> Self;
}

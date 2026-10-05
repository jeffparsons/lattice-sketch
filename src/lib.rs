/// A lattice with a least element.
pub trait Lattice {
    /// The least element, and the identity for [`join`](Lattice::join).
    fn bottom() -> Self;

    /// The least upper bound of `self` and `other`.
    fn join(&self, other: &Self) -> Self;

    /// The greatest lower bound of `self` and `other`.
    fn meet(&self, other: &Self) -> Self;
}

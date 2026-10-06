use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U24(u32);

impl From<u8> for U24 {
    fn from(value: u8) -> Self {
        U24(value.into())
    }
}

impl From<u16> for U24 {
    fn from(value: u16) -> Self {
        U24(value.into())
    }
}

impl TryFrom<u32> for U24 {
    type Error = OutOfRangeError;

    fn try_from(value: u32) -> Result<Self, OutOfRangeError> {
        if value < 1 << 24 {
            Ok(U24(value))
        } else {
            Err(OutOfRangeError(()))
        }
    }
}

impl From<U24> for u32 {
    fn from(value: U24) -> Self {
        value.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfRangeError(());

impl fmt::Display for OutOfRangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("value out of range for the target type")
    }
}

impl std::error::Error for OutOfRangeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u24_accepts_exactly_the_values_that_fit_in_24_bits() {
        assert_eq!(U24::try_from(0u32).map(u32::from), Ok(0));
        assert_eq!(
            U24::try_from((1u32 << 24) - 1).map(u32::from),
            Ok((1 << 24) - 1)
        );
        assert_eq!(U24::try_from(1u32 << 24), Err(OutOfRangeError(())));
        assert_eq!(U24::try_from(u32::MAX), Err(OutOfRangeError(())));
    }
}

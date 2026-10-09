use std::fmt;

macro_rules! narrow_integers {
    ($($name:ident: $bits:literal bits in $repr:ty, from [$($narrower:ty),*]);* $(;)?) => {$(
        #[doc = concat!(
            "An unsigned ", $bits, "-bit integer, stored in a `", stringify!($repr), "`.\n\n",
            "Exists so that [`PackedSketch`](crate::PackedSketch) can store a bucket in ", $bits,
            " bits rather than a whole `", stringify!($repr), "`. Forms a ",
            "[`Lattice`](crate::Lattice) under its usual ordering like the built-in integers. ",
            "\n\nConvert from `", stringify!($repr), "` with `TryFrom`, which returns ",
            "[`OutOfRangeError`] if the value is out of range. Convert back to `",
            stringify!($repr), "` with `From`.",
        )]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($repr);

        $(
            impl From<$narrower> for $name {
                #[inline]
                fn from(value: $narrower) -> Self {
                    $name(value.into())
                }
            }
        )*

        impl TryFrom<$repr> for $name {
            type Error = OutOfRangeError;

            #[inline]
            fn try_from(value: $repr) -> Result<Self, OutOfRangeError> {
                if value < 1 << $bits {
                    Ok($name(value))
                } else {
                    Err(OutOfRangeError(()))
                }
            }
        }

        impl From<$name> for $repr {
            #[inline]
            fn from(value: $name) -> Self {
                value.0
            }
        }
    )*};
}

narrow_integers! {
    U24: 24 bits in u32, from [u8, u16];
    U40: 40 bits in u64, from [u8, u16, u32];
    U48: 48 bits in u64, from [u8, u16, u32];
    U56: 56 bits in u64, from [u8, u16, u32];
}

/// An error returned when a value is out of range for [`U24`], [`U40`], [`U48`] or [`U56`].
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
    fn each_type_accepts_exactly_the_values_that_fit_in_its_bits() {
        macro_rules! check {
            ($($name:ident: $bits:literal bits in $repr:ty);*) => {$(
                let largest: $repr = (1 << $bits) - 1;
                assert_eq!($name::try_from(0 as $repr).map(<$repr>::from), Ok(0));
                assert_eq!($name::try_from(largest).map(<$repr>::from), Ok(largest));
                assert_eq!($name::try_from(largest + 1), Err(OutOfRangeError(())));
                assert_eq!($name::try_from(<$repr>::MAX), Err(OutOfRangeError(())));
            )*};
        }
        check!(U24: 24 bits in u32; U40: 40 bits in u64; U48: 48 bits in u64; U56: 56 bits in u64);
    }
}

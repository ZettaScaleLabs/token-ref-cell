//! Errors.

use core::fmt;

/// Error returned by [`TokenCell::try_borrow`](crate::TokenCell::try_borrow).
#[derive(Debug)]
pub struct BorrowError;

impl BorrowError {
    // This ensures the panicking code is outlined from inlined functions
    pub(crate) fn panic(self) -> ! {
        panic!("{self}")
    }
}

impl fmt::Display for BorrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "wrong token")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BorrowError {}

#[allow(missing_docs)]
/// Error returned by [`TokenCell::try_borrow_mut`](crate::TokenCell::try_borrow_mut).
#[derive(Debug)]
pub enum BorrowMutError {
    WrongToken,
    NotUniqueToken,
}

impl BorrowMutError {
    // This ensures the panicking code is outlined from inlined functions
    pub(crate) fn panic(self) -> ! {
        panic!("{self}")
    }
}

impl fmt::Display for BorrowMutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongToken => write!(f, "wrong token"),
            Self::NotUniqueToken => write!(f, "not unique token"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BorrowMutError {}

/// Error returned when unique token has already been initialized.
#[derive(Debug)]
pub struct AlreadyInitialized;

impl fmt::Display for AlreadyInitialized {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "already initialized")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AlreadyInitialized {}

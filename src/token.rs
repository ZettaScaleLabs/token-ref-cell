//! Implementations of [`Token`] used with [`TokenCell`](crate::TokenCell).
//!
//! The recommended token implementation is [`BoxToken`].

use core::{
    fmt,
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Token type to be used with [`TokenCell`](crate::TokenCell).
///
/// # Safety
///
/// If `Token::is_unique` returns true, then there must be no other instances of the same token
/// type with `Token::id` returning the same id as the current "unique" instance.
pub unsafe trait Token {
    /// Id of the token.
    type Id: Clone + Eq;
    /// Return the token id.
    fn id(&self) -> Self::Id;
    /// Returns true if the token is "unique", see [safety](Self#safety)
    fn is_unique(&mut self) -> bool;
}

/// Const token that can be used with [`TokenCell::new_const`](crate::TokenCell::new_const).
///
/// Tokens generated with [`singleton_token!`](crate::singleton_token) macro implement this trait.
pub trait ConstToken: Token {
    /// Constant token id.
    const ID: Self::Id;
}

/// Wrapper for pointer used as [`Token::Id`] .
///
/// The pointer is only used for comparison, so `PtrId` can implement
/// `Send`/`Sync` because the pointee is never accessed.
pub struct PtrId<T: ?Sized>(*const T);

impl<T: ?Sized> PtrId<T> {
    /// Wrap a pointer.
    #[inline]
    pub fn new(ptr: *const T) -> Self {
        Self(ptr)
    }
}

impl<T: ?Sized> From<*const T> for PtrId<T> {
    #[inline]
    fn from(value: *const T) -> Self {
        Self(value)
    }
}

impl<T: ?Sized> From<*mut T> for PtrId<T> {
    #[inline]
    fn from(value: *mut T) -> Self {
        Self(value)
    }
}

impl<T: ?Sized> From<&T> for PtrId<T> {
    #[inline]
    fn from(value: &T) -> Self {
        Self(value)
    }
}

impl<T: ?Sized> From<&mut T> for PtrId<T> {
    #[inline]
    fn from(value: &mut T) -> Self {
        Self(value)
    }
}

// SAFETY: the pointee is never accessed
unsafe impl<T: ?Sized> Send for PtrId<T> {}
// SAFETY: the pointee is never accessed
unsafe impl<T: ?Sized> Sync for PtrId<T> {}

impl<T: ?Sized> fmt::Debug for PtrId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<T: ?Sized> Copy for PtrId<T> {}
impl<T: ?Sized> Clone for PtrId<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> PartialEq for PtrId<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}
impl<T: ?Sized> Eq for PtrId<T> {}

/// Generate a singleton token type.
///
/// The generated type provides a `new` method, as well as a fallible
/// `try_new` version. These methods ensure there is only a single
/// instance of the singleton at a time; singleton can still be dropped
/// and re-instantiated.
///
/// Contrary to other provided token types, the singleton token,
/// as well as its id, are zero-sized. This makes it an (almost)
/// zero-cost abstraction, the only (negligible) cost is an atomic swap
/// at instantiation.
#[macro_export]
macro_rules! singleton_token {
    ($name:ident) => {
        $crate::singleton_token!(pub(self) struct $name);
    };
    ($(#[$($attr:meta)*])* struct $name:ident;) => {
        $crate::singleton_token!($(#[$($attr)*])* pub(self) struct $name;);
    };
    ($(#[$($attr:meta)*])* $vis:vis struct $name:ident;) => {
        $(#[$($attr)*])*
        $vis struct $name;
        const _: () = {
            static INITIALIZED: ::core::sync::atomic::AtomicBool = ::core::sync::atomic::AtomicBool::new(false);
            impl $name {
                #[inline]
                $vis fn try_new() -> Result<Self, $crate::error::AlreadyInitialized> {
                    if INITIALIZED.swap(true, ::core::sync::atomic::Ordering::Relaxed) {
                        Err($crate::error::AlreadyInitialized)
                    } else {
                        Ok(Self)
                    }
                }

                #[inline]
                $vis fn new() -> Self {
                    Self::try_new().unwrap()
                }
            }

            impl Drop for $name {
                #[inline]
                fn drop(&mut self) {
                    INITIALIZED.store(false, ::core::sync::atomic::Ordering::Relaxed);
                }
            }

            // SAFETY: Each forged token type can only have a single instance
            unsafe impl $crate::token::Token for $name {
                type Id = ();

                #[inline]
                fn id(&self) -> Self::Id {}

                #[inline]
                fn is_unique(&mut self) -> bool {
                    true
                }
            }

            impl $crate::token::ConstToken for $name {
                const ID: Self::Id = ();
            }
        };
    };
}

static DYNAMIC_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Dynamic token, equivalent to an `usize` in memory.
///
/// Each token instantiated with [`DynamicToken::new`] is unique,
/// but the number of instantiated token in all the program lifetime
/// **must** not overflow `isize::MAX`.
///
/// This constraint comes from the trivial unicity implementation,
/// an `AtomicUsize`, with no possible reuse of dropped tokens.
/// You should use smart-pointer based token instead, as they use
/// non-trivial unicity algorithm named "memory allocator".
#[derive(Debug, Eq, PartialEq)]
pub struct DynamicToken(usize);

impl DynamicToken {
    /// Create a unique token.
    #[inline]
    pub fn new() -> Self {
        let token = DYNAMIC_COUNTER.fetch_add(1, Ordering::Relaxed);
        if token > isize::MAX as usize {
            #[inline(never)]
            #[cold]
            pub(crate) fn abort() -> ! {
                #[cfg(feature = "std")]
                {
                    std::process::abort();
                }
                #[cfg(not(feature = "std"))]
                {
                    struct Abort;
                    impl Drop for Abort {
                        fn drop(&mut self) {
                            panic!();
                        }
                    }
                    let _a = Abort;
                    panic!("abort");
                }
            }
            abort();
        }
        Self(token)
    }
}

impl Default for DynamicToken {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: Each `DynamicToken` has a different inner token and cannot be cloned
unsafe impl Token for DynamicToken {
    type Id = usize;

    #[inline]
    fn id(&self) -> Self::Id {
        self.0
    }

    #[inline]
    fn is_unique(&mut self) -> bool {
        true
    }
}

/// Abstraction of an exclusive/mutable reference as a token.
///
/// The reference should point to a pinned object, otherwise moving
/// the object will "invalidate" the  cells initialized with the
/// previous reference.
#[repr(transparent)]
pub struct RefMutToken<T: ?Sized>(T);

impl<T: ?Sized> RefMutToken<T> {
    /// Convert an immutable reference into an immutable `RefMutToken` reference.
    #[inline]
    pub fn from_ref(t: &T) -> &Self {
        // SAFETY: `RefMutToken` is `repr(transparent)`
        unsafe { &*(t as *const T as *const Self) }
    }

    /// Convert a mutable reference into a mutable `RefMutToken` reference.
    #[inline]
    pub fn from_mut(t: &mut T) -> &mut Self {
        // SAFETY: `RefMutToken` is `repr(transparent)`
        unsafe { &mut *(t as *mut T as *mut Self) }
    }
}

impl<T: ?Sized> Deref for RefMutToken<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ?Sized> DerefMut for RefMutToken<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T: ?Sized> From<&'a T> for &'a RefMutToken<T> {
    #[inline]
    fn from(value: &'a T) -> Self {
        RefMutToken::from_ref(value)
    }
}

impl<'a, T: ?Sized> From<&'a mut T> for &'a mut RefMutToken<T> {
    #[inline]
    fn from(value: &'a mut T) -> Self {
        RefMutToken::from_mut(value)
    }
}

// SAFETY: token uniqueness is guaranteed by mutable reference properties
unsafe impl<T: ?Sized> Token for RefMutToken<T> {
    type Id = PtrId<T>;

    #[inline]
    fn id(&self) -> Self::Id {
        (&self.0).into()
    }

    #[inline]
    fn is_unique(&mut self) -> bool {
        true
    }
}

/// Abstraction of a pinned exclusive/mutable reference as a token.
#[repr(transparent)]
pub struct PinToken<T: ?Sized>(T);

impl<T: ?Sized> PinToken<T> {
    /// Convert an immutable reference into an immutable `RefMutToken` reference.
    #[inline]
    pub fn from_ref(t: Pin<&T>) -> &Self {
        // SAFETY: `PinToken` is `repr(transparent)`
        unsafe { &*(t.get_ref() as *const T as *const Self) }
    }

    /// Convert a mutable reference into a mutable `RefMutToken` reference.
    #[inline]
    pub fn from_mut(t: Pin<&mut T>) -> &mut Self {
        // SAFETY: mutable ref is never accessed and `RefMutToken` is `repr(transparent)`
        unsafe { &mut *(t.get_unchecked_mut() as *mut T as *mut Self) }
    }
}

impl<T: ?Sized> Deref for PinToken<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ?Sized> DerefMut for PinToken<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T: ?Sized> From<Pin<&'a T>> for &'a PinToken<T> {
    #[inline]
    fn from(value: Pin<&'a T>) -> Self {
        PinToken::from_ref(value)
    }
}

impl<'a, T: ?Sized> From<Pin<&'a mut T>> for &'a mut PinToken<T> {
    #[inline]
    fn from(value: Pin<&'a mut T>) -> Self {
        PinToken::from_mut(value)
    }
}

// SAFETY: token uniqueness is guaranteed by mutable reference properties
unsafe impl<T: ?Sized> Token for PinToken<T> {
    type Id = PtrId<T>;

    #[inline]
    fn id(&self) -> Self::Id {
        (&self.0).into()
    }

    #[inline]
    fn is_unique(&mut self) -> bool {
        true
    }
}

#[cfg(feature = "alloc")]
mod with_alloc {
    extern crate alloc;
    use alloc::{boxed::Box, rc::Rc, sync::Arc};

    use crate::token::{PtrId, Token};

    /// Dummy struct which can be used with smart-pointer-based token implementations.
    ///
    /// Smart-pointer requires a non-zero-sized type parameter to be used as token.
    /// Use it if, like me, you don't like finding names for dummy objects;
    /// `Box<AllocatedToken>` is still better than `Box<u8>`.
    #[allow(dead_code)]
    #[derive(Debug, Default)]
    pub struct AllocatedToken(u8);

    /// A wrapper for `Box<AllocatedToken>`.
    ///
    /// This is the recommended token implementation.
    #[derive(Debug, Default)]
    pub struct BoxToken(Box<AllocatedToken>);

    impl BoxToken {
        /// Allocate a `BoxToken`.
        pub fn new() -> Self {
            Self::default()
        }
    }

    // SAFETY: `BoxToken` is a wrapper around `Box` which implements `Token`
    unsafe impl Token for BoxToken {
        type Id = <Box<AllocatedToken> as Token>::Id;

        #[inline]
        fn id(&self) -> Self::Id {
            self.0.id()
        }

        #[inline]
        fn is_unique(&mut self) -> bool {
            self.0.is_unique()
        }
    }

    trait NotZeroSized<T> {
        const ASSERT_SIZE_IS_NOT_ZERO: () = assert!(size_of::<T>() > 0);
    }

    impl<T> NotZeroSized<T> for T {}

    macro_rules! check_not_zero_sized {
        ($T:ty) => {
            #[allow(path_statements)]
            {
                <$T>::ASSERT_SIZE_IS_NOT_ZERO;
            }
        };
    }

    // SAFETY: It's not possible to have simultaneously two boxes with the same pointer/token id,
    // as long as `T` is not zero-sized (ensured by a compile-time check)
    /// `T` must not be zero-sized.
    unsafe impl<T> Token for Box<T> {
        type Id = PtrId<T>;

        #[inline]
        fn id(&self) -> Self::Id {
            check_not_zero_sized!(T);
            self.as_ref().into()
        }

        #[inline]
        fn is_unique(&mut self) -> bool {
            check_not_zero_sized!(T);
            true
        }
    }

    // SAFETY: `Rc::get_mut` ensures the unicity of the `Rc` instance "owning" its inner pointer,
    // as long as `T` is not zero-sized (ensured by a compile-time check)
    /// `T` must not be zero-sized.
    unsafe impl<T> Token for Rc<T> {
        type Id = PtrId<T>;

        #[inline]
        fn id(&self) -> Self::Id {
            check_not_zero_sized!(T);
            self.as_ref().into()
        }

        #[inline]
        fn is_unique(&mut self) -> bool {
            check_not_zero_sized!(T);
            Rc::get_mut(self).is_some()
        }
    }

    // SAFETY: `Arc::get_mut` ensures the unicity of the `Rc` instance "owning" its inner pointer,
    // as long as `T` is not zero-sized (ensured by a compile-time check)
    /// `T` must not be zero-sized.
    unsafe impl<T> Token for Arc<T> {
        type Id = PtrId<T>;

        #[inline]
        fn id(&self) -> Self::Id {
            check_not_zero_sized!(T);
            self.as_ref().into()
        }

        #[inline]
        fn is_unique(&mut self) -> bool {
            check_not_zero_sized!(T);
            Arc::get_mut(self).is_some()
        }
    }
}
#[cfg(feature = "alloc")]
pub use with_alloc::*;

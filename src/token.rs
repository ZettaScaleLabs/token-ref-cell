//! Implementations of [`Token`] used with [`TokenRefCell`](crate::TokenRefCell).
//!
//! The recommended token implementation is [`BoxToken`].

use core::{
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

#[doc(hidden)]
pub use crate::repr_hack::TokenId;

/// Token type to be used with [`TokenRefCell`](crate::TokenRefCell).
///
/// It defines an `Id` type, which is stored in the cell and used to check its accesses.
///
/// `TokenId` bound is a hidden trait used to hack Rust type system in order to allow
/// `TokenRefCell` being defined with `#[repr(transparent)]` when `Id` is `()`.
/// It can be implemented using hidden `impl_token_id!` macro, for example
/// `impl_token_id!(PtrId<T: ?Sized>)`.
/// <br>
/// This bound is temporary and will be removed when Rust type system will allow
/// const expressions like `size_of::<Tk>() == 0` to be used as generic parameter.
///
/// # Safety
///
/// If [`Token::is_unique`] returns true, then there must be no other instances of the same token
/// type with [`Token::id`] returning the same id as the current "unique" instance;
/// if the token type is neither `Send` nor `Sync`, this unicity constraint is relaxed to the
/// current thread.
/// <br>
/// Token implementations can rely on the fact that [`TokenRefCell`](crate::TokenRefCell),
/// [`Ref`](crate::Ref), [`RefMut`](crate::RefMut), [`Reborrow`](crate::Reborrow),
/// and [`ReborrowMut`](crate::ReborrowMut) are invariant on their `Tk: Token` generic parameter.
pub unsafe trait Token {
    /// Id of the token.
    type Id: Clone + Eq + TokenId;
    /// Returns the token id.
    fn id(&self) -> Self::Id;
    /// Returns true if the token is "unique", see [safety](Self#safety)
    fn is_unique(&mut self) -> bool;
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

/// Generate a zero-sized singleton token type.
///
/// The generated type provides a `new` method, as well as a fallible
/// `try_new` version. These methods ensure there is only a single
/// instance of the singleton at a time; singleton can still be dropped
/// and re-instantiated.
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
                $vis fn new() -> Self {
                    match Self::try_new() {
                        Ok(tk) => tk,
                        Err(err) => err.panic(),
                    }
                }

                $vis fn try_new() -> Result<Self, $crate::error::AlreadyInitialized> {
                    if INITIALIZED.swap(true, ::core::sync::atomic::Ordering::Relaxed) {
                        Err($crate::error::AlreadyInitialized)
                    } else {
                        Ok(Self)
                    }
                }

            }

            impl Drop for $name {
                fn drop(&mut self) {
                    INITIALIZED.store(false, ::core::sync::atomic::Ordering::Relaxed);
                }
            }

            // SAFETY: forged token initialization guarantees there is only one single instance
            unsafe impl $crate::token::Token for $name {
                type Id = ();

                #[inline]
                fn id(&self) -> Self::Id {}

                #[inline]
                fn is_unique(&mut self) -> bool {
                    true
                }
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
    ///
    /// # Panics
    ///
    /// Panics when `usize::MAX` tokens has already been created.
    pub fn new() -> Self {
        let incr = |c| (c != usize::MAX).then(|| c + 1);
        match DYNAMIC_COUNTER.fetch_update(Ordering::Relaxed, Ordering::Relaxed, incr) {
            Ok(c) => Self(c + 1),
            Err(_) => panic!("No more dynamic token available"),
        }
    }
}

impl Default for DynamicToken {
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

/// Abstraction of a reference as a token.
///
/// The reference should point to a pinned object, otherwise moving the object
/// will "invalidate" the cells initialized with the previous reference.
#[derive(Debug)]
#[repr(transparent)]
pub struct RefToken<T: ?Sized>(T);

impl<T: ?Sized> RefToken<T> {
    /// Convert an immutable reference into an immutable `RefToken` reference.
    #[inline]
    pub fn from_ref(t: &T) -> &Self
    where
        T: Unpin,
    {
        // SAFETY: `RefToken` is `repr(transparent)`
        unsafe { &*(t as *const T as *const Self) }
    }

    /// Convert a mutable reference into a mutable `RefToken` reference.
    #[inline]
    pub fn from_mut(t: &mut T) -> &mut Self
    where
        T: Unpin,
    {
        // SAFETY: `RefToken` is `repr(transparent)`
        unsafe { &mut *(t as *mut T as *mut Self) }
    }

    /// Convert a pinned immutable reference into an immutable `RefToken` reference.
    #[inline]
    pub fn from_pin(t: Pin<&T>) -> &Self {
        // SAFETY: `RefToken` is `repr(transparent)`
        unsafe { &*(t.get_ref() as *const T as *const Self) }
    }

    /// Convert a pinned mutable reference into a mutable `RefToken` reference.
    #[inline]
    pub fn from_pin_mut(t: Pin<&mut T>) -> &mut Self {
        // SAFETY: mutable ref is never accessed if `T` is not `Unpin`,
        // and `RefToken` is `repr(transparent)`
        unsafe { &mut *(t.get_unchecked_mut() as *mut T as *mut Self) }
    }

    /// Get a pinned immutable reference.
    #[inline]
    pub fn as_pin(&self) -> Pin<&T> {
        // SAFETY: `RefToken` was initialized with pinned reference unless `T: Unpin`
        unsafe { Pin::new_unchecked(&self.0) }
    }

    /// Get a pinned mutable reference.
    #[inline]
    pub fn as_pin_mut(&mut self) -> Pin<&mut T> {
        // SAFETY: `PinToken` was initialized with pinned reference unless `T: Unpin`
        unsafe { Pin::new_unchecked(&mut self.0) }
    }
}

impl<T: ?Sized> Deref for RefToken<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Unpin + ?Sized> DerefMut for RefToken<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: ?Sized> AsRef<T> for RefToken<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: Unpin + ?Sized> AsMut<T> for RefToken<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<'a, T: Unpin + ?Sized> From<&'a T> for &'a RefToken<T> {
    #[inline]
    fn from(value: &'a T) -> Self {
        RefToken::from_ref(value)
    }
}

impl<'a, T: Unpin + ?Sized> From<&'a mut T> for &'a mut RefToken<T> {
    #[inline]
    fn from(value: &'a mut T) -> Self {
        RefToken::from_mut(value)
    }
}

impl<'a, T: ?Sized> From<Pin<&'a T>> for &'a RefToken<T> {
    #[inline]
    fn from(value: Pin<&'a T>) -> Self {
        RefToken::from_pin(value)
    }
}

impl<'a, T: ?Sized> From<Pin<&'a mut T>> for &'a mut RefToken<T> {
    #[inline]
    fn from(value: Pin<&'a mut T>) -> Self {
        RefToken::from_pin_mut(value)
    }
}

// SAFETY: token uniqueness is guaranteed by mutable reference properties
unsafe impl<T: ?Sized> Token for RefToken<T> {
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

#[cfg(feature = "std")]
mod with_std {
    use std::{
        any::TypeId, cell::RefCell, collections::BTreeSet, fmt, fmt::Formatter,
        marker::PhantomData, sync::Mutex,
    };

    use crate::{error::AlreadyInitialized, token::Token};

    static TYPE_TOKENS: Mutex<BTreeSet<TypeId>> = Mutex::new(BTreeSet::new());

    /// Zero-sized token implementation which use a type as unicity marker.
    pub struct TypeToken<T: ?Sized + 'static>(PhantomData<fn(T) -> T>);

    impl<T: ?Sized + 'static> fmt::Debug for TypeToken<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            f.debug_struct("TypeToken").finish()
        }
    }

    impl<T: ?Sized + 'static> TypeToken<T> {
        /// Initializes a `TypeToken`.
        pub fn new() -> Self {
            Self::try_new().unwrap()
        }

        /// Initializes `TypeToken`, fails if it has already been initialized.
        pub fn try_new() -> Result<Self, AlreadyInitialized> {
            if TYPE_TOKENS.lock().unwrap().insert(TypeId::of::<Self>()) {
                Ok(Self(PhantomData))
            } else {
                Err(AlreadyInitialized)
            }
        }
    }

    impl<T: ?Sized + 'static> Drop for TypeToken<T> {
        fn drop(&mut self) {
            TYPE_TOKENS.lock().unwrap().remove(&TypeId::of::<Self>());
        }
    }

    impl<T: ?Sized + 'static> Default for TypeToken<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    // SAFETY: `TypeToken` initialization guarantees there is only one single instance,
    // taking into account that it's invariant on `T`
    unsafe impl<T: ?Sized + 'static> Token for TypeToken<T> {
        type Id = ();

        fn id(&self) -> Self::Id {}

        fn is_unique(&mut self) -> bool {
            true
        }
    }

    std::thread_local! {
        static LOCAL_TYPE_TOKENS: RefCell<BTreeSet<TypeId>> = const { RefCell::new(BTreeSet::new()) };
    }

    /// Zero-sized thread-local token implementation which use a type as unicity marker.
    pub struct LocalTypeToken<T: ?Sized + 'static>(PhantomData<*mut T>);

    impl<T: ?Sized + 'static> fmt::Debug for LocalTypeToken<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            f.debug_struct("LocalTypeToken").finish()
        }
    }

    impl<T: ?Sized + 'static> LocalTypeToken<T> {
        /// Initializes a `LocalTypeToken`.
        pub fn new() -> Self {
            Self::try_new().unwrap()
        }

        /// Initializes a `LocalTypeToken`, fails if it has already been initialized in the thread.
        pub fn try_new() -> Result<Self, AlreadyInitialized> {
            if LOCAL_TYPE_TOKENS.with_borrow_mut(|types| types.insert(TypeId::of::<Self>())) {
                Ok(Self(PhantomData))
            } else {
                Err(AlreadyInitialized)
            }
        }
    }

    impl<T: ?Sized + 'static> Drop for LocalTypeToken<T> {
        fn drop(&mut self) {
            LOCAL_TYPE_TOKENS.with_borrow_mut(|types| types.remove(&TypeId::of::<Self>()));
        }
    }

    impl<T: ?Sized + 'static> Default for LocalTypeToken<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    // SAFETY: `LocalTypeToken` initialization guarantees there is only one single instance
    // in the thread, taking into account that it's invariant on `T`
    unsafe impl<T: ?Sized + 'static> Token for LocalTypeToken<T> {
        type Id = ();

        fn id(&self) -> Self::Id {}

        fn is_unique(&mut self) -> bool {
            true
        }
    }
}

#[cfg(feature = "std")]
pub use with_std::*;

/// Lifetime-based token implementation.
#[derive(Debug, PartialEq, Eq)]
pub struct LifetimeToken<'id>(PhantomData<fn(&'id ()) -> &'id ()>);

impl LifetimeToken<'_> {
    /// Creates a `LifetimeToken` with a unique lifetime, only valid for the scope of
    /// the provided closure.
    pub fn scope<R>(f: impl FnOnce(LifetimeToken) -> R) -> R {
        f(Self(PhantomData))
    }
}

// SAFETY: The only way to instantiate a `LifetimeToken` is using `LifetimeToken::scope`, as
// well as generativity, and both ensures that the unicity of the token by the unicity of its
// lifetime.
unsafe impl Token for LifetimeToken<'_> {
    type Id = ();

    fn id(&self) -> Self::Id {}

    fn is_unique(&mut self) -> bool {
        true
    }
}

#[cfg(feature = "generativity")]
impl<'id> From<generativity::Guard<'id>> for LifetimeToken<'id> {
    fn from(_: generativity::Guard<'id>) -> Self {
        Self(PhantomData)
    }
}

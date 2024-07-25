//! Implementations of [`Token`] used with [`TokenCell`].

use core::{
    fmt,
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    error::{BorrowError, BorrowMutError},
    Ref, RefMut, TokenCell,
};

/// Token type to be used with [`TokenCell`].
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

/// Wrapper for pointer used as [`Token::Id`] .
///
/// The pointer is only used for comparison, so `PtrId` can implement
/// `Send`/`Sync` because the pointee is never accessed.
pub struct PtrId<T: ?Sized>(*const T);

impl<T: ?Sized> PtrId<T> {
    /// Wrap a pointer.
    pub fn new(ptr: *const T) -> Self {
        Self(ptr)
    }
}

impl<T: ?Sized> From<*const T> for PtrId<T> {
    fn from(value: *const T) -> Self {
        Self(value)
    }
}

impl<T: ?Sized> From<*mut T> for PtrId<T> {
    fn from(value: *mut T) -> Self {
        Self(value)
    }
}

impl<T: ?Sized> From<&T> for PtrId<T> {
    fn from(value: &T) -> Self {
        Self(value)
    }
}

impl<T: ?Sized> From<&mut T> for PtrId<T> {
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
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> PartialEq for PtrId<T> {
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
macro_rules! token {
    ($name:ident) => {
        $crate::token!(pub(self) struct $name);
    };
    ($(#[$($attr:meta)*])* struct $name:ident;) => {
        $crate::token!($(#[$($attr)*])* pub(self) struct $name;);
    };
    ($(#[$($attr:meta)*])* $vis:vis struct $name:ident;) => {
        $(#[$($attr)*])*
        $vis struct $name;
        const _: () = {
            static INITIALIZED: ::core::sync::atomic::AtomicBool = ::core::sync::atomic::AtomicBool::new(false);
            impl $name {
                $vis fn try_new() -> Result<Self, $crate::error::AlreadyInitialized> {
                    if INITIALIZED.swap(true, ::core::sync::atomic::Ordering::Relaxed) {
                        Err($crate::error::AlreadyInitialized)
                    } else {
                        Ok(Self)
                    }
                }

                $vis fn new() -> Self {
                    Self::try_new().unwrap()
                }
            }

            impl Drop for $name {
                fn drop(&mut self) {
                    INITIALIZED.store(false, ::core::sync::atomic::Ordering::Relaxed);
                }
            }

            // SAFETY: Each forged token type can only have a single instance
            unsafe impl $crate::Token for $name {
                type Id = ();

                fn id(&self) -> Self::Id {}

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
/// non-trivial unicity algorithm named "allocator".
#[derive(Debug, Eq, PartialEq)]
pub struct DynamicToken(usize);

impl DynamicToken {
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
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: Each `DynamicToken` has a different inner token and cannot be cloned
unsafe impl Token for DynamicToken {
    type Id = usize;

    fn id(&self) -> Self::Id {
        self.0
    }

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
    pub fn from_ref(t: &T) -> &Self {
        // SAFETY: `RefMutToken` is `repr(transparent)`
        unsafe { &*(t as *const T as *const Self) }
    }

    pub fn from_mut(t: &mut T) -> &mut Self {
        // SAFETY: `RefMutToken` is `repr(transparent)`
        unsafe { &mut *(t as *mut T as *mut Self) }
    }

    #[inline]
    pub fn try_borrow<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(&'a T) -> &'a TokenCell<U, Self>,
    ) -> Result<Ref<'a, U, Self>, BorrowError> {
        self.try_borrow_option(|r| Some(cell(r))).unwrap()
    }

    #[inline]
    pub fn borrow<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(&'a T) -> &'a TokenCell<U, Self>,
    ) -> Ref<'a, U, Self> {
        self.try_borrow(cell).unwrap()
    }

    #[inline]
    pub fn try_borrow_option<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(&'a T) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<Result<Ref<'a, U, Self>, BorrowError>> {
        self.try_borrow_iter(cell).next()
    }

    #[inline]
    pub fn borrow_option<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(&'a T) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<Ref<'a, U, Self>> {
        self.try_borrow_option(cell).map(Result::unwrap)
    }

    #[inline]
    pub fn try_borrow_iter<'a, U: ?Sized + 'a, I: IntoIterator<Item = &'a TokenCell<U, Self>>>(
        &'a self,
        cells: impl FnOnce(&'a T) -> I,
    ) -> impl Iterator<Item = Result<Ref<'a, U, Self>, BorrowError>> {
        let token_id = self.id();
        cells(self)
            .into_iter()
            .map(move |cell| cell.get_ref(token_id))
    }

    #[inline]
    pub fn borrow_iter<'a, U: ?Sized + 'a, I: IntoIterator<Item = &'a TokenCell<U, Self>>>(
        &'a self,
        cells: impl FnOnce(&'a T) -> I,
    ) -> impl Iterator<Item = Ref<'a, U, Self>> {
        self.try_borrow_iter(cells).map(Result::unwrap)
    }

    #[inline]
    pub fn try_borrow_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(&'a mut T) -> &'a TokenCell<U, Self>,
    ) -> Result<RefMut<'a, U, Self>, BorrowMutError> {
        self.try_borrow_option_mut(|r| Some(cell(r))).unwrap()
    }

    #[inline]
    pub fn borrow_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(&'a mut T) -> &'a TokenCell<U, Self>,
    ) -> RefMut<'a, U, Self> {
        self.try_borrow_mut(cell).unwrap()
    }

    #[inline]
    pub fn try_borrow_option_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(&'a mut T) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<Result<RefMut<'a, U, Self>, BorrowMutError>> {
        self.try_borrow_iter_mut(cell).next()
    }

    #[inline]
    pub fn borrow_option_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(&'a mut T) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<RefMut<'a, U, Self>> {
        self.try_borrow_option_mut(cell).map(Result::unwrap)
    }

    #[inline]
    pub fn try_borrow_iter_mut<
        'a,
        U: ?Sized + 'a,
        I: IntoIterator<Item = &'a TokenCell<U, Self>>,
    >(
        &'a mut self,
        cells: impl FnOnce(&'a mut T) -> I,
    ) -> impl Iterator<Item = Result<RefMut<'a, U, Self>, BorrowMutError>> {
        let token_id = self.id();
        cells(self)
            .into_iter()
            // SAFETY: `RefMutToken::is_unique` always returns true
            .map(move |cell| unsafe { cell.get_mut(token_id) })
    }

    #[inline]
    pub fn borrow_iter_mut<'a, U: ?Sized + 'a, I: IntoIterator<Item = &'a TokenCell<U, Self>>>(
        &'a mut self,
        cells: impl FnOnce(&'a mut T) -> I,
    ) -> impl Iterator<Item = RefMut<'a, U, Self>> {
        self.try_borrow_iter_mut(cells).map(Result::unwrap)
    }
}

impl<T: ?Sized> Deref for RefMutToken<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ?Sized> DerefMut for RefMutToken<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T: ?Sized> From<&'a T> for &'a RefMutToken<T> {
    fn from(value: &'a T) -> Self {
        RefMutToken::from_ref(value)
    }
}

impl<'a, T: ?Sized> From<&'a mut T> for &'a mut RefMutToken<T> {
    fn from(value: &'a mut T) -> Self {
        RefMutToken::from_mut(value)
    }
}

// SAFETY: token uniqueness is guaranteed by mutable reference properties
unsafe impl<T: ?Sized> Token for RefMutToken<T> {
    type Id = PtrId<T>;

    fn id(&self) -> Self::Id {
        (&self.0).into()
    }

    fn is_unique(&mut self) -> bool {
        true
    }
}

/// Abstraction of a pinned exclusive/mutable reference as a token.
#[repr(transparent)]
pub struct PinToken<T: ?Sized>(T);

impl<T: ?Sized> PinToken<T> {
    pub fn from_ref(t: Pin<&T>) -> &Self {
        // SAFETY: `PinToken` is `repr(transparent)`
        unsafe { &*(t.get_ref() as *const T as *const Self) }
    }

    pub fn from_mut(t: Pin<&mut T>) -> &mut Self {
        // SAFETY: mutable ref is never accessed and `RefMutToken` is `repr(transparent)`
        unsafe { &mut *(t.get_unchecked_mut() as *mut T as *mut Self) }
    }

    #[inline]
    pub fn try_borrow<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(Pin<&'a T>) -> &'a TokenCell<U, Self>,
    ) -> Result<Ref<'a, U, Self>, BorrowError> {
        self.try_borrow_option(|r| Some(cell(r))).unwrap()
    }

    #[inline]
    pub fn borrow<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(Pin<&'a T>) -> &'a TokenCell<U, Self>,
    ) -> Ref<'a, U, Self> {
        self.try_borrow(cell).unwrap()
    }

    #[inline]
    pub fn try_borrow_option<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(Pin<&'a T>) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<Result<Ref<'a, U, Self>, BorrowError>> {
        self.try_borrow_iter(cell).next()
    }

    #[inline]
    pub fn borrow_option<'a, U: ?Sized>(
        &'a self,
        cell: impl FnOnce(Pin<&'a T>) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<Ref<'a, U, Self>> {
        self.try_borrow_option(cell).map(Result::unwrap)
    }

    #[inline]
    pub fn try_borrow_iter<'a, U: ?Sized + 'a, I: IntoIterator<Item = &'a TokenCell<U, Self>>>(
        &'a self,
        cells: impl FnOnce(Pin<&'a T>) -> I,
    ) -> impl Iterator<Item = Result<Ref<'a, U, Self>, BorrowError>> {
        let token_id = self.id();
        // SAFETY: `PinToken` is initialized with pin data
        cells(unsafe { Pin::new_unchecked(self) })
            .into_iter()
            .map(move |cell| cell.get_ref(token_id))
    }

    #[inline]
    pub fn borrow_iter<'a, U: ?Sized + 'a, I: IntoIterator<Item = &'a TokenCell<U, Self>>>(
        &'a self,
        cells: impl FnOnce(Pin<&'a T>) -> I,
    ) -> impl Iterator<Item = Ref<'a, U, Self>> {
        self.try_borrow_iter(cells).map(Result::unwrap)
    }

    #[inline]
    pub fn try_borrow_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(Pin<&'a mut T>) -> &'a TokenCell<U, Self>,
    ) -> Result<RefMut<'a, U, Self>, BorrowMutError> {
        self.try_borrow_option_mut(|r| Some(cell(r))).unwrap()
    }

    #[inline]
    pub fn borrow_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(Pin<&'a mut T>) -> &'a TokenCell<U, Self>,
    ) -> RefMut<'a, U, Self> {
        self.try_borrow_mut(cell).unwrap()
    }

    #[inline]
    pub fn try_borrow_option_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(Pin<&'a mut T>) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<Result<RefMut<'a, U, Self>, BorrowMutError>> {
        self.try_borrow_iter_mut(cell).next()
    }

    #[inline]
    pub fn borrow_option_mut<'a, U: ?Sized>(
        &'a mut self,
        cell: impl FnOnce(Pin<&'a mut T>) -> Option<&'a TokenCell<U, Self>>,
    ) -> Option<RefMut<'a, U, Self>> {
        self.try_borrow_option_mut(cell).map(Result::unwrap)
    }

    #[inline]
    pub fn try_borrow_iter_mut<
        'a,
        U: ?Sized + 'a,
        I: IntoIterator<Item = &'a TokenCell<U, Self>>,
    >(
        &'a mut self,
        cells: impl FnOnce(Pin<&'a mut T>) -> I,
    ) -> impl Iterator<Item = Result<RefMut<'a, U, Self>, BorrowMutError>> {
        let token_id = self.id();
        // SAFETY: `PinToken` is initialized with pin data
        cells(unsafe { Pin::new_unchecked(self) })
            .into_iter()
            // SAFETY: `RefMutToken::is_unique` always returns true
            .map(move |cell| unsafe { cell.get_mut(token_id) })
    }

    #[inline]
    pub fn borrow_iter_mut<'a, U: ?Sized + 'a, I: IntoIterator<Item = &'a TokenCell<U, Self>>>(
        &'a mut self,
        cells: impl FnOnce(Pin<&'a mut T>) -> I,
    ) -> impl Iterator<Item = RefMut<'a, U, Self>> {
        self.try_borrow_iter_mut(cells).map(Result::unwrap)
    }
}

impl<T: ?Sized> Deref for PinToken<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ?Sized> DerefMut for PinToken<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T: ?Sized> From<Pin<&'a T>> for &'a PinToken<T> {
    fn from(value: Pin<&'a T>) -> Self {
        PinToken::from_ref(value)
    }
}

impl<'a, T: ?Sized> From<Pin<&'a mut T>> for &'a mut PinToken<T> {
    fn from(value: Pin<&'a mut T>) -> Self {
        PinToken::from_mut(value)
    }
}

// SAFETY: token uniqueness is guaranteed by mutable reference properties
unsafe impl<T: ?Sized> Token for PinToken<T> {
    type Id = PtrId<T>;

    fn id(&self) -> Self::Id {
        (&self.0).into()
    }

    fn is_unique(&mut self) -> bool {
        true
    }
}

/// Dummy struct which can be used with smart-pointer-based token implementations.
///
/// Smart-pointer requires a non-zero-sized type parameter to be used as token.
/// Use it if, like me, you don't like finding names for dummy objects;
/// `Box<AllocatedToken>` is still better than `Box<u8>`.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct AllocatedToken(u8);

#[cfg(feature = "alloc")]
const _: () = {
    extern crate alloc;
    use alloc::{boxed::Box, rc::Rc, sync::Arc};

    trait NotZeroSized<T> {
        const ASSERT_SIZE_IS_NOT_ZERO: () = assert!(size_of::<T>() > 0);
    }

    impl<T> NotZeroSized<T> for T {}

    // SAFETY: It's not possible to have simultaneously two boxes with the same pointer/token id,
    // as long as `T` is not zero-sized (ensured by a compile-time check)
    /// `T` must not be zero-sized.
    unsafe impl<T> Token for Box<T> {
        type Id = PtrId<T>;

        fn id(&self) -> Self::Id {
            {
                T::ASSERT_SIZE_IS_NOT_ZERO
            }
            self.as_ref().into()
        }

        fn is_unique(&mut self) -> bool {
            {
                T::ASSERT_SIZE_IS_NOT_ZERO
            }
            true
        }
    }

    // SAFETY: `Rc::get_mut` ensures the unicity of the `Rc` instance "owning" its inner pointer,
    // as long as `T` is not zero-sized (ensured by a compile-time check)
    /// `T` must not be zero-sized.
    unsafe impl<T> Token for Rc<T> {
        type Id = PtrId<T>;

        fn id(&self) -> Self::Id {
            {
                T::ASSERT_SIZE_IS_NOT_ZERO
            }
            self.as_ref().into()
        }

        fn is_unique(&mut self) -> bool {
            {
                T::ASSERT_SIZE_IS_NOT_ZERO
            }
            Rc::get_mut(self).is_some()
        }
    }

    // SAFETY: `Arc::get_mut` ensures the unicity of the `Rc` instance "owning" its inner pointer,
    // as long as `T` is not zero-sized (ensured by a compile-time check)
    /// `T` must not be zero-sized.
    unsafe impl<T> Token for Arc<T> {
        type Id = PtrId<T>;

        fn id(&self) -> Self::Id {
            {
                T::ASSERT_SIZE_IS_NOT_ZERO
            }
            self.as_ref().into()
        }

        fn is_unique(&mut self) -> bool {
            {
                T::ASSERT_SIZE_IS_NOT_ZERO
            }
            Arc::get_mut(self).is_some()
        }
    }
};

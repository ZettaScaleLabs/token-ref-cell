#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(clippy::semicolon_if_nothing_returned)]
#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![forbid(unsafe_op_in_unsafe_fn)]
#![forbid(clippy::undocumented_unsafe_blocks)]
//! This library provides [`TokenCell`], an interior mutability cell, which uses an
//! external [`Token`] to synchronize its accesses.
//!
//! Multiple token [implementations](token) are provided, the easiest to use being the
//! smart-pointer-based ones: every `Box<T>` can indeed be used as a token (as long as `T`
//! is not a ZST). The recommended token implementation is [`BoxToken`], and it's the
//! default value of the generic parameter of [`TokenCell`].
//!
//! The runtime cost is very lightweight: only one pointer comparison for
//! [`TokenCell::borrow`]/[`TokenCell::borrow_mut`] when using [`BoxToken`]
//! (and zero-cost when using [`singleton_token!`]).
//! <br>
//! Because one token can be used with multiple cells, it's possible for example to use
//! a single rwlock wrapping a token to synchronize mutable access to multiple `Arc` data.
//!
//! # Examples
//!
//! ```rust
//! # use std::sync::{Arc, RwLock};
//! # use token_cell2::{TokenCell, BoxToken};
//! let mut token = RwLock::new(BoxToken::new());
//! // Initialize a vector of arcs
//! let token_ref = token.read().unwrap();
//! let arc_vec = vec![Arc::new(TokenCell::new(0, &*token_ref)); 42];
//! drop(token_ref);
//! // Use only one rwlock to write to all the arcs
//! let mut token_mut = token.write().unwrap();
//! for cell in &arc_vec {
//!     *cell.borrow_mut(&mut *token_mut) += 1;
//! }
//! drop(token_mut)
//! ```

use core::{
    cell::UnsafeCell,
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    error::{BorrowError, BorrowMutError},
    token::{ConstToken, Token},
};

pub mod error;
pub mod token;

#[cfg(feature = "alloc")]
pub use token::BoxToken;

// Outline panic formatting, see `RefCell` implementation
macro_rules! unwrap {
    ($expr:expr) => {
        match $expr {
            Ok(ok) => ok,
            Err(err) => err.panic(),
        }
    };
}

// Ensure `Tk` invariance to avoid subtyping.
struct TokenId<Tk: Token + ?Sized> {
    id: Tk::Id,
    _phantom: PhantomData<fn(Tk) -> Tk>,
}

impl<Tk: Token + ?Sized> TokenId<Tk> {
    const fn new(id: Tk::Id) -> Self {
        let _phantom = PhantomData;
        Self { id, _phantom }
    }
}

impl<Tk: Token + ?Sized> Clone for TokenId<Tk> {
    fn clone(&self) -> Self {
        Self::new(self.id.clone())
    }
}

impl<Tk: Token + ?Sized> PartialEq for TokenId<Tk> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<Tk: Token + ?Sized> fmt::Debug for TokenId<Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.id, f)
    }
}

/// Interior mutability cell using an external [`Token`] to synchronize accesses.
pub struct TokenCell<
    T: ?Sized,
    #[cfg(not(feature = "alloc"))] Tk: Token + ?Sized,
    #[cfg(feature = "alloc")] Tk: Token + ?Sized = BoxToken,
> {
    token_id: TokenId<Tk>,
    cell: UnsafeCell<T>,
}

impl<T: ?Sized, Tk: Token + ?Sized> AsRef<TokenCell<T, Tk>> for &TokenCell<T, Tk> {
    fn as_ref(&self) -> &TokenCell<T, Tk> {
        self
    }
}

// SAFETY: Inner `UnsafeCell` access is synchronized using `Token` uniqueness guarantee.
unsafe impl<T: ?Sized + Sync, Tk: Token + ?Sized> Sync for TokenCell<T, Tk> where Tk::Id: Sync {}

impl<T, Tk: Token + ?Sized> TokenCell<T, Tk> {
    /// Creates a new `TokenCell` containing a `value`, synchronized by the given `token`.
    #[inline]
    pub fn new<'a>(value: T, token: &'a Tk) -> Self
    where
        Tk: 'a,
    {
        Self {
            token_id: TokenId::new(token.id()),
            cell: value.into(),
        }
    }

    /// Creates a new `TokenCell` containing a `value`, synchronized by a const token.
    #[inline]
    pub fn new_const(value: T) -> Self
    where
        Tk: ConstToken,
    {
        Self {
            token_id: TokenId::new(Tk::ID),
            cell: UnsafeCell::new(value),
        }
    }

    /// Consumes the `TokenCell`, returning the wrapped value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.cell.into_inner()
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> TokenCell<T, Tk> {
    /// Set a new token to synchronize the cell.
    #[inline]
    pub fn set_token<'a>(&mut self, token: &'a Tk)
    where
        Tk: 'a,
    {
        self.token_id = TokenId::new(token.id());
    }

    #[inline]
    fn get_ref(&self, token_id: TokenId<Tk>) -> Result<Ref<T, Tk>, BorrowError> {
        if self.token_id == token_id {
            Ok(Ref {
                // SAFETY: `Token` trait guarantees that there can be only one token
                // able to borrow the cell. Having a shared reference to this token
                // ensures that the cell cannot be borrowed mutably.
                inner: unsafe { &*self.cell.get() },
                token_id,
                _phantom: PhantomData,
            })
        } else {
            Err(BorrowError)
        }
    }

    /// Immutably borrows the wrapped value using a shared reference to the token.
    ///
    /// The token reference is reborrowed in the returned `Ref`, preventing any aliasing
    /// mutable borrow. Multiple immutable borrows can be taken out at the same time.
    ///
    /// The check runtime cost is only a token id comparison (when the token is not a
    /// [`singleton_token!`]); there is no write, contrary to [`RefCell`](core::cell::RefCell)
    /// or locks.
    ///
    /// # Panics
    ///
    /// Panics if the token doesn't match the one used at cell initialization (or set with
    /// [`set_token`](Self::set_token)). For a non-panicking variant, use
    /// [`try_borrow`](Self::try_borrow).
    #[inline]
    pub fn borrow<'a>(&'a self, token: &'a Tk) -> Ref<'a, T, Tk> {
        unwrap!(self.try_borrow(token))
    }

    /// Tries to immutably borrow the wrapped value using a shared reference to the token.
    ///
    /// Returns an error if the token doesn't match the one used at cell initialization
    /// (or set with [`set_token`](Self::set_token)). This is the non-panicking variant of
    /// [`borrow`](Self::borrow).
    ///
    /// The token reference is reborrowed in the returned `Ref`, preventing any aliasing
    /// mutable borrow. Multiple immutable borrows can be taken out at the same time.
    ///
    /// The check runtime cost is only a token id comparison (when the token is not a
    /// [`singleton_token!`]); there is no write, contrary to [`RefCell`](core::cell::RefCell)
    /// or locks.
    #[inline]
    pub fn try_borrow<'a>(&'a self, token: &'a Tk) -> Result<Ref<'a, T, Tk>, BorrowError> {
        self.get_ref(TokenId::new(token.id()))
    }

    /// # Safety
    ///
    /// Token id must be retrieved from a unique token.
    #[inline]
    unsafe fn get_mut(&self, token_id: TokenId<Tk>) -> Result<RefMut<T, Tk>, BorrowMutError> {
        if self.token_id == token_id {
            Ok(RefMut {
                // SAFETY: `Token` trait guarantees that there can be only one token
                // able to borrow the cell. Having an exclusive reference to this token
                // ensures that the cell is exclusively borrowed.
                inner: unsafe { &mut *self.cell.get() },
                token_id,
                _phantom: PhantomData,
            })
        } else {
            Err(BorrowMutError::WrongToken)
        }
    }

    /// Mutably borrows the wrapped value using an exclusive reference to the token.
    ///
    /// The token reference is reborrowed in the returned `RefMut`, preventing any aliasing
    /// mutable/immutable borrow.
    ///
    /// The check runtime cost is only a token id comparison (when the token is not a
    /// [`singleton_token!`]), as well as a token unicity check (mostly a noop for most
    /// of the token implementations); there is no write, contrary to
    /// [`RefCell`](core::cell::RefCell) or locks.
    ///
    /// # Panics
    ///
    /// Panics if the token doesn't match the one used at cell initialization (or set with
    /// [`set_token`](Self::set_token)), or if the token is not [unique](Token#safety).
    /// For a non-panicking variant, use [`try_borrow`](Self::try_borrow_mut).
    #[inline]
    pub fn borrow_mut<'a>(&'a self, token: &'a mut Tk) -> RefMut<'a, T, Tk> {
        unwrap!(self.try_borrow_mut(token))
    }

    /// Mutably borrows the wrapped value using an exclusive reference to the token.
    ///
    /// Returns an error if the token doesn't match the one used at cell initialization
    /// (or set with [`set_token`](Self::set_token)), or if the token is not
    /// [unique](Token#safety). For a non-panicking variant, use
    /// [`try_borrow`](Self::try_borrow_mut).
    ///
    /// The token reference is reborrowed in the returned `RefMut`, preventing any aliasing
    /// mutable/immutable borrow.
    ///
    /// The check runtime cost is only a token id comparison (when the token is not a
    /// [`singleton_token!`]), as well as a token unicity check (mostly a noop for most
    /// of the token implementations); there is no write, contrary to
    /// [`RefCell`](core::cell::RefCell) or locks.
    #[inline]
    pub fn try_borrow_mut<'a>(
        &'a self,
        token: &'a mut Tk,
    ) -> Result<RefMut<'a, T, Tk>, BorrowMutError> {
        if !token.is_unique() {
            return Err(BorrowMutError::NotUniqueToken);
        }
        // SAFETY: uniqueness is checked above
        unsafe { self.get_mut(TokenId::new(token.id())) }
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> Deref for TokenCell<T, Tk> {
    type Target = UnsafeCell<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.cell
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> DerefMut for TokenCell<T, Tk> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cell
    }
}

impl<T: ?Sized + fmt::Debug, Tk: Token + ?Sized> fmt::Debug for TokenCell<T, Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenCell")
            .field("token", &self.token_id)
            .finish_non_exhaustive()
    }
}

/// A wrapper type for an immutably borrowed value from a [`TokenCell`].
///
/// The token used for borrowing synchronization is also borrowed.
/// Dedicated methods allows reusing the token for further borrowing.
pub struct Ref<
    'b,
    T: ?Sized,
    #[cfg(not(feature = "alloc"))] Tk: Token + ?Sized,
    #[cfg(feature = "alloc")] Tk: Token + ?Sized = BoxToken,
> {
    inner: &'b T,
    token_id: TokenId<Tk>,
    _phantom: PhantomData<&'b Tk>,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> Ref<'b, T, Tk> {
    /// Unwraps the inner reference.
    pub fn into_ref(self) -> &'b T {
        self.inner
    }

    /// Copies a `Ref`.
    ///
    /// This is an associated function that needs to be used as
    /// `Ref::clone(...)`. A `Clone` implementation or a method would interfere
    /// with the widespread use of `cell.borrow(&token).clone()` to clone the contents of
    /// a `TokenCell`.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    #[inline]
    pub fn clone(this: &Self) -> Self {
        Self {
            inner: this.inner,
            token_id: this.token_id.clone(),
            _phantom: PhantomData,
        }
    }

    /// Uses borrowed token shared reference to reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow`].
    #[inline]
    pub fn reborrow<U: ?Sized>(&self, cell: impl FnOnce(&T) -> &TokenCell<U, Tk>) -> Ref<U, Tk> {
        unwrap!(self.try_reborrow(cell))
    }

    /// Uses borrowed token shared reference to reborrow a `TokenCell`.
    ///
    /// See [`TokenCell::try_borrow`].
    #[inline]
    pub fn try_reborrow<U: ?Sized>(
        &self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
    ) -> Result<Ref<U, Tk>, BorrowError> {
        cell(self.inner).get_ref(self.token_id.clone())
    }

    /// Uses borrowed token shared reference to optionally reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow`].
    #[inline]
    pub fn reborrow_opt<U: ?Sized>(
        &self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<Ref<U, Tk>> {
        Some(unwrap!(self.try_reborrow_opt(cell)?))
    }

    /// Uses borrowed token shared reference to optionally reborrow a [`TokenCell`].
    ///
    /// See [`TokenCell::try_borrow`].
    #[inline]
    pub fn try_reborrow_opt<U: ?Sized>(
        &self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<Result<Ref<U, Tk>, BorrowError>> {
        Some(cell(self.inner)?.get_ref(self.token_id.clone()))
    }

    /// Wraps the borrowed token shared reference into a stateful [`Reborrow`] instance.
    ///
    /// It can be used for example to reborrow cells from an iterator.
    #[inline]
    pub fn reborrow_stateful<'a, S>(
        &'a self,
        state: impl FnOnce(&'a T) -> S,
    ) -> Reborrow<'a, S, Tk> {
        Reborrow {
            state: state(self.inner),
            token_id: self.token_id.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> Deref for Ref<'_, T, Tk> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: ?Sized + fmt::Debug, Tk: Token + ?Sized> fmt::Debug for Ref<'_, T, Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ref")
            .field("inner", &self.inner)
            .field("token_id", &self.token_id)
            .finish()
    }
}

impl<T: ?Sized + fmt::Display, Tk: Token + ?Sized> fmt::Display for Ref<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

/// Stateful wrapper around a borrowed token shared reference.
pub struct Reborrow<'b, S: ?Sized, Tk: Token + ?Sized> {
    token_id: TokenId<Tk>,
    _phantom: PhantomData<&'b Tk>,
    state: S,
}

impl<'b, S: ?Sized, Tk: Token + ?Sized> Reborrow<'b, S, Tk> {
    /// Uses borrowed token shared reference to reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow`].
    pub fn reborrow<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> &TokenCell<U, Tk>,
    ) -> Ref<U, Tk> {
        unwrap!(self.try_reborrow(cell))
    }

    /// Uses borrowed token shared reference to reborrow a `TokenCell`.
    ///
    /// See [`TokenCell::try_borrow`].
    pub fn try_reborrow<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> &TokenCell<U, Tk>,
    ) -> Result<Ref<U, Tk>, BorrowError> {
        cell(&mut self.state).get_ref(self.token_id.clone())
    }

    /// Uses borrowed token shared reference to optionally reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow`].
    pub fn reborrow_opt<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<Ref<U, Tk>> {
        Some(unwrap!(self.try_reborrow_opt(cell)?))
    }

    /// Uses borrowed token shared reference to optionally reborrow a [`TokenCell`].
    ///
    /// See [`TokenCell::try_borrow`].
    pub fn try_reborrow_opt<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<Result<Ref<U, Tk>, BorrowError>> {
        Some(cell(&mut self.state)?.get_ref(self.token_id.clone()))
    }
}

impl<S: ?Sized, Tk: Token + ?Sized> Deref for Reborrow<'_, S, Tk> {
    type Target = S;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<S: ?Sized, Tk: Token + ?Sized> DerefMut for Reborrow<'_, S, Tk> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<S: ?Sized + fmt::Debug, Tk: Token + ?Sized> fmt::Debug for Reborrow<'_, S, Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reborrow")
            .field("state", &&self.state)
            .field("token_id", &self.token_id)
            .finish()
    }
}

/// A wrapper type for a mutably borrowed value from a [`TokenCell`].
///
/// The token can be reused for further borrowing.
pub struct RefMut<
    'b,
    T: ?Sized,
    #[cfg(not(feature = "alloc"))] Tk: Token + ?Sized,
    #[cfg(feature = "alloc")] Tk: Token + ?Sized = BoxToken,
> {
    inner: &'b mut T,
    token_id: TokenId<Tk>,
    _phantom: PhantomData<&'b mut Tk>,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> RefMut<'b, T, Tk> {
    /// Unwraps the inner reference.
    pub fn into_mut(self) -> &'b mut T {
        self.inner
    }

    /// Borrows a `RefMut` as a [`Ref`].
    pub fn as_ref(&self) -> Ref<T, Tk> {
        Ref {
            inner: self.inner,
            token_id: self.token_id.clone(),
            _phantom: PhantomData,
        }
    }

    /// Uses borrowed token exclusive reference to reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow_mut`].
    #[inline]
    pub fn reborrow_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut T) -> &TokenCell<U, Tk>,
    ) -> RefMut<U, Tk> {
        unwrap!(self.try_reborrow_mut(cell))
    }

    /// Uses borrowed token exclusive reference to reborrow a `TokenCell`.
    ///
    /// See [`TokenCell::try_borrow_mut`].
    #[inline]
    pub fn try_reborrow_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut T) -> &TokenCell<U, Tk>,
    ) -> Result<RefMut<U, Tk>, BorrowMutError> {
        // SAFETY: token uniqueness has been checked to build the `RefMut`
        unsafe { cell(self.inner).get_mut(self.token_id.clone()) }
    }

    /// Uses borrowed token exclusive reference to optionally  reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow_mut`].
    #[inline]
    pub fn reborrow_opt_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut T) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<RefMut<U, Tk>> {
        Some(unwrap!(self.try_reborrow_opt_mut(cell)?))
    }

    /// Uses borrowed token exclusive reference to optionally reborrow a `TokenCell`.
    ///
    /// See [`TokenCell::try_borrow_mut`].
    #[inline]
    pub fn try_reborrow_opt_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut T) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<Result<RefMut<U, Tk>, BorrowMutError>> {
        // SAFETY: token uniqueness has been checked to build the `RefMut`
        Some(unsafe { cell(self.inner)?.get_mut(self.token_id.clone()) })
    }

    /// Wraps the borrowed token exclusive reference into a stateful [`ReborrowMut`] instance.
    ///
    /// It can be used for example to reborrow cells from an iterator.
    #[inline]
    pub fn reborrow_stateful_mut<'a, S>(
        &'a mut self,
        state: impl FnOnce(&'a mut T) -> S,
    ) -> ReborrowMut<'a, S, Tk> {
        ReborrowMut {
            state: state(self.inner),
            token_id: self.token_id.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> Deref for RefMut<'_, T, Tk> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> DerefMut for RefMut<'_, T, Tk> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

impl<T: ?Sized + fmt::Debug, Tk: Token + ?Sized> fmt::Debug for RefMut<'_, T, Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefMut")
            .field("inner", &self.inner)
            .field("token_id", &self.token_id)
            .finish()
    }
}

impl<T: ?Sized + fmt::Display, Tk: Token + ?Sized> fmt::Display for RefMut<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

/// Stateful wrapper around a borrowed token exclusive reference.
pub struct ReborrowMut<'b, S: ?Sized, Tk: Token + ?Sized> {
    token_id: TokenId<Tk>,
    _phantom: PhantomData<&'b mut Tk>,
    state: S,
}

impl<S: ?Sized, Tk: Token + ?Sized> ReborrowMut<'_, S, Tk> {
    /// Uses borrowed token exclusive reference to reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow_mut`].
    pub fn reborrow_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> &TokenCell<U, Tk>,
    ) -> RefMut<U, Tk> {
        unwrap!(self.try_reborrow_mut(cell))
    }

    /// Uses borrowed token exclusive reference to reborrow a `TokenCell`.
    ///
    /// See [`TokenCell::try_borrow_mut`].
    pub fn try_reborrow_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> &TokenCell<U, Tk>,
    ) -> Result<RefMut<U, Tk>, BorrowMutError> {
        // SAFETY: token uniqueness has been checked to build the `RefMut` and then `ReborrowMut`.
        unsafe { cell(&mut self.state).get_mut(self.token_id.clone()) }
    }

    /// Uses borrowed token exclusive reference to optionally  reborrow a [`TokenCell`].
    ///
    /// # Panics
    ///
    /// See [`TokenCell::borrow_mut`].
    pub fn reborrow_opt_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<RefMut<U, Tk>> {
        Some(unwrap!(self.try_reborrow_opt_mut(cell)?))
    }

    /// Uses borrowed token exclusive reference to optionally reborrow a `TokenCell`.
    ///
    /// See [`TokenCell::try_borrow_mut`].
    pub fn try_reborrow_opt_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&mut S) -> Option<&TokenCell<U, Tk>>,
    ) -> Option<Result<RefMut<U, Tk>, BorrowMutError>> {
        // SAFETY: token uniqueness has been checked to build the `RefMut` and then `ReborrowMut`.
        Some(unsafe { cell(&mut self.state)?.get_mut(self.token_id.clone()) })
    }
}

impl<S: ?Sized, Tk: Token + ?Sized> Deref for ReborrowMut<'_, S, Tk> {
    type Target = S;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<S: ?Sized, Tk: Token + ?Sized> DerefMut for ReborrowMut<'_, S, Tk> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<S: ?Sized + fmt::Debug, Tk: Token + ?Sized> fmt::Debug for ReborrowMut<'_, S, Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReborrowMut")
            .field("state", &&self.state)
            .field("token_id", &self.token_id)
            .finish()
    }
}

//! This library provides [`TokenCell`], an interior mutability cell, which uses an
//! external [`Token`] to synchronize its accesses.
//!
//! The runtime cost is very lightweight (if not elided). Because one token can be used
//! with multiple cells, it's possible for example to use a single mutex wrapping a token
//! to synchronize mutable access to multiple `Arc` data.
//!
//! Multiple token [implementations](token) are provided, the easiest to use being the
//! smart-pointer-based ones: every `Box<T>` can indeed be used as a token (as long as `T`
//! is not a ZST). The recommended token implementation is [`BoxToken`], and it's the
//! default value of the generic parameter of [`TokenCell`].
//!
//! # Examples
//!
//! ```rust
//! # use std::sync::{Arc, RwLock};
//! # use token_cell2::{TokenCell, BoxToken};
//! let mut token = RwLock::new(BoxToken::new());
//! // Initialize a vector of arcs
//! let mut arc_vec = Vec::new();
//! let token_ref = token.read().unwrap();
//! for i in 0..4 {
//!     arc_vec.push(Arc::new(TokenCell::new(i, &*token_ref)));
//! }
//! drop(token_ref);
//! // Use only one rwlock to write to all the arcs
//! let mut token_mut = token.write().unwrap();
//! for cell in &arc_vec {
//!     *cell.borrow_mut(&mut *token_mut) += 1;
//! }
//! drop(token_mut)
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

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

/// Interior mutability cell using an external [`Token`] to synchronize accesses.
pub struct TokenCell<
    T: ?Sized,
    #[cfg(not(feature = "alloc"))] Tk: Token + ?Sized,
    #[cfg(feature = "alloc")] Tk: Token + ?Sized = BoxToken,
> {
    token_id: Tk::Id,
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
    #[inline]
    pub fn new<'a>(value: T, token: &'a Tk) -> Self
    where
        Tk: 'a,
    {
        Self {
            token_id: token.id(),
            cell: value.into(),
        }
    }

    #[inline]
    pub fn new_const(value: T) -> Self
    where
        Tk: ConstToken,
    {
        Self {
            token_id: Tk::ID,
            cell: UnsafeCell::new(value),
        }
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.cell.into_inner()
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> TokenCell<T, Tk> {
    #[inline]
    pub fn set_token<'a>(&mut self, token: impl Into<&'a Tk>)
    where
        Tk: 'a,
    {
        self.token_id = token.into().id()
    }

    #[inline]
    fn get_ref(&self, token_id: Tk::Id) -> Result<Ref<T, Tk>, BorrowError> {
        if token_id == self.token_id {
            Ok(Ref {
                token_id,
                // SAFETY: `Token` trait guarantees that there can be only one token
                // able to borrow the cell. Having a shared reference to this token
                // ensures that the cell cannot be borrowed mutably.
                inner: unsafe { &*self.cell.get() },
                _phantom: PhantomData,
            })
        } else {
            Err(BorrowError)
        }
    }

    #[inline]
    pub fn try_borrow<'a>(&'a self, token: &'a Tk) -> Result<Ref<'a, T, Tk>, BorrowError> {
        self.get_ref(token.id())
    }

    #[inline]
    pub fn borrow<'a>(&'a self, token: &'a Tk) -> Ref<'a, T, Tk> {
        self.try_borrow(token).unwrap()
    }

    /// # Safety
    ///
    /// Token id must be retrieved from a unique token.
    #[inline]
    unsafe fn get_mut(&self, token_id: Tk::Id) -> Result<RefMut<T, Tk>, BorrowMutError> {
        if token_id == self.token_id {
            Ok(RefMut {
                token_id,
                // SAFETY: `Token` trait guarantees that there can be only one token
                // able to borrow the cell. Having an exclusive reference to this token
                // ensures that the cell is exclusively borrowed.
                inner: unsafe { &mut *self.cell.get() },
                _phantom: PhantomData,
            })
        } else {
            Err(BorrowMutError)
        }
    }

    #[inline]
    pub fn try_borrow_mut<'a>(
        &'a self,
        token: &'a mut Tk,
    ) -> Result<RefMut<'a, T, Tk>, BorrowMutError> {
        if !token.is_unique() {
            return Err(BorrowMutError);
        }
        // SAFETY: uniqueness is checked above
        unsafe { self.get_mut(token.id()) }
    }

    #[inline]
    pub fn borrow_mut<'a>(&'a self, token: &'a mut Tk) -> RefMut<'a, T, Tk> {
        self.try_borrow_mut(token).unwrap()
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
/// The token can be reused for further borrowing.
#[derive(Debug)]
pub struct Ref<
    'b,
    T: ?Sized,
    #[cfg(not(feature = "alloc"))] Tk: Token + ?Sized,
    #[cfg(feature = "alloc")] Tk: Token + ?Sized = BoxToken,
> {
    token_id: Tk::Id,
    inner: &'b T,
    _phantom: PhantomData<&'b Tk>,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> Ref<'b, T, Tk> {
    #[allow(clippy::should_implement_trait)]
    pub fn clone(this: &Self) -> Self {
        Self {
            token_id: this.token_id.clone(),
            inner: this.inner,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn try_reborrow<U, R>(
        &self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Result<Ref<U, Tk>, BorrowError>) -> R,
    ) -> R {
        f(cell(self.inner).get_ref(self.token_id.clone()))
    }

    #[inline]
    pub fn reborrow<U, R>(
        &self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Ref<U, Tk>) -> R,
    ) -> R {
        self.try_reborrow(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_opt<U, R>(
        &self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(Result<Ref<U, Tk>, BorrowError>) -> R,
    ) -> Option<R> {
        Some(f(cell(self.inner)?.get_ref(self.token_id.clone())))
    }

    #[inline]
    pub fn reborrow_opt<U, R>(
        &self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(Ref<U, Tk>) -> R,
    ) -> Option<R> {
        self.try_reborrow_opt(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_iter<'a, U: 'a, I, R>(
        &'a self,
        cell: impl FnOnce(&'a T) -> I,
        mut f: impl FnMut(Result<Ref<U, Tk>, BorrowError>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        cell(self.inner)
            .into_iter()
            .map(move |cell| f(cell.get_ref(self.token_id.clone())))
    }

    #[inline]
    pub fn reborrow_iter<'a, U: 'a, I: IntoIterator + 'a, R>(
        &'a self,
        cell: impl FnOnce(&T) -> I,
        mut f: impl FnMut(Ref<U, Tk>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        self.try_reborrow_iter(cell, move |res| f(res.unwrap()))
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> Deref for Ref<'_, T, Tk> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: ?Sized + fmt::Display, Tk: Token + ?Sized> fmt::Display for Ref<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

/// A wrapper type for a mutably borrowed value from a [`TokenCell`].
///
/// The token can be reused for further borrowing.
#[derive(Debug)]
pub struct RefMut<
    'b,
    T: ?Sized,
    #[cfg(not(feature = "alloc"))] Tk: Token + ?Sized,
    #[cfg(feature = "alloc")] Tk: Token + ?Sized = BoxToken,
> {
    token_id: Tk::Id,
    inner: &'b mut T,
    _phantom: PhantomData<&'b mut Tk>,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> RefMut<'b, T, Tk> {
    #[inline]
    pub fn try_reborrow<U, R>(
        &self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Result<Ref<U, Tk>, BorrowError>) -> R,
    ) -> R {
        f(cell(self.inner).get_ref(self.token_id.clone()))
    }

    #[inline]
    pub fn reborrow<U, R>(
        &self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Ref<U, Tk>) -> R,
    ) -> R {
        self.try_reborrow(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_opt<U, R>(
        &self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(Result<Ref<U, Tk>, BorrowError>) -> R,
    ) -> Option<R> {
        Some(f(cell(self.inner)?.get_ref(self.token_id.clone())))
    }

    #[inline]
    pub fn reborrow_opt<U, R>(
        &self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(Ref<U, Tk>) -> R,
    ) -> Option<R> {
        self.try_reborrow_opt(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_iter<'a, U: 'a, I, R>(
        &'a self,
        cell: impl FnOnce(&'a T) -> I,
        mut f: impl FnMut(Result<Ref<U, Tk>, BorrowError>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        cell(self.inner)
            .into_iter()
            .map(move |cell| f(cell.get_ref(self.token_id.clone())))
    }

    #[inline]
    pub fn reborrow_iter<'a, U: 'a, I: IntoIterator + 'a, R>(
        &'a self,
        cell: impl FnOnce(&'a T) -> I,
        mut f: impl FnMut(Ref<U, Tk>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        self.try_reborrow_iter(cell, move |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_mut<U, R>(
        &mut self,
        cell: impl FnOnce(&mut T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Result<RefMut<U, Tk>, BorrowMutError>) -> R,
    ) -> R {
        // SAFETY: token uniqueness has been checked to build the `RefMut`
        f(unsafe { cell(self.inner).get_mut(self.token_id.clone()) })
    }

    #[inline]
    pub fn reborrow_mut<U, R>(
        &mut self,
        cell: impl FnOnce(&mut T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(RefMut<U, Tk>) -> R,
    ) -> R {
        self.try_reborrow_mut(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_opt_mut<U, R>(
        &mut self,
        cell: impl FnOnce(&mut T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(Result<RefMut<U, Tk>, BorrowMutError>) -> R,
    ) -> Option<R> {
        // SAFETY: token uniqueness has been checked to build the `RefMut`
        Some(f(unsafe {
            cell(self.inner)?.get_mut(self.token_id.clone())
        }))
    }

    #[inline]
    pub fn reborrow_opt_mut<U, R>(
        &mut self,
        cell: impl FnOnce(&mut T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(RefMut<U, Tk>) -> R,
    ) -> Option<R> {
        self.try_reborrow_opt_mut(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_iter_mut<'a, U: 'a, I, R>(
        &'a mut self,
        cells: impl FnOnce(&'a mut T) -> I,
        mut f: impl FnMut(Result<RefMut<U, Tk>, BorrowMutError>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        let token_id = self.token_id.clone();
        cells(self.inner)
            .into_iter()
            // SAFETY: token uniqueness has been checked to build the `RefMut`
            .map(move |cell| f(unsafe { cell.as_ref().get_mut(token_id.clone()) }))
    }

    #[inline]
    pub fn reborrow_iter_mut<'a, U: 'a, I, R>(
        &'a mut self,
        cells: impl FnOnce(&'a mut T) -> I,
        mut f: impl FnMut(RefMut<U, Tk>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        self.try_reborrow_iter_mut(cells, move |res| f(res.unwrap()))
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

impl<T: ?Sized + fmt::Display, Tk: Token + ?Sized> fmt::Display for RefMut<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

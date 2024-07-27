//! This library provides [`TokenCell`], an interior mutability cell, which uses an
//! external [`Token`] to synchronize its accesses.
//!
//! The runtime cost is very lightweight (if not elided). Because one token can be used
//! with multiple cells, it's possible for example to use a single mutex wrapping a token
//! to synchronize mutable access to multiple `Arc` data.
//!
//! Multiple token [implementations](crate::token) are provided, the easiest to use being the
//! smart-pointer-based ones: every `Box<T>` can indeed be used as a token (as long as `T`
//! is not a ZST).
//!
//! # Examples
//!
//! ```rust
//! # use std::sync::Arc;
//! # use token_cell2::{TokenCell, token::AllocatedToken};
//! let mut token = Box::new(AllocatedToken::default());
//! let mut arc_vec = Vec::new();
//! for i in 0..4 {
//!     arc_vec.push(Arc::new(TokenCell::new(i, &token)));
//! }
//! for cell in &arc_vec {
//!     *cell.borrow_mut(&mut token) += 1;
//! }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

use core::{
    borrow::Borrow,
    cell::UnsafeCell,
    fmt,
    ops::{Deref, DerefMut},
};

use crate::{
    error::{BorrowError, BorrowMutError},
    token::{ConstToken, Token},
};

pub mod error;
pub mod token;

/// Interior mutability cell using an external [`Token`] to synchronize accesses.
pub struct TokenCell<T: ?Sized, Tk: Token + ?Sized> {
    token_id: Tk::Id,
    cell: UnsafeCell<T>,
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
    pub fn try_borrow<'a>(&'a self, token: &'a Tk) -> Result<Ref<'a, T, Tk>, BorrowError> {
        if token.id() == self.token_id {
            // SAFETY: `Token` trait guarantees that there can be only one token
            // able to borrow the cell. Having a shared reference to this token
            // ensures that the cell cannot be borrowed mutably.
            let inner = unsafe { &*self.cell.get() };
            Ok(Ref { inner, token })
        } else {
            Err(BorrowError)
        }
    }

    #[inline]
    pub fn borrow<'a>(&'a self, token: &'a Tk) -> Ref<'a, T, Tk> {
        self.try_borrow(token).unwrap()
    }

    #[inline]
    pub fn try_borrow_mut<'a>(
        &'a self,
        token: &'a mut Tk,
    ) -> Result<RefMut<'a, T, Tk>, BorrowMutError> {
        if token.is_unique() && token.id() == self.token_id {
            // SAFETY: `Token` trait guarantees that there can be only one token
            // able to borrow the cell. Having an exclusive reference to this token
            // ensures that the cell is exclusively borrowed.
            let inner = unsafe { &mut *self.cell.get() };
            Ok(RefMut { inner, token })
        } else {
            Err(BorrowMutError)
        }
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
pub struct Ref<'b, T: ?Sized, Tk: Token + ?Sized> {
    token: &'b Tk,
    inner: &'b T,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> Ref<'b, T, Tk> {
    pub fn clone(this: &Self) -> Self {
        Self {
            token: this.token,
            inner: this.inner,
        }
    }

    #[inline]
    pub fn try_reborrow<'a, U, R>(
        &'a self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Result<Ref<U, Tk>, BorrowError>) -> R,
    ) -> R {
        f(cell(self.inner).try_borrow(self.token))
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
        Some(f(cell(self.inner)?.try_borrow(self.token)))
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
            .map(move |cell| f(cell.try_borrow(self.token)))
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
pub struct RefMut<'b, T: ?Sized, Tk: Token + ?Sized> {
    token: &'b mut Tk,
    inner: &'b mut T,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> RefMut<'b, T, Tk> {
    #[inline]
    pub fn try_reborrow<U, R>(
        &self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Result<Ref<U, Tk>, BorrowError>) -> R,
    ) -> R {
        f(cell(self.inner).try_borrow(self.token))
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
        Some(f(cell(self.inner)?.try_borrow(self.token)))
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
            .map(move |cell| f(cell.try_borrow(self.token)))
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
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(Result<RefMut<U, Tk>, BorrowMutError>) -> R,
    ) -> R {
        f(cell(self.inner).try_borrow_mut(self.token))
    }

    #[inline]
    pub fn reborrow_mut<U, R>(
        &mut self,
        cell: impl FnOnce(&T) -> &TokenCell<U, Tk>,
        f: impl FnOnce(RefMut<U, Tk>) -> R,
    ) -> R {
        self.try_reborrow_mut(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_opt_mut<U, R>(
        &mut self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(Result<RefMut<U, Tk>, BorrowMutError>) -> R,
    ) -> Option<R> {
        Some(f(cell(self.inner)?.try_borrow_mut(self.token)))
    }

    #[inline]
    pub fn reborrow_opt_mut<U, R>(
        &mut self,
        cell: impl FnOnce(&T) -> Option<&TokenCell<U, Tk>>,
        f: impl FnOnce(RefMut<U, Tk>) -> R,
    ) -> Option<R> {
        self.try_reborrow_opt_mut(cell, |res| f(res.unwrap()))
    }

    #[inline]
    pub fn try_reborrow_iter_mut<'a, U: 'a, I, R>(
        &'a mut self,
        cell: impl FnOnce(&'a T) -> I,
        mut f: impl FnMut(Result<RefMut<U, Tk>, BorrowMutError>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        let token = &mut *self.token;
        cell(self.inner)
            .into_iter()
            .map(move |cell| f(cell.try_borrow_mut(token)))
    }

    #[inline]
    pub fn reborrow_iter_mut<'a, U: 'a, I, R>(
        &'a mut self,
        cell: impl FnOnce(&'a T) -> I,
        mut f: impl FnMut(RefMut<U, Tk>) -> R + 'a,
    ) -> impl Iterator<Item = R> + 'a
    where
        I: IntoIterator<Item = &'a TokenCell<U, Tk>> + 'a,
    {
        self.try_reborrow_iter_mut(cell, move |res| f(res.unwrap()))
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

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
    pub fn try_borrow<'a>(
        &'a self,
        token: impl Into<&'a Tk>,
    ) -> Result<Ref<'a, T, Tk>, BorrowError> {
        let token = token.into();
        if token.id() == self.token_id {
            // TODO safety
            let inner = unsafe { &*self.cell.get() };
            Ok(Ref { inner, token })
        } else {
            Err(BorrowError)
        }
    }

    #[inline]
    pub fn borrow<'a>(&'a self, token: impl Into<&'a Tk>) -> Ref<'a, T, Tk> {
        self.try_borrow(token).unwrap()
    }

    #[inline]
    pub fn try_borrow_mut<'a>(
        &'a self,
        token: impl Into<&'a mut Tk>,
    ) -> Result<RefMut<'a, T, Tk>, BorrowMutError> {
        let token = token.into();
        if token.is_unique() && token.id() == self.token_id {
            // TODO safety
            let inner = unsafe { &mut *self.cell.get() };
            Ok(RefMut { inner, token })
        } else {
            Err(BorrowMutError)
        }
    }

    #[inline]
    pub fn borrow_mut<'a>(&'a self, token: impl Into<&'a mut Tk>) -> RefMut<'a, T, Tk> {
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
pub struct Ref<'b, T: ?Sized, Tk: ?Sized> {
    pub inner: &'b T,
    pub token: &'b Tk,
}

impl<T: ?Sized, Tk: ?Sized> Deref for Ref<'_, T, Tk> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: ?Sized + fmt::Display, Tk: ?Sized> fmt::Display for Ref<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

/// A wrapper type for a mutably borrowed value from a [`TokenCell`].
///
/// The token can be reused for further borrowing.
#[derive(Debug)]
pub struct RefMut<'b, T: ?Sized, Tk: ?Sized> {
    pub inner: &'b mut T,
    pub token: &'b mut Tk,
}

impl<T: ?Sized, Tk: ?Sized> Deref for RefMut<'_, T, Tk> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: ?Sized, Tk: ?Sized> DerefMut for RefMut<'_, T, Tk> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

impl<T: ?Sized + fmt::Display, Tk: ?Sized> fmt::Display for RefMut<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

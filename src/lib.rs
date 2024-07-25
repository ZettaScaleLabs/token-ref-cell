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
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{
    error::{BorrowError, BorrowMutError},
    token::Token,
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
    pub fn set_token<'a>(&mut self, token: impl Into<&'a Tk>)
    where
        Tk: 'a,
    {
        self.token_id = token.into().id()
    }

    #[inline]
    pub const fn with_token_id(value: T, token_id: Tk::Id) -> Self {
        let cell = UnsafeCell::new(value);
        Self { token_id, cell }
    }

    #[inline]
    pub const fn token_id(&self) -> &Tk::Id {
        &self.token_id
    }

    #[inline]
    pub fn set_token_id(&mut self, token_id: Tk::Id) {
        self.token_id = token_id;
    }

    #[inline]
    pub fn into_inner(self) -> T {
        self.cell.into_inner()
    }
}
impl<T: ?Sized, Tk: Token + ?Sized> TokenCell<T, Tk> {
    #[inline]
    pub(crate) fn get_ref(&self, token_id: Tk::Id) -> Result<Ref<T, Tk>, BorrowError> {
        if token_id == self.token_id {
            Ok(Ref {
                token_id,
                value: NonNull::new(self.cell.get()).unwrap(),
                marker: PhantomData,
            })
        } else {
            Err(BorrowError)
        }
    }

    #[inline]
    pub fn try_borrow<'a>(
        &'a self,
        token: impl Into<&'a Tk>,
    ) -> Result<Ref<'a, T, Tk>, BorrowError> {
        self.get_ref(token.into().id())
    }

    #[inline]
    pub fn borrow<'a>(&'a self, token: impl Into<&'a Tk>) -> Ref<'a, T, Tk> {
        self.try_borrow(token).unwrap()
    }

    #[inline]
    pub(crate) unsafe fn get_mut(&self, token_id: Tk::Id) -> Result<RefMut<T, Tk>, BorrowMutError> {
        if token_id == self.token_id {
            Ok(RefMut {
                token_id,
                value: NonNull::new(self.cell.get()).unwrap(),
                marker: PhantomData,
            })
        } else {
            Err(BorrowMutError)
        }
    }

    #[inline]
    pub fn try_borrow_mut<'a>(
        &'a self,
        token: impl Into<&'a mut Tk>,
    ) -> Result<RefMut<'a, T, Tk>, BorrowMutError> {
        let token = token.into();
        if !token.is_unique() {
            return Err(BorrowMutError);
        }
        // SAFETY: uniqueness is checked above
        unsafe { self.get_mut(token.id()) }
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
/// It can be used to reborrow a component of the borrowed data.
pub struct Ref<'b, T: ?Sized, Tk: Token + ?Sized> {
    token_id: Tk::Id,
    // NB: we use a pointer instead of `&'b T` to avoid `noalias` violations, because a
    // `Ref` argument doesn't hold immutability for its whole scope, only until it drops.
    value: NonNull<T>,
    marker: PhantomData<(&'b Tk, &'b T)>,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> Ref<'b, T, Tk> {
    #[inline]
    pub(crate) fn new(token_id: Tk::Id, value: *const T) -> Self {
        Self {
            token_id,
            value: NonNull::new(value.cast_mut()).unwrap(),
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn token_id(&self) -> &Tk::Id {
        &self.token_id
    }

    #[inline]
    pub fn into_ref(self) -> &'b T {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_ref() }
    }

    #[inline]
    pub fn try_reborrow<U>(
        &self,
        cell: impl FnOnce(&'b T) -> &'b TokenCell<U, Tk>,
    ) -> Result<Ref<'b, U, Tk>, BorrowError> {
        self.try_reborrow_option(|r| Some(cell(r))).unwrap()
    }

    #[inline]
    pub fn reborrow<U>(&self, cell: impl FnOnce(&'b T) -> &'b TokenCell<U, Tk>) -> Ref<'b, U, Tk> {
        self.try_reborrow(cell).unwrap()
    }

    #[inline]
    pub fn try_reborrow_option<U>(
        &self,
        cell: impl FnOnce(&'b T) -> Option<&'b TokenCell<U, Tk>>,
    ) -> Option<Result<Ref<'b, U, Tk>, BorrowError>> {
        self.try_reborrow_iter(cell).next()
    }

    #[inline]
    pub fn reborrow_option<U>(
        &self,
        cell: impl FnOnce(&'b T) -> Option<&'b TokenCell<U, Tk>>,
    ) -> Option<Ref<'b, U, Tk>> {
        self.try_reborrow_option(cell).map(Result::unwrap)
    }

    #[inline]
    pub fn try_reborrow_iter<U: ?Sized + 'b, I: IntoIterator<Item = &'b TokenCell<U, Tk>>>(
        &self,
        cells: impl FnOnce(&'b T) -> I,
    ) -> impl Iterator<Item = Result<Ref<'b, U, Tk>, BorrowError>> {
        let token_id = self.token_id.clone();
        // SAFETY: the value is accessible as long as we hold our borrow.
        cells(unsafe { self.value.as_ref() })
            .into_iter()
            .map(move |cell| cell.get_ref(token_id.clone()))
    }

    #[inline]
    pub fn reborrow_iter<U: ?Sized + 'b, I: IntoIterator<Item = &'b TokenCell<U, Tk>>>(
        &self,
        cells: impl FnOnce(&'b T) -> I,
    ) -> impl Iterator<Item = Ref<'b, U, Tk>> {
        self.try_reborrow_iter(cells).map(Result::unwrap)
    }

    #[inline]
    pub fn map<U: ?Sized, F>(orig: Ref<'b, T, Tk>, f: F) -> Ref<'b, U, Tk>
    where
        F: FnOnce(&T) -> &U,
    {
        Ref::new(orig.token_id.clone(), f(&*orig))
    }

    #[inline]
    pub fn filter_map<U: ?Sized, F>(orig: Ref<'b, T, Tk>, f: F) -> Result<Ref<'b, U, Tk>, Self>
    where
        F: FnOnce(&T) -> Option<&U>,
    {
        match f(&*orig) {
            Some(value) => Ok(Ref::new(orig.token_id.clone(), value)),
            None => Err(orig),
        }
    }

    #[inline]
    pub fn map_split<U: ?Sized, V: ?Sized, F>(
        orig: Ref<'b, T, Tk>,
        f: F,
    ) -> (Ref<'b, U, Tk>, Ref<'b, V, Tk>)
    where
        F: FnOnce(&T) -> (&U, &V),
    {
        let (a, b) = f(&*orig);
        (
            Ref::new(orig.token_id.clone(), a),
            Ref::new(orig.token_id.clone(), b),
        )
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> Deref for Ref<'_, T, Tk> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_ref() }
    }
}

impl<T: ?Sized + fmt::Debug, Tk: Token + ?Sized> fmt::Debug for Ref<'_, T, Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ref")
            .field("token_id", &self.token_id)
            .field("value", self)
            .finish()
    }
}

impl<T: ?Sized + fmt::Display, Tk: Token + ?Sized> fmt::Display for Ref<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

/// A wrapper type for a mutably borrowed value from a [`TokenCell`].
///
/// It can be used to reborrow a component of the borrowed data.
pub struct RefMut<'b, T: ?Sized, Tk: Token + ?Sized> {
    token_id: Tk::Id,
    // NB: we use a pointer instead of `&'b mut T` to avoid `noalias` violations, because a
    // `RefMut` argument doesn't hold exclusivity for its whole scope, only until it drops.
    value: NonNull<T>,
    // `NonNull` is covariant over `T`, so we need to reintroduce invariance.
    marker: PhantomData<(&'b mut Tk, &'b mut T)>,
}

impl<'b, T: ?Sized, Tk: Token + ?Sized> RefMut<'b, T, Tk> {
    #[inline]
    pub(crate) fn new(token_id: Tk::Id, value: *mut T) -> Self {
        Self {
            token_id,
            value: NonNull::new(value).unwrap(),
            marker: PhantomData,
        }
    }

    pub fn into_mut(mut self) -> &'b mut T {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_mut() }
    }

    #[inline]
    pub fn token_id(&self) -> &Tk::Id {
        &self.token_id
    }

    #[inline]
    pub fn try_reborrow<U: ?Sized>(
        &self,
        cell: impl FnOnce(&'b T) -> &'b TokenCell<U, Tk>,
    ) -> Result<Ref<'b, U, Tk>, BorrowError> {
        self.try_reborrow_option(|r| Some(cell(r))).unwrap()
    }

    #[inline]
    pub fn reborrow<U: ?Sized>(
        &self,
        cell: impl FnOnce(&'b T) -> &'b TokenCell<U, Tk>,
    ) -> Ref<'b, U, Tk> {
        self.try_reborrow(cell).unwrap()
    }

    #[inline]
    pub fn try_reborrow_option<U: ?Sized>(
        &self,
        cell: impl FnOnce(&'b T) -> Option<&'b TokenCell<U, Tk>>,
    ) -> Option<Result<Ref<'b, U, Tk>, BorrowError>> {
        self.try_reborrow_iter(cell).next()
    }

    #[inline]
    pub fn reborrow_option<U: ?Sized>(
        &self,
        cell: impl FnOnce(&'b T) -> Option<&'b TokenCell<U, Tk>>,
    ) -> Option<Ref<'b, U, Tk>> {
        self.try_reborrow_option(cell).map(Result::unwrap)
    }

    #[inline]
    pub fn try_reborrow_iter<U: ?Sized + 'b, I: IntoIterator<Item = &'b TokenCell<U, Tk>>>(
        &self,
        cells: impl FnOnce(&'b T) -> I,
    ) -> impl Iterator<Item = Result<Ref<'b, U, Tk>, BorrowError>> {
        let token_id = self.token_id.clone();
        // SAFETY: the value is accessible as long as we hold our borrow.
        cells(unsafe { self.value.as_ref() })
            .into_iter()
            .map(move |cell| cell.get_ref(token_id.clone()))
    }

    #[inline]
    pub fn reborrow_iter<U: ?Sized + 'b, I: IntoIterator<Item = &'b TokenCell<U, Tk>>>(
        &self,
        cells: impl FnOnce(&'b T) -> I,
    ) -> impl Iterator<Item = Ref<'b, U, Tk>> {
        self.try_reborrow_iter(cells).map(Result::unwrap)
    }

    #[inline]
    pub fn try_reborrow_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&'b mut T) -> &'b TokenCell<U, Tk>,
    ) -> Result<RefMut<'b, U, Tk>, BorrowMutError> {
        self.try_reborrow_option_mut(|r| Some(cell(r))).unwrap()
    }

    #[inline]
    pub fn reborrow_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&'b mut T) -> &'b TokenCell<U, Tk>,
    ) -> RefMut<'b, U, Tk> {
        self.try_reborrow_mut(cell).unwrap()
    }

    #[inline]
    pub fn try_reborrow_option_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&'b mut T) -> Option<&'b TokenCell<U, Tk>>,
    ) -> Option<Result<RefMut<'b, U, Tk>, BorrowMutError>> {
        self.try_reborrow_iter_mut(cell).next()
    }

    #[inline]
    pub fn reborrow_option_mut<U: ?Sized>(
        &mut self,
        cell: impl FnOnce(&'b mut T) -> Option<&'b TokenCell<U, Tk>>,
    ) -> Option<RefMut<'b, U, Tk>> {
        self.try_reborrow_option_mut(cell).map(Result::unwrap)
    }

    #[inline]
    pub fn try_reborrow_iter_mut<U: ?Sized + 'b, I: IntoIterator<Item = &'b TokenCell<U, Tk>>>(
        &mut self,
        cells: impl FnOnce(&'b mut T) -> I,
    ) -> impl Iterator<Item = Result<RefMut<'b, U, Tk>, BorrowMutError>> {
        let token_id = self.token_id.clone();
        // SAFETY: the value is accessible as long as we hold our borrow.
        cells(unsafe { self.value.as_mut() })
            .into_iter()
            // SAFETY: uniqueness has been checked at `RefMut` creation, and token mutability
            // lifetime is shared by `RefMut`
            .map(move |cell| unsafe { cell.get_mut(token_id.clone()) })
    }

    #[inline]
    pub fn reborrow_iter_mut<U: ?Sized + 'b, I: IntoIterator<Item = &'b TokenCell<U, Tk>>>(
        &mut self,
        cells: impl FnOnce(&'b mut T) -> I,
    ) -> impl Iterator<Item = RefMut<'b, U, Tk>> {
        self.try_reborrow_iter_mut(cells).map(Result::unwrap)
    }

    #[inline]
    pub fn map<U: ?Sized, F>(mut orig: RefMut<'b, T, Tk>, f: F) -> RefMut<'b, U, Tk>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        RefMut::new(orig.token_id.clone(), f(&mut *orig))
    }

    #[inline]
    pub fn filter_map<U: ?Sized, F>(
        mut orig: RefMut<'b, T, Tk>,
        f: F,
    ) -> Result<RefMut<'b, U, Tk>, Self>
    where
        F: FnOnce(&mut T) -> Option<&mut U>,
    {
        // SAFETY: function holds onto an exclusive reference for the duration
        // of its call through `orig`, and the pointer is only de-referenced
        // inside the function call never allowing the exclusive reference to
        // escape.
        match f(unsafe { orig.value.as_mut() }) {
            Some(value) => Ok(RefMut::new(orig.token_id.clone(), value)),
            None => Err(orig),
        }
    }

    #[inline]
    pub fn map_split<U: ?Sized, V: ?Sized, F>(
        mut orig: RefMut<'b, T, Tk>,
        f: F,
    ) -> (RefMut<'b, U, Tk>, RefMut<'b, V, Tk>)
    where
        F: FnOnce(&mut T) -> (&mut U, &mut V),
    {
        let token_id = orig.token_id.clone();
        let (a, b) = f(&mut *orig);
        (RefMut::new(token_id.clone(), a), RefMut::new(token_id, b))
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> Deref for RefMut<'_, T, Tk> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_ref() }
    }
}

impl<T: ?Sized, Tk: Token + ?Sized> DerefMut for RefMut<'_, T, Tk> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: the value is accessible as long as we hold our borrow.
        unsafe { self.value.as_mut() }
    }
}

impl<T: ?Sized + fmt::Debug, Tk: Token + ?Sized + fmt::Debug> fmt::Debug for RefMut<'_, T, Tk>
where
    Tk::Id: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefMut")
            .field("token_id", &self.token_id)
            .field("value", self)
            .finish()
    }
}

impl<T: ?Sized + fmt::Display, Tk: Token + ?Sized> fmt::Display for RefMut<'_, T, Tk> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &**self)
    }
}

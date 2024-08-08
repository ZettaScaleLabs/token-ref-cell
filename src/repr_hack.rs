use core::cell::UnsafeCell;

use crate::token::PtrId;

mod private {
    /// Sealed trait marker.
    pub trait Sealed {}
}

#[doc(hidden)]
pub trait TokenId: Sized {
    #[doc(hidden)]
    type CellRepr: CellRepr<Self>;
}

#[doc(hidden)]
#[derive(Debug)]
pub struct ReprTransparent;
#[doc(hidden)]
#[derive(Debug)]
pub struct ReprRust;
#[doc(hidden)]
pub trait CellRepr<Id>: private::Sealed {
    type Cell<T: ?Sized>: ?Sized;
    fn new<T>(value: T, token_id: Id) -> Self::Cell<T>;
    fn inner<T>(cell: Self::Cell<T>) -> T;
    fn get<T: ?Sized>(cell: &Self::Cell<T>) -> *mut T;
    fn get_mut<T: ?Sized>(cell: &mut Self::Cell<T>) -> &mut T;
    fn token_id<T: ?Sized>(cell: &Self::Cell<T>) -> &Id;
    fn set_token_id<T: ?Sized>(cell: &mut Self::Cell<T>, token_id: Id);
}

impl private::Sealed for ReprTransparent {}
impl CellRepr<()> for ReprTransparent {
    type Cell<T: ?Sized> = UnsafeCell<T>;

    #[inline(always)]
    fn new<T>(value: T, _token_id: ()) -> Self::Cell<T> {
        UnsafeCell::new(value)
    }

    #[inline(always)]
    fn inner<T>(cell: Self::Cell<T>) -> T {
        cell.into_inner()
    }

    #[inline(always)]
    fn get<T: ?Sized>(cell: &Self::Cell<T>) -> *mut T {
        cell.get()
    }

    #[inline(always)]
    fn get_mut<T: ?Sized>(cell: &mut Self::Cell<T>) -> &mut T {
        cell.get_mut()
    }

    #[inline(always)]
    fn token_id<T: ?Sized>(_cell: &Self::Cell<T>) -> &() {
        &()
    }

    #[inline(always)]
    fn set_token_id<T: ?Sized>(_cell: &mut Self::Cell<T>, _token_id: ()) {}
}

impl private::Sealed for ReprRust {}
impl<Id> CellRepr<Id> for ReprRust {
    type Cell<T: ?Sized> = (Id, UnsafeCell<T>);

    #[inline(always)]
    fn new<T>(value: T, token_id: Id) -> Self::Cell<T> {
        (token_id, UnsafeCell::new(value))
    }

    #[inline(always)]
    fn inner<T>(cell: Self::Cell<T>) -> T {
        cell.1.into_inner()
    }

    #[inline(always)]
    fn get<T: ?Sized>(cell: &Self::Cell<T>) -> *mut T {
        cell.1.get()
    }

    #[inline(always)]
    fn get_mut<T: ?Sized>(cell: &mut Self::Cell<T>) -> &mut T {
        cell.1.get_mut()
    }

    #[inline(always)]
    fn token_id<T: ?Sized>(cell: &Self::Cell<T>) -> &Id {
        &cell.0
    }

    #[inline(always)]
    fn set_token_id<T: ?Sized>(cell: &mut Self::Cell<T>, token_id: Id) {
        cell.0 = token_id;
    }
}

impl TokenId for () {
    type CellRepr = ReprTransparent;
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_token_id {
    ($ty:ident $(<$($tt:tt)*)?) => {
        impl$(<$($tt)*)? $crate::token::TokenId for $crate::_remove_bounds!($ty $(<$($tt)*)?) {
            type CellRepr = ReprRust;
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _remove_bounds {
    ($ty:ident) => { $ty };
    ($ty:ident $(<$($tt:tt)*)?) => { $crate::_remove_bounds!(@ $ty{} $($($tt)*)?) };
    (@ $ty:ident{$($arg:tt,)*} $arg2:tt>) => { $ty<$($arg,)* $arg2> };
    (@ $ty:ident{$($arg:tt,)*} $arg2:tt, $($tt:tt)*) => { $crate::_remove_bounds!(@ $ty{$($arg,)* $arg2,} $($tt)*) };
    (@ $ty:ident{$($arg:tt,)*} $arg2:tt: $($tt:tt)*) => { $crate::_remove_bounds!(@@ $ty{$($arg,)* $arg2,} $($tt)*) };
    (@@ $ty:ident{$($arg:tt,)*} >) => { $ty<$($arg,)*> };
    (@@ $ty:ident{$($arg:tt,)*} , $($tt:tt)*) => { $crate::_remove_bounds!(@ $ty{$($arg,)*} $($tt)*) };
    (@@ $ty:ident{$($arg:tt,)*} $tt2:tt $($tt:tt)*) => { $crate::_remove_bounds!(@@ $ty{$($arg,)*} $($tt)*) };
}

impl_token_id!(usize);
impl_token_id!(PtrId<T: ?Sized>);

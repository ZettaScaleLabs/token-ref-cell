# token-ref-cell

This library provides `TokenRefCell`, an interior mutability cell, which uses an
external `Token` to synchronize its accesses.

Multiple token implementations are provided, the easiest to use being the
smart-pointer-based ones: every `Box<T>` can indeed be used as a token (as long as `T`
is not a ZST). The recommended token implementation is `BoxToken`, and it's the
default value of the generic parameter of `TokenRefCell`.

The runtime cost is very lightweight: only one pointer comparison for
`TokenRefCell::borrow`/`TokenRefCell::borrow_mut` when using `BoxToken`
(and zero-cost when using `singleton_token!`).
<br>
Because one token can be used with multiple cells, it's possible for example to use
a single rwlock wrapping a token to synchronize mutable access to multiple `Arc` data.

## Examples

```rust
use std::sync::{Arc, RwLock};
use token_ref_cell::{TokenRefCell, BoxToken};

let mut token = RwLock::new(BoxToken::new());
// Initialize a vector of arcs
let token_ref = token.read().unwrap();
let arc_vec = vec![Arc::new(TokenRefCell::new(0, &*token_ref)); 42];
drop(token_ref);
// Use only one rwlock to write to all the arcs
let mut token_mut = token.write().unwrap();
for cell in &arc_vec {
    *cell.borrow_mut(&mut *token_mut) += 1;
}
drop(token_mut)
```

*Based on https://crates.io/crates/token-cell concept.*
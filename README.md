# token-ref-cell

This library provides `TokenRefCell`, an interior mutability cell which uses an
external `Token` reference to synchronize its accesses.

Contrary to other standard cells like `RefCell`, `TokenRefCell` is `Sync` 
as long as its token is `Send + Sync`; it can thus be used in multithreaded programs.

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

## Why another crate?

Many crates based on the same principle exists: [`qcell`](https://crates.io/crates/qcell), [`ghost-cell`](https://crates.io/crates/ghost-cell), [`token-cell`](https://crates.io/crates/token-cell), [`singleton-cell`](https://crates.io/crates/singleton-cell), etc.

When I started writing `token-ref-cell`, I only knew `token-cell` one, as it was written by a guy previously working in the same company as me. But I wanted to take a new approach, notably designing an API closer than standard `RefCell` one, hence the name `TokenRefCell`. In fact, my goal was simple: to make the graph example compile, and for that I needed to enable "re-borrowing", i.e. reusing the token used in a mutable borrow to mutably borrow a "sub-cell". 

When I was satisfied with the result, before publishing it, I search for other similar crates, and I found the list above, and realized I'd reimplemented the same concept as a bunch of people, especially `qcell` which uses a `Box` for its cell token/owner. However, this fresh implementation still has a few specificities which makes it relevant:
- a familiar API close to `RefCell` one;
- a unified API with a single `Token` trait, contrary to `qcell` which provides four different cell types;
- a larger choice of token implementations, thanks to the simplicity of the `Token` trait: singleton types, smart-pointers, pinned/unpinned references, etc.;
- `no_std` implementation, compatible with custom allocators (doesn't require `alloc` crate, and doesn't require allocator at all using `RefToken` for example);
- re-borrowing.
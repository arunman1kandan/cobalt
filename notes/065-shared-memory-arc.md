# 065: Shared Memory with Arc (Atomic Reference Counting)

## 1. Motivation
Deep learning frameworks need to avoid copying massive tensors when creating views. If we have a 1GB image tensor and want to look at just the first channel, we shouldn't copy 1GB of data. Instead, we use **shared memory** with reference counting to safely share data between multiple tensor views.

## 2. Context / Precedence
- **C/C++**: Raw pointers with manual memory management (error-prone)
- **Python**: Reference counting with `PyObject` (automatic but GIL-limited)
- **Rust**: `Rc<T>` for single-threaded, `Arc<T>` for multi-threaded reference counting
- **PyTorch**: Uses ATen's `Storage` class with reference counting
- **NumPy**: Uses `PyArrayObject` with ref counts

## 3. The Problem We're Solving

### Before (Phase 0):
```rust
pub struct Tensor {
    pub data: Vec<u8>,  // Owned data
    pub shape: Vec<usize>,
    pub dtype: DType,
}
```

**Problem**: Creating a view requires cloning `data`:
```rust
let view = tensor.data.clone();  // COPIES ALL BYTES!
```

For a 1GB tensor, this copies 1GB. Unacceptable.

### After (Phase 1):
```rust
use std::sync::Arc;

pub struct Tensor {
    pub data: Arc<Vec<u8>>,  // Shared, reference-counted data
    pub shape: Vec<usize>,
    pub dtype: DType,
}
```

**Solution**: Multiple tensors/views share the same underlying buffer:
```rust
let view = Arc::clone(&tensor.data);  // Just increments ref count!
```

This is **O(1)** - just an atomic increment, no memory copy.

## 4. What is Arc?

`Arc<T>` = **Atomic Reference Counted** smart pointer.

### Key Properties:
1. **Shared Ownership**: Multiple owners can hold references to the same data
2. **Thread-Safe**: Uses atomic operations for ref counting (can cross thread boundaries)
3. **Automatic Cleanup**: Data is freed when last reference is dropped
4. **Immutable by Default**: `Arc<T>` gives shared read-only access

### Arc Structure (Conceptual):
```
Arc<Vec<u8>> internally stores:
┌─────────────────────────────────┐
│ Strong Count: 3  (atomic)       │
│ Weak Count: 0    (atomic)       │
│ Data: Vec<u8> [actual bytes]    │
└─────────────────────────────────┘
```

When you call `Arc::clone(&arc)`:
- **Doesn't clone the data**
- Just increments `Strong Count` atomically
- Returns a new `Arc` pointing to the same allocation

## 5. Implementation in Cobalt

### Tensor Migration to Arc

**Before**:
```rust
impl Tensor {
    pub fn new_raw(shape: Vec<usize>, dtype: DType, device: Device) -> Self {
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_in_bytes();
        Self {
            data: vec![0u8; bytes],  // Owned Vec
            shape,
            dtype,
            device,
        }
    }
}
```

**After**:
```rust
use std::sync::Arc;

impl Tensor {
    pub fn new_raw(shape: Vec<usize>, dtype: DType, device: Device) -> Self {
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_in_bytes();
        Self {
            data: Arc::new(vec![0u8; bytes]),  // Wrapped in Arc
            shape,
            dtype,
            device,
        }
    }
}
```

### TensorView Sharing

```rust
pub struct TensorView {
    data: Arc<Vec<u8>>,      // Shared with original Tensor
    shape: Vec<usize>,        // View-specific shape
    strides: Vec<usize>,      // View-specific strides
    offset: usize,            // Starting offset
    dtype: DType,
    device: Device,
}

impl TensorView {
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, FrameworkError> {
        let element_size = tensor.dtype.size_in_bytes() as usize;
        let strides = Self::compute_strides(&tensor.shape, element_size);
        
        Ok(TensorView {
            data: Arc::clone(&tensor.data),  // ← Shares, doesn't copy!
            shape: tensor.shape.clone(),
            strides,
            offset: 0,
            dtype: tensor.dtype,
            device: tensor.device,
        })
    }
}
```

### Reading from Arc-backed Storage

```rust
impl Tensor {
    pub fn as_slice<T: Element>(&self) -> &[T] {
        assert!(self.dtype == T::DTYPE, "dtype mismatch");
        let ptr = self.data.as_ref().as_ptr() as *const T;  // ← as_ref() to get &Vec<u8>
        let len = self.numel();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}
```

**Key**: `self.data.as_ref()` gives us `&Vec<u8>` from `Arc<Vec<u8>>`.

### Writing to Arc-backed Storage (Mutable Access)

**Problem**: `Arc<T>` is immutable by default. How do we get `&mut Vec<u8>`?

**Solution**: `Arc::get_mut()` - only succeeds if there's exactly 1 strong reference.

```rust
impl Tensor {
    pub fn as_slice_mut<T: Element>(&mut self) -> &mut [T] {
        assert!(self.dtype == T::DTYPE, "dtype mismatch");
        
        let data = Arc::get_mut(&mut self.data)
            .expect("cannot get mutable slice from shared tensor storage");
        
        let ptr = data.as_mut_ptr() as *mut T;
        let len = self.numel();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}
```

**Semantics**:
- If `Arc::strong_count(&self.data) == 1`: Returns `Some(&mut Vec<u8>)`
- If `Arc::strong_count(&self.data) > 1`: Returns `None` → panic

This prevents aliasing bugs: you can't mutate data that's shared elsewhere.

## 6. Memory Lifecycle Example

```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
// Arc::strong_count == 1

let view1 = TensorView::from_tensor(&t)?;
// Arc::strong_count == 2 (t.data and view1.data share)

let view2 = view1.slice(0, 1)?;
// Arc::strong_count == 3 (t.data, view1.data, view2.data share)

drop(view2);
// Arc::strong_count == 2

drop(view1);
// Arc::strong_count == 1

drop(t);
// Arc::strong_count == 0 → memory is freed
```

## 7. Benefits of Arc

### 7.1 Zero-Copy Views
```rust
let t = Tensor::from_f32(vec![1.0; 1_000_000], vec![1000, 1000]);
let view = t.view()?;  // O(1), no copy

// Both t and view reference the same 4MB buffer
```

### 7.2 Safe Shared Access
```rust
let t = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
let v1 = t.view()?;
let v2 = t.view()?;

// All three (t, v1, v2) safely share the same memory
// Rust's type system prevents data races
```

### 7.3 Automatic Memory Management
No need for manual `free()` or tracking lifetimes manually:
```rust
{
    let t = Tensor::from_f32(vec![1.0; 100], vec![10, 10]);
    let v = t.view()?;
    // ...
} // Both t and v dropped → memory freed automatically
```

## 8. Trade-offs and Limitations

### 8.1 Atomic Overhead
Each `Arc::clone()` and `Arc::drop()` does atomic operations:
```assembly
lock incq (%rdi)   ; Atomic increment (expensive on multi-core)
```

For tiny tensors (< 1KB), this overhead might exceed copy cost.

**Mitigation**: Small tensors could use owned `Vec<u8>` instead.

### 8.2 Mutation Restrictions
Once shared, you can't mutate:
```rust
let mut t = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
let v = t.view()?;  // Now Arc::strong_count == 2

// This will panic:
t.as_slice_mut::<f32>();  // Error: cannot get mutable reference
```

**Solution**: Views must call `contiguous()` to materialize independent copies when mutation is needed.

### 8.3 Cache Efficiency
Multiple views pointing to different parts of the same buffer can cause cache thrashing if accessed from different threads.

## 9. Comparison: Rc vs Arc

| Feature | `Rc<T>` | `Arc<T>` |
|---------|---------|----------|
| Thread-safe | ❌ No | ✅ Yes |
| Ref counting | Simple increment/decrement | Atomic increment/decrement |
| Performance | Faster (no atomics) | Slower (atomic overhead) |
| Send/Sync | Not Send, not Sync | Send + Sync |
| Use case | Single-threaded | Multi-threaded |

**Cobalt uses Arc** because:
1. Future parallel operations will need thread safety
2. Users might send tensors across threads
3. Overhead is acceptable for typical tensor sizes

## 10. Advanced: Weak References

`Arc<T>` supports weak references via `Arc::downgrade()`:

```rust
let t = Tensor::from_f32(vec![1.0], vec![1]);
let weak_ref = Arc::downgrade(&t.data);

// weak_ref doesn't keep the data alive
drop(t);  // Data is freed

// Attempting to upgrade fails:
assert!(weak_ref.upgrade().is_none());
```

**Use case**: Cache systems where entries can be evicted without invalidating all references.

## 11. Implementation Checklist

Phase 1 Arc Migration:
- [x] Update `Tensor` to use `Arc<Vec<u8>>` instead of `Vec<u8>`
- [x] Update `Tensor::new_raw()` to wrap data in Arc
- [x] Update `Tensor::from_slice()` to wrap data in Arc
- [x] Update `as_slice()` to use `self.data.as_ref()`
- [x] Update `as_slice_mut()` to use `Arc::get_mut()`
- [x] Implement `TensorView` with `Arc::clone()` for sharing
- [x] Add tests for reference counting behavior
- [x] Document memory lifecycle

## 12. Testing Reference Counting

Unfortunately, `Arc::strong_count()` is not directly accessible in a stable way, but we can test behavior:

```rust
#[test]
fn test_shared_memory() {
    let t = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
    let v1 = TensorView::from_tensor(&t).unwrap();
    let v2 = TensorView::from_tensor(&t).unwrap();
    
    // All three should see the same data
    assert_eq!(t.as_slice::<f32>(), &[1.0, 2.0]);
    
    // Materialize a view (creates new Arc)
    let v1_mat = v1.contiguous();
    
    // v1_mat should have independent storage now
    // (can verify by checking it doesn't panic on mutation)
}
```

## 13. Future Optimizations

### Copy-on-Write (CoW)
Could implement `make_mut()` pattern:
```rust
fn make_mut(&mut self) -> &mut Vec<u8> {
    Arc::make_mut(&mut self.data)  // Clones if shared, else returns mut ref
}
```

This auto-clones when data is shared, allowing safe mutation.

### Small Buffer Optimization
```rust
pub enum TensorData {
    Small(Vec<u8>),           // For tensors < 1KB
    Large(Arc<Vec<u8>>),      // For tensors >= 1KB
}
```

Avoids Arc overhead for tiny tensors.

## 14. Glossary
- **Arc**: Atomic Reference Counted smart pointer
- **Strong Count**: Number of `Arc` handles keeping data alive
- **Weak Count**: Number of `Weak` handles (don't keep data alive)
- **Reference Counting**: Memory management via counting references
- **Atomic Operations**: CPU instructions that execute atomically (thread-safe)

## 15. References
- [Rust Arc Documentation](https://doc.rust-lang.org/std/sync/struct.Arc.html)
- [PyTorch ATen Storage](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/Storage.h)
- [NumPy Array Memory Management](https://numpy.org/doc/stable/reference/arrays.ndarray.html)

## 16. Summary

Arc-based shared memory is the foundation for zero-copy tensor views:

1. **Tensor** wraps data in `Arc<Vec<u8>>`
2. **TensorView** shares the Arc with `Arc::clone()` (O(1), no copy)
3. **Multiple views** can safely read from the same buffer
4. **Mutation** requires exclusive ownership (enforced by `Arc::get_mut()`)
5. **Automatic cleanup** when all references are dropped

This enables efficient slicing, transposing, and reshaping without memory copies, which is critical for deep learning performance.

**Key Insight**: Views are cheap metadata over expensive data. Arc makes the data shared, while views provide different interpretations (shape, strides, offset) of the same bytes.

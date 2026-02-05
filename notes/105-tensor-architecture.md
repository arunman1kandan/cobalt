# 105: Tensor Architecture and Implementation

## 1. Motivation
This note provides a comprehensive overview of Cobalt's tensor system architecture, connecting all the pieces: type erasure, Arc-based memory, views, operations, and broadcasting. It serves as a map to understand how everything fits together.

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       COBALT TENSOR SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│   Tensor     │◄───────│  TensorView  │        │    DType     │
│              │        │              │        │   (Runtime)  │
│ Arc<Vec<u8>> │        │ Arc<Vec<u8>> │        │              │
│ shape        │        │ shape        │        │   FP32       │
│ dtype        │        │ strides      │        │   FP64       │
│ device       │        │ offset       │        │   INT32      │
│              │        │ dtype        │        │   ...16 types│
└──────┬───────┘        └──────┬───────┘        └──────┬───────┘
       │                       │                       │
       │                       │                       │
       ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      OPERATIONS LAYER                         │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Elementwise │  │    MatMul    │  │  Activations │       │
│  │  (Add, Mul)  │  │              │  │ (ReLU, Sfmx) │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Broadcasting System                      │    │
│  │  (NumPy-compatible shape expansion)                  │    │
│  └──────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      BACKEND LAYER                            │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │     CPU      │  │   GPU (fut)  │  │  SIMD Opts   │       │
│  │  (Scalar)    │  │              │  │  (AVX2/512)  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────────────────────────────────────────────────────┘
```

## 3. Core Tensor Structure

### 3.1 Design Philosophy

**Type Erasure**: Tensor is NOT generic over element type.
```rust
// NOT this:
pub struct Tensor<T> { ... }  // ✗ Would need Tensor<f32>, Tensor<i32>, etc.

// THIS:
pub struct Tensor {
    data: Arc<Vec<u8>>,  // ✓ Runtime polymorphism
    dtype: DType,
}
```

**Why?**
- Single `Tensor` type works with all dtypes
- Operations can switch dtype at runtime
- Mix FP32 and INT32 tensors in same vector
- Matches PyTorch/NumPy behavior

### 3.2 Field Breakdown

```rust
pub struct Tensor {
    pub data: Arc<Vec<u8>>,      // Shared memory (see 065-shared-memory-arc.md)
    pub shape: Vec<usize>,        // Logical dimensions [2, 3, 4]
    pub dtype: DType,             // Runtime type tag
    pub device: Device,           // CPU/GPU (future)
}
```

**`data: Arc<Vec<u8>>`**
- Type-erased bytes
- Reference-counted for view sharing
- See note [065-shared-memory-arc.md](065-shared-memory-arc.md)

**`shape: Vec<usize>`**
- Logical dimensions
- Product of shape = number of elements
- Empty tensor: `shape = [0]`
- Scalar: `shape = []`

**`dtype: DType`**
- Runtime type tag
- 16 supported types (see [100-dtype-system.md](100-dtype-system.md))
- Determines element size and interpretation

**`device: Device`**
- CPU or GPU allocation
- Future: enables heterogeneous computing

## 4. Element Trait

### 4.1 Connecting Rust Types to Runtime DTypes

```rust
pub trait Element: Copy + Clone + 'static + std::fmt::Debug + PartialEq {
    const DTYPE: DType;
}

impl Element for f32 { const DTYPE: DType = DType::FP32; }
impl Element for f64 { const DTYPE: DType = DType::FP64; }
impl Element for i32 { const DTYPE: DType = DType::INT32; }
// ... 16 total implementations
```

**Purpose**: Bridge compile-time types (`f32`) to runtime types (`DType::FP32`).

### 4.2 Type-Safe Access

```rust
impl Tensor {
    pub fn as_slice<T: Element>(&self) -> &[T] {
        assert!(self.dtype == T::DTYPE, "dtype mismatch");
        let ptr = self.data.as_ref().as_ptr() as *const T;
        let len = self.numel();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}
```

**Safety**: Runtime check prevents `as_slice::<f32>()` on an INT32 tensor.

## 5. Tensor Construction

### 5.1 From Typed Slice

```rust
impl Tensor {
    pub fn from_slice<T: Element>(data: &[T], shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert!(numel == data.len(), "shape mismatch");

        let size = std::mem::size_of::<T>();
        let ptr = data.as_ptr() as *const u8;
        let bytes_len = numel * size;
        
        let mut bytes = Vec::with_capacity(bytes_len);
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, bytes_len);
            bytes.extend_from_slice(slice);
        }

        Self {
            data: Arc::new(bytes),
            shape,
            dtype: T::DTYPE,
            device: Device::CPU,
        }
    }
}
```

**Example**:
```rust
let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
// dtype = FP32, data = [0x00, 0x00, 0x80, 0x3F, ...] (IEEE 754 bytes)
```

### 5.2 Raw Allocation

```rust
pub fn new_raw(shape: Vec<usize>, dtype: DType, device: Device) -> Self {
    let numel: usize = shape.iter().product();
    let bytes = numel * dtype.size_in_bytes();
    Self {
        data: Arc::new(vec![0u8; bytes]),
        shape,
        dtype,
        device,
    }
}
```

## 6. TensorView vs Tensor

| Feature | Tensor | TensorView |
|---------|--------|------------|
| **Storage** | Owns Arc (but data is shared) | Shares Arc |
| **Strides** | Implicit (contiguous) | Explicit (can be non-contiguous) |
| **Offset** | Always 0 | Can be >0 (for slicing) |
| **Use Case** | Result of operations | Slicing, transposing |
| **Materialize** | Already materialized | May need `contiguous()` |

**Design**: Tensor is always contiguous; TensorView handles non-contiguous layouts.

## 7. Operations Architecture

### 7.1 Operation Dispatch

```rust
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, FrameworkError> {
        crate::ops::elementwise::add(self, other)
    }
}
```

**Delegation**: Tensor methods delegate to `ops` module.

### 7.2 Dtype Dispatch (Runtime Polymorphism)

```rust
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    match a.dtype {
        DType::FP32 => add_typed::<f32>(a, b),
        DType::FP64 => add_typed::<f64>(a, b),
        DType::INT32 => add_typed::<i32>(a, b),
        // ... 16 cases
        _ => Err(FrameworkError::UnsupportedDType(format!("{:?}", a.dtype))),
    }
}

fn add_typed<T: Element>(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    // Monomorphized code for specific type T
    let a_data = a.as_slice::<T>();
    let b_data = b.as_slice::<T>();
    // ...
}
```

**Pattern**: Match on `dtype` → call generic function → compiler monomorphizes.

### 7.3 Broadcasting Integration

```rust
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    let out_shape = crate::broadcast::broadcast_shapes(&a.shape, &b.shape)?;
    
    // Create output tensor
    let mut out = Tensor::new_raw(out_shape.clone(), a.dtype, a.device);
    
    match a.dtype {
        DType::FP32 => {
            let a_slice = a.as_slice::<f32>();
            let b_slice = b.as_slice::<f32>();
            let out_slice = out.as_slice_mut::<f32>();
            
            // Broadcast-aware iteration
            ops::cpu::add::add_fp32_broadcast(a_slice, &a.shape, b_slice, &b.shape, out_slice, &out_shape);
        }
        // ...
    }
    
    Ok(out)
}
```

See [070-broadcasting.md](070-broadcasting.md) for details.

## 8. Memory Lifecycle

### 8.1 Creation
```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3]);
// Arc::strong_count = 1
```

### 8.2 View Sharing
```rust
let v1 = TensorView::from_tensor(&t)?;
// Arc::strong_count = 2 (t.data and v1.data share)

let v2 = v1.slice(0, 2)?;
// Arc::strong_count = 3
```

### 8.3 Materialization
```rust
let mat = v2.contiguous();
// Arc::strong_count of original data = 3
// mat.data is NEW Arc with strong_count = 1
```

### 8.4 Cleanup
```rust
drop(v2);  // strong_count = 2
drop(v1);  // strong_count = 1
drop(t);   // strong_count = 0 → memory freed
```

## 9. Error Handling

### 9.1 Error Types

```rust
pub enum FrameworkError {
    ShapeMismatch { expected: String, got: String },
    BroadcastMismatch { a: Vec<usize>, b: Vec<usize> },
    DTypeMismatch,
    DeviceMismatch,
    UnsupportedOp(&'static str),
    UnsupportedDType(String),
    IndexOutOfBounds { index: usize, length: usize },
    InvalidDimension { dim: usize, rank: usize },
    DuplicateAxis { axis: usize },
}
```

### 9.2 Error Propagation

```rust
pub fn matmul(&self, other: &Tensor) -> Result<Tensor, FrameworkError> {
    if self.dtype != other.dtype {
        return Err(FrameworkError::DTypeMismatch);
    }
    
    if self.shape.len() != 2 || other.shape.len() != 2 {
        return Err(FrameworkError::UnsupportedOp("matmul requires 2D tensors"));
    }
    
    if self.shape[1] != other.shape[0] {
        return Err(FrameworkError::ShapeMismatch {
            expected: format!("Inner dimensions to match, got {} and {}", self.shape[1], other.shape[0]),
            got: format!("{:?} @ {:?}", self.shape, other.shape),
        });
    }
    
    // Operation proceeds...
}
```

**Philosophy**: Validate early, return descriptive errors.

## 10. Backend Organization

### 10.1 Directory Structure
```
src/backend/
├── mod.rs              # Backend trait definitions
├── ops.rs              # Operation dispatch
└── cpu/
    ├── mod.rs          # CPU backend registration
    ├── add.rs          # Add implementation
    ├── mul.rs          # Mul implementation
    ├── matmul.rs       # MatMul implementation
    ├── relu.rs         # ReLU implementation
    ├── softmax.rs      # Softmax implementation
    ├── add_scalar.rs   # Scalar addition (broadcasting)
    ├── add_avx2.rs     # SIMD optimizations (AVX2)
    ├── add_avx512.rs   # SIMD optimizations (AVX512)
    └── isa.rs          # ISA detection
```

### 10.2 CPU Backend Pattern

Each operation file:
```rust
// add.rs
pub fn add_fp32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { add_avx512::add_fp32_avx512(a, b, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { add_avx2::add_fp32_avx2(a, b, out) };
        }
    }
    
    // Fallback: scalar
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}
```

**Dispatch hierarchy**: AVX512 > AVX2 > Scalar

### 10.3 SIMD Optimization Example

```rust
// add_avx2.rs
#[target_feature(enable = "avx2")]
pub unsafe fn add_fp32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    
    let len = a.len();
    let chunks = len / 8;  // Process 8 floats at a time
    
    for i in 0..chunks {
        let idx = i * 8;
        
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out.as_mut_ptr().add(idx), vc);
    }
    
    // Handle remainder
    for i in (chunks * 8)..len {
        out[i] = a[i] + b[i];
    }
}
```

**Performance**: 8x throughput for aligned data.

## 11. Testing Architecture

### 11.1 Test Organization

```rust
#[cfg(test)]
mod tests {
    mod tensor_tests { /* Basic tensor creation */ }
    mod add_tests { /* Addition with broadcasting */ }
    mod mul_tests { /* Multiplication */ }
    mod matmul_tests { /* Matrix multiplication */ }
    mod relu_tests { /* ReLU activation */ }
    mod softmax_tests { /* Softmax */ }
    mod broadcast_tests { /* Broadcasting rules */ }
    mod error_tests { /* Error conditions */ }
}
```

### 11.2 Test Coverage

- **Unit tests**: Each operation, each dtype
- **Integration tests**: Broadcasting + operations
- **Error tests**: Invalid shapes, dtype mismatches
- **Edge cases**: Empty tensors, scalars, large dimensions

### 11.3 Test Example

```rust
#[test]
fn test_add_broadcast_2d() {
    let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::from_f32(vec![10.0, 20.0, 30.0], vec![3]);
    
    let c = a.add(&b).unwrap();
    
    assert_eq!(c.shape, vec![2, 3]);
    assert_eq!(c.as_f32_slice(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}
```

## 12. Performance Considerations

### 12.1 Memory Allocation

**Minimize allocations**:
- Views share Arc, no allocation
- Operations allocate output once
- Reuse buffers when possible (future: in-place ops)

### 12.2 Cache Locality

**Contiguous access patterns**:
- Operations iterate sequentially when possible
- Transpose → materialize if used multiple times
- SIMD requires contiguous data

### 12.3 SIMD Utilization

**FP32 operations**:
- AVX2: 8 floats/instruction
- AVX512: 16 floats/instruction
- Throughput: 2000+ million ops/sec

### 12.4 Broadcasting Overhead

**Small shapes**: Broadcasting check is negligible
**Large shapes**: Strided iteration slower than contiguous

## 13. Future Extensions

### 13.1 GPU Backend
```rust
pub enum Device {
    CPU,
    CUDA(usize),  // GPU device index
}
```

- CUDA kernel dispatch
- Host-device transfers
- Asynchronous execution

### 13.2 Autograd
```rust
pub struct Tensor {
    data: Arc<Vec<u8>>,
    shape: Vec<usize>,
    dtype: DType,
    device: Device,
    grad: Option<Box<Tensor>>,  // Gradient storage
    requires_grad: bool,
}
```

- Tape-based autodiff
- Backward pass computation
- Gradient accumulation

### 13.3 Lazy Execution
```rust
pub enum TensorData {
    Materialized(Arc<Vec<u8>>),
    Deferred(Box<dyn ComputeGraph>),
}
```

- Build computation graph
- Fuse operations
- Execute on demand

## 14. Design Principles

### 14.1 Explicit Over Implicit
- Errors are explicit (Result types)
- Dtype mismatches caught at runtime
- Broadcasting rules follow NumPy exactly

### 14.2 Zero-Cost Abstractions Where Possible
- Views are O(1) metadata
- Generic functions monomorphize to specialized code
- SIMD when supported

### 14.3 Safety First, Performance Second
- Type-safe access via Element trait
- Arc prevents use-after-free
- Bounds checking in debug builds

### 14.4 Documentation-Driven Development
- Every concept has a detailed note
- Examples accompany each feature
- Architecture is transparent

## 15. Integration Example

Complete workflow using all components:

```rust
// 1. Create tensor (type erasure + Arc allocation)
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

// 2. Create view (Arc sharing)
let v = TensorView::from_tensor(&t)?;

// 3. Slice (metadata update)
let sliced = v.slice(0, 2)?;

// 4. Transpose (stride swap)
let transposed = sliced.transpose(0, 1)?;

// 5. Materialize (copy to contiguous)
let mat = transposed.contiguous();

// 6. Element-wise operation (broadcasting + dtype dispatch + SIMD)
let bias = Tensor::from_f32(vec![10.0], vec![1]);
let result = mat.add(&bias)?;

// 7. Activation (dtype dispatch)
let activated = result.relu()?;
```

## 16. Summary

Cobalt's tensor architecture combines:

1. **Type Erasure**: Single `Tensor` type via `Arc<Vec<u8>>` + `DType`
2. **Shared Memory**: Arc-based reference counting for views
3. **Views**: Zero-copy slicing and transposition
4. **Broadcasting**: NumPy-compatible shape expansion
5. **Dtype Dispatch**: Runtime polymorphism via match statements
6. **SIMD**: AVX2/AVX512 optimizations for FP32
7. **Error Handling**: Explicit Result types with descriptive errors
8. **Testing**: Comprehensive coverage across dtypes and operations

**Result**: A flexible, performant tensor system suitable for deep learning experimentation and education.

See individual notes for deep dives:
- [010-tensors.md](010-tensors.md) - Tensor fundamentals
- [015-datatypes.md](015-datatypes.md) - Type system
- [065-shared-memory-arc.md](065-shared-memory-arc.md) - Memory management
- [085-view-implementation.md](085-view-implementation.md) - Views
- [095-contiguity-materialization.md](095-contiguity-materialization.md) - Contiguity
- [100-dtype-system.md](100-dtype-system.md) - 16-dtype support

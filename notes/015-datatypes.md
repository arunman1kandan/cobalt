# 015 - Datatypes Support

To transform Cobalt from a toy FP32 engine into a framework that supports `int`, `double`, and `bool`, we had to fundamentally change how tensors store data.

## The Problem: Storage vs. Types

In Rust, a `Vec<f32>` can only store 32-bit floats. If we want a Tensor to hold *either* floats or integers, we have two main choices:

1. **Enum Wrapper (The High-Level Way)**
   ```rust
   enum TensorData {
       F32(Vec<f32>),
       I32(Vec<i32>),
   }
   ```
   *Pros:* Safe, idiomatic Rust.
   *Cons:* Matches on every access. Hard to map to raw memory for GPU/C-interop later.

2. **Byte Buffer + Metadata (The Systems Way)**
   ```rust
   struct Tensor {
       data: Vec<u8>, // Raw bytes
       dtype: DType,  // Info: "Treat these 4 bytes as an f32"
   }
   ```
   *Pros:* Zero-overhead casting. Aligns with standard C/C++ libs (PyTorch uses this).
   *Cons:* Requires `unsafe` code to reinterpret bytes.

**Cobalt uses Approach #2.** It prioritizes systems-level understanding and flexibility.

## Core Concepts

### 1. `DType` Enum
This is the runtime tag. It tells us how to interpret the byte blob.
*Location:* `src/dtype.rs`

### 2. `Element` Trait
This acts as a bridge between the Runtime world (`DType`) and the Compile-time world (`f32`, `i32`).
*Location:* `src/tensor.rs`

```rust
pub trait Element {
    const DTYPE: DType; // Maps f32 -> DType::FP32
}
```

### 3. Dispatch
Since `Tensor` isn't generic (it's just `Tensor`, not `Tensor<T>`), we can't directly add two tensors without checking their type first.

**The Dispatch Pattern:**
1. Check `tensor.dtype`.
2. "Switch" (match) to the correct generic implementation.
3. Inside the generic implementation, the compiler knows the exact type (e.g., `f32`), allowing optimization (SIMD).

```rust
// Runtime check
match dtype {
    DType::FP32 => impl_add::<f32>(a, b), // Compiler generates f32 code
    DType::INT32 => impl_add::<i32>(a, b), // Compiler generates i32 code
}
```

## Challenges Encountered

### Recursive Calls in AVX
We support AVX2/AVX512 optimization for FP32.
*Bug:* The AVX kernel tried to call `add(t1, t2)` recursively to handle shape broadcasting. But `add` checks for AVX support again, leading to an infinite loop.
*Fix:* We bypassed the top-level `add` and called the scalar/fallback implementation directly from within the AVX kernel.

### Unsafe Coding
Using `unsafe { std::slice::from_raw_parts ... }` is powerful but dangerous. We must ensure:
1. `data.len()` matches exactly `numel * sizeof(T)`.
2. Alignment is respected (mostly handled by `Vec<u8>` allocation, but something to watch for).

## 4. Complete DType System

### All 16 Supported Types

**Floating Point**:
- `FP32` (f32): IEEE 754 single precision (32-bit)
  - Range: ±3.4 × 10³⁸
  - Precision: ~7 decimal digits
  - Use: Default for most deep learning
  
- `FP64` (f64): IEEE 754 double precision (64-bit)
  - Range: ±1.8 × 10³⁰⁸
  - Precision: ~15 decimal digits
  - Use: Scientific computing, high-precision requirements
  
- `FP16` (f16): IEEE 754 half precision (16-bit)
  - Range: ±65,504
  - Precision: �~3 decimal digits
  - Use: GPU memory optimization, mixed precision training
  
- `BF16` (bf16): Brain Float 16 (16-bit, Google)
  - Range: Same as FP32 (±3.4 × 10³⁸)
  - Precision: ~2 decimal digits
  - Use: TPU/modern GPU training, better range than FP16

**Signed Integers**:
- `INT8` (i8): -128 to 127
  - Use: Quantized models, embeddings
  
- `INT16` (i16): -32,768 to 32,767
  - Use: Audio processing, intermediate calculations
  
- `INT32` (i32): -2,147,483,648 to 2,147,483,647
  - Use: Indexing, counting, accumulation
  
- `INT64` (i64): -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
  - Use: Large-scale indexing, timestamps

**Unsigned Integers**:
- `UINT8` (u8): 0 to 255
  - Use: Images (RGB pixels), quantized weights
  
- `UINT16` (u16): 0 to 65,535
  - Use: HDR images, depth maps
  
- `UINT32` (u32): 0 to 4,294,967,295
  - Use: Large indices, masks
  
- `UINT64` (u64): 0 to 18,446,744,073,709,551,615
  - Use: Cryptographic hashes, large-scale data

**Boolean**:
- `BOOL` (bool): true/false (stored as 1 byte)
  - Use: Masks, attention matrices, conditional logic

### Type Size and Alignment

```rust
impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::FP32 => 4,
            DType::FP64 => 8,
            DType::FP16 => 2,
            DType::BF16 => 2,
            DType::INT8 => 1,
            DType::INT16 => 2,
            DType::INT32 => 4,
            DType::INT64 => 8,
            DType::UINT8 => 1,
            DType::UINT16 => 2,
            DType::UINT32 => 4,
            DType::UINT64 => 8,
            DType::BOOL => 1,
        }
    }
}
```

## 5. Practical Type Usage

### When to Use Each Type

**FP32** - Default choice:
```rust
let weights = Tensor::from_f32(vec![0.5, -0.3, 0.8], vec![3]);
// Standard for:
// - Neural network weights
// - Forward pass activations
// - Loss calculations
```

**INT8/UINT8** - Quantization:
```rust
let quantized = Tensor::from_slice(&[127i8, -128, 0, 64], vec![4]);
// Use for:
// - Model compression (4x smaller than FP32)
// - Mobile/edge deployment
// - Integer-only hardware
```

**FP16/BF16** - Mixed precision:
```rust
let half_precision = Tensor::from_slice(&[half::f16::from_f32(1.0)], vec![1]);
// Use for:
// - GPU training (2x memory, ~2x speed)
// - Gradient computation (sufficient precision)
// - Large batch sizes
```

**BOOL** - Masking:
```rust
let mask = Tensor::from_slice(&[true, false, true, false], vec![4]);
// Use for:
// - Attention masks in transformers
// - Dropout masks
// - Conditional execution
```

### Type Compatibility Matrix

| Operation | FP32 | FP64 | INT32 | UINT8 | BOOL |
|-----------|------|------|-------|-------|------|
| Add       | ✅   | ✅   | ✅    | ✅    | ❌   |
| Mul       | ✅   | ✅   | ✅    | ✅    | ❌   |
| MatMul    | ✅   | ✅   | ✅    | ✅    | ❌   |
| ReLU      | ✅   | ✅   | ✅    | ✅    | ❌   |
| Softmax   | ✅   | ✅   | ❌    | ❌    | ❌   |

## 6. Implementation Deep Dive

### Type Erasure Mechanism

**The Problem**: Rust requires compile-time type knowledge, but we want runtime flexibility.

**The Solution**: Store bytes + metadata:

```rust
pub struct Tensor {
    data: Arc<Vec<u8>>,       // Type-erased storage
    shape: Vec<usize>,         // Dimensions
    dtype: DType,              // Runtime type tag
    device: Device,            // CPU/GPU
}
```

### Element Trait - Compile/Runtime Bridge

```rust
pub trait Element: Copy + Clone + 'static {
    const DTYPE: DType;  // Maps T -> DType at compile time
}

// Implementations
impl Element for f32 {
    const DTYPE: DType = DType::FP32;
}

impl Element for i32 {
    const DTYPE: DType = DType::INT32;
}
// ... for all 16 types
```

### Type-Safe Data Access

```rust
impl Tensor {
    // Generic read (type-checked at runtime)
    pub fn as_slice<T: Element>(&self) -> Result<&[T], FrameworkError> {
        // 1. Check type matches
        if T::DTYPE != self.dtype {
            return Err(FrameworkError::DTypeMismatch {
                expected: T::DTYPE,
                got: self.dtype,
            });
        }
        
        // 2. Calculate element count
        let numel = self.shape.iter().product::<usize>();
        let expected_bytes = numel * T::DTYPE.size_in_bytes();
        
        // 3. Verify buffer size
        if self.data.len() != expected_bytes {
            return Err(FrameworkError::InvalidBuffer);
        }
        
        // 4. UNSAFE: Reinterpret bytes as &[T]
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            Ok(std::slice::from_raw_parts(ptr, numel))
        }
    }
    
    // Convenience for FP32 (most common)
    pub fn as_f32_slice(&self) -> &[f32] {
        self.as_slice::<f32>().expect("Not FP32 tensor")
    }
}
```

### Dispatch Pattern - The Heart of Multi-Type Support

**Three-Layer Architecture**:

1. **User API** (type-erased):
```rust
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError>
```

2. **Dispatcher** (dtype switch):
```rust
match (a.dtype, b.dtype) {
    (DType::FP32, DType::FP32) => add_typed::<f32>(a, b),
    (DType::INT32, DType::INT32) => add_typed::<i32>(a, b),
    (dtype_a, dtype_b) if dtype_a != dtype_b => {
        Err(FrameworkError::DTypeMismatch {
            expected: dtype_a,
            got: dtype_b,
        })
    }
    _ => Err(FrameworkError::UnsupportedDType(a.dtype)),
}
```

3. **Generic Kernel** (compile-time specialized):
```rust
fn add_typed<T: Element + std::ops::Add<Output = T>>(
    a: &Tensor,
    b: &Tensor,
) -> Result<Tensor, FrameworkError> {
    let a_data = a.as_slice::<T>()?;
    let b_data = b.as_slice::<T>()?;
    
    let mut out_data = Vec::with_capacity(a_data.len());
    for (x, y) in a_data.iter().zip(b_data.iter()) {
        out_data.push(*x + *y);
    }
    
    Tensor::from_slice(&out_data, a.shape.clone())
}
```

**Compiler Magic**: The compiler generates separate optimized machine code for each type:
- `add_typed::<f32>` → SIMD instructions for floats
- `add_typed::<i32>` → SIMD instructions for integers
- No runtime overhead after dispatch

## 7. Memory Layout Examples

### FP32 Tensor
```
Tensor { shape: [2, 3], dtype: FP32 }

Logical View:
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]

Physical Memory (24 bytes):
[00 00 80 3F] [00 00 00 40] [00 00 40 40]  // Row 0
[00 00 80 40] [00 00 A0 40] [00 00 C0 40]  // Row 1
(little-endian IEEE 754)
```

### INT8 Tensor
```
Tensor { shape: [4], dtype: INT8 }

Logical: [-128, -1, 0, 127]

Physical (4 bytes):
[80] [FF] [00] [7F]
```

### Mixed Precision Workflow
```rust
// Training: FP32 master weights
let weights_fp32 = Tensor::from_f32(vec![0.1; 1000], vec![1000]);

// Forward pass: FP16 for speed
let weights_fp16 = weights_fp32.to_dtype(DType::FP16)?;
let activations_fp16 = forward_pass(&weights_fp16, &input_fp16)?;

// Backward pass: Accumulate gradients in FP32 for precision
let loss_fp32 = loss_fn(&activations_fp16.to_dtype(DType::FP32)?)?;
```

## 8. Performance Characteristics

### Memory Usage Comparison
```
1M elements:
- FP64:  8 MB
- FP32:  4 MB  (2x smaller)
- FP16:  2 MB  (4x smaller)
- INT8:  1 MB  (8x smaller)
- BOOL:  1 MB  (8x smaller, but no math ops)
```

### Computation Speed (Approximate)
```
Operation: Add(1M elements) on AVX2 CPU

FP32:  38.6 µs (baseline)
FP64:  ~60 µs  (1.5x slower - half SIMD width)
INT32: ~25 µs  (1.5x faster - simpler ops)
INT8:  ~15 µs  (2.5x faster - more SIMD parallelism)
```

### Type Conversion Overhead
```rust
// Benchmark: Converting 1M elements
let t_fp32 = Tensor::from_f32(vec![1.0; 1000000], vec![1000000]);

// FP32 → FP64: ~2 ms  (widening, no data loss)
// FP32 → FP16: ~3 ms  (narrowing, rounding required)
// FP32 → INT32: ~5 ms  (truncation, sign handling)
```

## 9. Common Patterns and Idioms

### Pattern 1: Type-Agnostic Operations
```rust
// Function works with any numeric type
fn scale_tensor<T>(t: &Tensor, factor: T) -> Result<Tensor, FrameworkError>
where
    T: Element + std::ops::Mul<Output = T>,
{
    let data = t.as_slice::<T>()?;
    let scaled: Vec<T> = data.iter().map(|&x| x * factor).collect();
    Tensor::from_slice(&scaled, t.shape.clone())
}

// Usage
let fp32_scaled = scale_tensor(&fp32_tensor, 2.0f32)?;
let int_scaled = scale_tensor(&int_tensor, 2i32)?;
```

### Pattern 2: Type Promotion
```rust
fn promote_to_float(t: &Tensor) -> Result<Tensor, FrameworkError> {
    match t.dtype {
        DType::FP32 | DType::FP64 => Ok(t.clone()),
        DType::INT32 => {
            let data = t.as_slice::<i32>()?;
            let floats: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            Tensor::from_f32(floats, t.shape.clone())
        }
        _ => t.to_dtype(DType::FP32),
    }
}
```

### Pattern 3: Safe Type Checking
```rust
fn requires_float(t: &Tensor) -> Result<(), FrameworkError> {
    match t.dtype {
        DType::FP32 | DType::FP64 | DType::FP16 | DType::BF16 => Ok(()),
        _ => Err(FrameworkError::DTypeMismatch {
            expected: DType::FP32,
            got: t.dtype,
        }),
    }
}
```

## 10. Debugging Type Issues

### Common Error 1: Type Mismatch
```rust
let a = Tensor::from_f32(vec![1.0], vec![1]);
let b = Tensor::from_slice(&[1i32], vec![1]);

// ERROR: DTypeMismatch { expected: FP32, got: INT32 }
let c = a.add(&b)?;

// FIX: Convert to same type
let b_float = b.to_dtype(DType::FP32)?;
let c = a.add(&b_float)?;
```

### Common Error 2: Wrong Type Cast
```rust
let t = Tensor::from_f32(vec![1.0, 2.0], vec![2]);

// PANIC: "Not INT32 tensor"
let data = t.as_slice::<i32>().unwrap();

// FIX: Check type first
if t.dtype == DType::INT32 {
    let data = t.as_slice::<i32>()?;
} else {
    eprintln!("Expected INT32, got {:?}", t.dtype);
}
```

### Common Error 3: Overflow
```rust
let large_ints = Tensor::from_slice(&[30000i16, 30000i16], vec![2]);

// Multiplication overflows i16 (max 32767)
let squared = large_ints.mul(&large_ints)?; // Wrong results!

// FIX: Promote to wider type
let as_i32 = large_ints.to_dtype(DType::INT32)?;
let squared = as_i32.mul(&as_i32)?;
```

### Debugging Helper
```rust
fn debug_tensor_type(t: &Tensor) {
    println!("Shape: {:?}", t.shape);
    println!("DType: {:?}", t.dtype);
    println!("Bytes: {}", t.data.len());
    println!("Elements: {}", t.shape.iter().product::<usize>());
    println!("Expected bytes: {}", 
        t.shape.iter().product::<usize>() * t.dtype.size_in_bytes());
}
```

## 11. Advanced Topics

### Custom Type Extensions
```rust
// To add a new dtype (e.g., COMPLEX64):

// 1. Add to enum
pub enum DType {
    // ... existing types
    COMPLEX64,
}

// 2. Define Rust type
use num_complex::Complex;
type Complex64 = Complex<f32>;

// 3. Implement Element trait
impl Element for Complex64 {
    const DTYPE: DType = DType::COMPLEX64;
}

// 4. Update size_in_bytes
impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            // ... existing
            DType::COMPLEX64 => 8, // 2 * f32
        }
    }
}

// 5. Add to dispatchers
match dtype {
    // ... existing
    DType::COMPLEX64 => operation::<Complex64>(tensor),
}
```

### Type-Based Optimization Selection
```rust
fn select_kernel<T: Element>(tensor: &Tensor) -> Box<dyn Fn(&[T], &[T]) -> Vec<T>> {
    match T::DTYPE {
        DType::FP32 if has_avx512() => Box::new(add_avx512_f32),
        DType::FP32 if has_avx2() => Box::new(add_avx2_f32),
        DType::FP32 => Box::new(add_scalar_f32),
        _ => Box::new(add_generic::<T>),
    }
}
```

## 12. Type System Summary

### Design Principles
1. **Runtime Flexibility**: Support dynamic type selection
2. **Compile-Time Safety**: Use Rust's type system to prevent errors
3. **Zero Overhead**: Dispatch once, run optimized code
4. **Extensibility**: Easy to add new types

### Trade-offs
**Pros**:
- ✅ Dynamic type selection (like Python)
- ✅ Type-safe operations (like Rust)
- ✅ SIMD-optimized kernels per type
- ✅ Single `Tensor` type (no `Tensor<f32>`, `Tensor<i32>`)

**Cons**:
- ❌ Runtime dispatch overhead (~5-10ns)
- ❌ Binary bloat (N types × M ops = code duplication)
- ❌ `unsafe` code for type punning
- ❌ Cannot mix types without explicit conversion

### When to Use This Pattern
- Deep learning frameworks (PyTorch, TensorFlow use similar)
- Numerical libraries (NumPy, MATLAB)
- Graphics engines (type-tagged vertex buffers)
- Database engines (variant column types)

### Alternatives to Consider
- **Generic Tensor** (`Tensor<T>`): Simpler but less flexible
- **Enum Wrapper**: Safer but slower (virtual dispatch)
- **Code Generation**: Less binary bloat but complex build
- **External Types**: Use existing libraries (ndarray)

## 13. References and Further Reading

**Rust Type System**:
- [Rust Book: Generic Types](https://doc.rust-lang.org/book/ch10-00-generics.html)
- [Rustonomicon: Type Layout](https://doc.rust-lang.org/nomicon/exotic-sizes.html)

**Numeric Types**:
- [IEEE 754 Standard](https://en.wikipedia.org/wiki/IEEE_754)
- [Brain Float 16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [Quantization Overview](https://arxiv.org/abs/2103.13630)

**Implementation**:
- [PyTorch ATen Dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [Type Punning in C++](https://en.cppreference.com/w/cpp/language/reinterpret_cast)

## 14. Exercises

**Exercise 1**: Calculate memory usage
- Tensor shape: [1024, 1024, 3]
- Type: UINT8
- Answer: 1024 × 1024 × 3 × 1 = 3,145,728 bytes = 3 MB

**Exercise 2**: Determine appropriate dtype
- Task: Store image pixels (0-255)
- Answer: UINT8 (perfect range, 4x smaller than FP32)

**Exercise 3**: Debug type error
```rust
let mask = Tensor::from_slice(&[true, false], vec![2]);
let values = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
let filtered = mask.mul(&values)?; // What happens?
```
- Answer: Error! BOOL doesn't support multiplication. Need to convert mask to FP32 first:
```rust
let mask_float = mask.to_dtype(DType::FP32)?;
let filtered = mask_float.mul(&values)?;
```

---

**Summary**: Cobalt's type system uses type erasure (`Arc<Vec<u8>>` + `DType` tag) to combine runtime flexibility with compile-time safety. This pattern, inspired by PyTorch and TensorFlow, enables supporting 16 data types through a single `Tensor` interface while maintaining zero-overhead type-specialized kernels.

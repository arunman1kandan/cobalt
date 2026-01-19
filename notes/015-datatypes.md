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

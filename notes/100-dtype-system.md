# Data Type System

## Overview

Cobalt supports a comprehensive set of data types (dtypes) for tensors, allowing flexibility across different use cases from deep learning to scientific computing. The type system is designed to be:

- **Type-safe**: Compile-time safety with runtime dispatch
- **Efficient**: Zero-copy operations where possible
- **Extensible**: Easy to add new types

## Supported Data Types

### Floating-Point Types

#### FP32 (32-bit Float)
- **Size**: 4 bytes
- **Range**: ±3.4 × 10³⁸
- **Precision**: ~7 decimal digits
- **Use Cases**: 
  - Default choice for deep learning
  - Good balance of range and precision
  - SIMD optimizations available (AVX2/AVX512)
- **Performance**: ✅ Excellent (SIMD accelerated)

```rust
let tensor = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3]);
```

#### FP64 (64-bit Double)
- **Size**: 8 bytes
- **Range**: ±1.7 × 10³⁰⁸
- **Precision**: ~15 decimal digits
- **Use Cases**:
  - Scientific computing requiring high precision
  - Accumulation of many values
  - Numerical stability critical applications
- **Performance**: ⚠️ Good (no SIMD yet)

```rust
let tensor = Tensor::from_slice(&[1.0f64, 2.0, 3.0], vec![3]);
```

#### FP16 (16-bit Half Precision)
- **Size**: 2 bytes
- **Range**: ±65,504
- **Precision**: ~3 decimal digits
- **Use Cases**:
  - Memory-constrained environments
  - Mobile/edge deployment
  - Mixed-precision training
  - 2x memory reduction vs FP32
- **Performance**: ⚠️ Software emulation (slower than FP32)
- **Note**: Uses `half::f16` from the `half` crate

```rust
use half::f16;
let data = [f16::from_f32(1.0), f16::from_f32(2.0)];
let tensor = Tensor::from_slice(&data, vec![2]);
```

#### BF16 (Brain Float 16)
- **Size**: 2 bytes
- **Range**: Same as FP32 (±3.4 × 10³⁸)
- **Precision**: ~2-3 decimal digits
- **Use Cases**:
  - Deep learning (developed by Google Brain)
  - Better than FP16 for training (wider range)
  - Mixed-precision training
  - TPU/AI accelerator optimized
- **Performance**: ⚠️ Software emulation
- **Advantages over FP16**: Easier to use (drop-in FP32 replacement), better gradient stability

```rust
use half::bf16;
let data = [bf16::from_f32(10.0), bf16::from_f32(20.0)];
let tensor = Tensor::from_slice(&data, vec![2]);
```

### Integer Types

#### INT32 (32-bit Signed Integer)
- **Size**: 4 bytes
- **Range**: -2,147,483,648 to 2,147,483,647
- **Use Cases**:
  - Indices and counters
  - Quantized neural networks
  - Integer computations
  - Labels and categorical data
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[1, 2, 3, 4], vec![4]);
```

#### INT64 (64-bit Signed Integer)
- **Size**: 8 bytes
- **Range**: -9.2 × 10¹⁸ to 9.2 × 10¹⁸
- **Use Cases**:
  - Large datasets (indices > 2B)
  - High-precision integer math
  - Timestamps
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[100i64, 200, 300], vec![3]);
```

#### INT16 (16-bit Signed Integer)
- **Size**: 2 bytes
- **Range**: -32,768 to 32,767
- **Use Cases**:
  - Memory-efficient quantization
  - Audio samples
  - Compact storage
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[100i16, 200, 300], vec![3]);
```

#### INT8 (8-bit Signed Integer)
- **Size**: 1 byte
- **Range**: -128 to 127
- **Use Cases**:
  - 8-bit quantization
  - Extreme memory efficiency
  - Model compression
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[-10i8, 0, 10], vec![3]);
```

### Unsigned Integer Types

#### UINT8 (8-bit Unsigned)
- **Size**: 1 byte
- **Range**: 0 to 255
- **Use Cases**:
  - Image data (pixels)
  - Probability bins
  - Raw byte data
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[128u8, 64, 32], vec![3]);
```

#### UINT16 (16-bit Unsigned)
- **Size**: 2 bytes
- **Range**: 0 to 65,535
- **Use Cases**:
  - High-resolution images
  - Larger index spaces
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[1000u16, 2000], vec![2]);
```

#### UINT32 (32-bit Unsigned)
- **Size**: 4 bytes
- **Range**: 0 to 4,294,967,295
- **Use Cases**:
  - Large positive indices
  - Hash values
  - Large dataset indices
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[1000000u32, 2000000], vec![2]);
```

#### UINT64 (64-bit Unsigned)
- **Size**: 8 bytes
- **Range**: 0 to 18.4 × 10¹⁸
- **Use Cases**:
  - Extremely large datasets
  - Unique identifiers
- **Performance**: ✅ Excellent

```rust
let tensor = Tensor::from_slice(&[1000000000u64, 2000000000], vec![2]);
```

## Type Selection Guide

### Deep Learning Training
- **FP32**: Default choice, best balance
- **BF16**: Mixed-precision training, memory savings
- **FP16**: Mobile deployment, edge devices

### Inference
- **FP32**: High accuracy
- **BF16/FP16**: Memory-constrained
- **INT8**: Quantized models, maximum efficiency

### Scientific Computing
- **FP64**: High precision requirements
- **FP32**: Standard precision

### Data Storage
- **UINT8**: Images, raw data
- **INT32**: Indices, labels
- **INT16**: Compressed data

## Memory Comparison

| Type | Size | Elements in 1GB | Speedup vs FP32 |
|------|------|-----------------|-----------------|
| FP64 | 8B   | 134M           | 0.5x            |
| FP32 | 4B   | 268M           | 1.0x (baseline) |
| INT32| 4B   | 268M           | 1.0x            |
| BF16 | 2B   | 536M           | 2.0x*           |
| FP16 | 2B   | 536M           | 2.0x*           |
| INT16| 2B   | 536M           | 2.0x            |
| UINT8| 1B   | 1.07B          | 4.0x            |
| INT8 | 1B   | 1.07B          | 4.0x            |

*Software emulation, actual speedup may vary

## Operations Support Matrix

| Operation | Float Types | Integer Types | Notes |
|-----------|-------------|---------------|-------|
| Add       | ✅          | ✅            | All types |
| Mul       | ✅          | ✅            | All types |
| MatMul    | ✅          | ❌            | FP32, FP64, FP16, BF16 |
| ReLU      | ✅          | ✅            | All numeric types |
| Softmax   | ✅          | ❌            | Float types only |

## Implementation Details

### Type Erasure
Cobalt uses a type-erased `Vec<u8>` storage with runtime dtype dispatch:

```rust
pub struct Tensor {
    data: Vec<u8>,
    dtype: DType,
    shape: Vec<usize>,
    // ...
}
```

### Element Trait
Types must implement the `Element` trait:

```rust
pub trait Element: Copy + Send + Sync + 'static {}
```

### Runtime Dispatch
Operations use match statements to dispatch to the correct implementation:

```rust
match self.dtype {
    DType::FP32 => /* FP32 implementation */,
    DType::FP64 => /* FP64 implementation */,
    DType::FP16 => /* FP16 implementation */,
    // ... more variants
}
```

## Future Enhancements

### Planned
- [ ] FP8 (E4M3, E5M2) - Ultra-low precision
- [ ] INT4 - 4-bit quantization
- [ ] BOOL - Binary tensors
- [ ] Complex types (Complex64, Complex128)

### Optimizations
- [ ] Hardware FP16 operations (when available)
- [ ] SIMD for FP64, INT32, INT64
- [ ] GPU support via CUDA/ROCm
- [ ] Vulkan compute shaders

## Best Practices

1. **Start with FP32**: Use it as your default, optimize later
2. **Profile before switching**: Measure actual memory/performance impact
3. **Use BF16 over FP16 for training**: Better gradient stability
4. **INT8 for deployment**: Maximum efficiency for inference
5. **Mind the conversions**: Type conversions can be expensive

## References

- [IEEE 754 Floating Point Standard](https://standards.ieee.org/ieee/754/6210/)
- [BFloat16 Paper (Google Brain)](https://arxiv.org/abs/1905.12322)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

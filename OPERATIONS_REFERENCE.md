# Cobalt Operations × Data Types Support Matrix

## ✅ Fully Implemented

| Operation | FP32 | FP64 | INT32 | INT64 | UINT8 | Notes |
|-----------|------|------|-------|-------|-------|-------|
| **Add** | ✅ | ✅ | ✅ | ✅ | ✅ | Full broadcasting, AVX2/512 for FP32 |
| **Mul** | ✅ | ✅ | ✅ | ✅ | ❌ | Full broadcasting support |
| **MatMul** | ✅ | ✅ | ❌ | ❌ | ❌ | 2D only, O(n³) naive impl |
| **ReLU** | ✅ | ✅ | ✅ | ✅ | ❌ | max(0, x) activation |
| **Softmax** | ✅ | ✅ | ❌ | ❌ | ❌ | Numerically stable, FP only |

## Operation Details

### Elementwise Operations

#### Add (Addition)
```rust
let a = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
let b = Tensor::from_f32(vec![3.0, 4.0], vec![2]);
let c = a.add(&b)?; // [4.0, 6.0]
```

**Supported Types**: FP32, FP64, INT32, INT64, UINT8  
**Broadcasting**: ✅ Full NumPy-style broadcasting  
**SIMD**: ✅ AVX2 (8x) and AVX512 (16x) for FP32  
**Performance**: ~5-10 GFLOPS for large tensors (CPU)

#### Mul (Multiplication)
```rust
let a = Tensor::from_f32(vec![2.0, 3.0], vec![2]);
let b = Tensor::from_f32(vec![4.0, 5.0], vec![2]);
let c = a.mul(&b)?; // [8.0, 15.0]
```

**Supported Types**: FP32, FP64, INT32, INT64  
**Broadcasting**: ✅ Full NumPy-style broadcasting  
**SIMD**: ❌ Not yet (planned for Phase 1)

### Matrix Operations

#### MatMul (Matrix Multiplication)
```rust
let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
let c = a.matmul(&b)?; // [19.0, 22.0, 43.0, 50.0]
```

**Supported Types**: FP32, FP64  
**Dimensions**: 2D only ([M, K] @ [K, N] -> [M, N])  
**Algorithm**: Naive O(n³) triple loop  
**Performance**: ~0.1-1 GFLOPS (CPU, unoptimized)  
**Planned**: Batched matmul, tiled/blocked optimization, BLAS integration

### Activation Functions

#### ReLU (Rectified Linear Unit)
```rust
let x = Tensor::from_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
let y = x.relu()?; // [0.0, 0.0, 0.0, 1.0, 2.0]
```

**Formula**: `f(x) = max(0, x)`  
**Supported Types**: FP32, FP64, INT32, INT64  
**SIMD**: ❌ Not yet (can use SIMD max instructions)  
**Use Case**: Hidden layer activations in neural networks

#### Softmax
```rust
let logits = Tensor::from_f32(vec![2.0, 1.0, 0.1], vec![3]);
let probs = logits.softmax()?; // [0.659, 0.242, 0.099]
```

**Formula**: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`  
**Supported Types**: FP32, FP64 (floating-point only)  
**Dimension**: Operates on last dimension  
**Numerical Stability**: ✅ Uses log-sum-exp trick (subtracts max)  
**Properties**: Output sums to 1.0, all values in [0, 1]  
**Use Case**: Classification layer output, attention weights

## Broadcasting Support

All elementwise operations (Add, Mul) support NumPy-style broadcasting:

```rust
// Scalar broadcasting
let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
let b = Tensor::from_f32(vec![10.0], vec![1]);
let c = a.add(&b)?; // [11.0, 12.0, 13.0, 14.0]

// Vector to matrix broadcasting
let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
let b = Tensor::from_f32(vec![10.0, 20.0, 30.0], vec![3]);
let c = a.add(&b)?; // Each row gets the vector added
```

**Rules**:
1. Right-align shapes
2. Dimensions are compatible if equal or one is 1
3. Output shape = max of each dimension
4. Implementation uses stride manipulation (zero-copy)

## Performance Characteristics

### Memory Bandwidth
- **Add/Mul**: Memory-bound (~10-50 GB/s on modern CPUs)
- **MatMul**: Compute-bound for large matrices (arithmetic intensity grows)
- **Activations**: Memory-bound (simple operations)

### SIMD Acceleration
- **FP32 Add**: Up to 16x speedup with AVX512
- **Other ops**: Not yet vectorized (planned)

### Cache Efficiency
- **MatMul**: Poor (Phase 0 naive implementation)
- **Elementwise**: Good (linear access patterns)

## Error Handling

All operations return `Result<Tensor, FrameworkError>` with these possible errors:

```rust
pub enum FrameworkError {
    ShapeMismatch { expected: String, got: String },
    BroadcastMismatch { a: Vec<usize>, b: Vec<usize> },
    DTypeMismatch,
    DeviceMismatch,
    UnsupportedOp(&'static str),
    UnsupportedDType(String),
}
```

### Examples:
```rust
// Dtype mismatch
let a_fp32 = Tensor::from_f32(vec![1.0], vec![1]);
let b_int32 = Tensor::from_slice(&[1], vec![1]);
a_fp32.add(&b_int32)?; // Error: DTypeMismatch

// Shape mismatch (no valid broadcast)
let a = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
let b = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3]);
a.add(&b)?; // Error: BroadcastMismatch

// Unsupported operation
let a_int = Tensor::from_slice(&[1, 2], vec![2]);
let b_int = Tensor::from_slice(&[3, 4], vec![2]);
a_int.matmul(&b_int)?; // Error: UnsupportedDType (matmul needs FP)
```

## Usage Examples by Data Type

### FP32 (Most Common)
```rust
let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

// All operations supported
a.add(&b)?;     // ✅
a.mul(&b)?;     // ✅
a.matmul(&b)?;  // ✅
a.relu()?;      // ✅
a.softmax()?;   // ✅
```

### FP64 (High Precision)
```rust
let a = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::from_slice(&[5.0f64, 6.0, 7.0, 8.0], vec![2, 2]);

a.add(&b)?;     // ✅
a.mul(&b)?;     // ✅
a.matmul(&b)?;  // ✅
a.relu()?;      // ✅
a.softmax()?;   // ✅
```

### INT32 (Integer Operations)
```rust
let a = Tensor::from_slice(&[1, 2, 3, 4], vec![2, 2]);
let b = Tensor::from_slice(&[10, 20, 30, 40], vec![2, 2]);

a.add(&b)?;     // ✅
a.mul(&b)?;     // ✅
a.matmul(&b)?;  // ❌ Unsupported
a.relu()?;      // ✅
a.softmax()?;   // ❌ Requires floating-point
```

### INT64 (Large Integers)
```rust
let a = Tensor::from_slice(&[1i64, 2, 3, 4], vec![2, 2]);
let b = Tensor::from_slice(&[10i64, 20, 30, 40], vec![2, 2]);

a.add(&b)?;     // ✅
a.mul(&b)?;     // ✅
a.matmul(&b)?;  // ❌ Unsupported
a.relu()?;      // ✅
a.softmax()?;   // ❌ Requires floating-point
```

## Testing Coverage

Each operation has comprehensive tests covering:
- ✅ Basic functionality
- ✅ All supported data types
- ✅ Broadcasting scenarios
- ✅ Edge cases (zeros, negatives, large values)
- ✅ Error conditions
- ✅ Numerical stability (for softmax)

**Total Tests**: 32  
**Pass Rate**: 100%

## Future Additions (Phase 1+)

### Planned Operations
- Sub (subtraction)
- Div (division)
- Pow (power)
- Exp, Log, Sqrt
- Sum, Mean, Max, Min (reductions)
- GELU, Sigmoid, Tanh (activations)
- Conv2D, Pooling
- Transpose, Reshape, Slice

### Planned Optimizations
- SIMD for all elementwise ops
- Tiled/blocked matmul
- Parallel operations (Rayon)
- BLAS integration (OpenBLAS, MKL)
- GPU support (CUDA)

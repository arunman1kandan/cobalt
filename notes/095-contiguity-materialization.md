# 095: Contiguity and Materialization in Depth

## 1. Motivation
Understanding **contiguity** is critical for high-performance tensor operations. A non-contiguous tensor can't use SIMD, can't be efficiently copied to GPU, and can't be reshaped. This note dives deep into what contiguity means, why it matters, and how materialization works.

## 2. What is Contiguity?

### 2.1 Informal Definition
A tensor is **contiguous** if its elements are stored sequentially in memory in **row-major (C-style)** order, without gaps or stride jumps.

### 2.2 Formal Definition
A tensor with shape `S = [s₀, s₁, ..., s_{n-1}]` is contiguous if and only if:

```
strides[i] = element_size * ∏_{j=i+1}^{n-1} s_j

For all i ∈ [0, n-1]
```

In other words:
- `strides[n-1]` (innermost) = `element_size`
- `strides[n-2]` = `element_size * s_{n-1}`
- `strides[n-3]` = `element_size * s_{n-1} * s_{n-2}`
- ...

### 2.3 Visual Example

**Contiguous [2, 3] FP32 tensor**:
```
Memory Layout:
[1.0][2.0][3.0][4.0][5.0][6.0]
 ↑    ↑    ↑    ↑    ↑    ↑
 0    4    8    12   16   20  (byte offsets)

Shape: [2, 3]
Strides: [12, 4]
Element size: 4

Check:
strides[1] = 4 = element_size ✓
strides[0] = 12 = 4 * 3 ✓
Contiguous: YES
```

**Non-Contiguous (Transposed)**:
```
Same memory, but interpreted as [3, 2]:
Shape: [3, 2]
Strides: [4, 12]  ← Swapped from [12, 4]

Check:
strides[1] = 12 ≠ element_size (4) ✗
Contiguous: NO
```

## 3. Why Contiguity Matters

### 3.1 SIMD Vectorization
Modern CPUs use **SIMD** (Single Instruction Multiple Data) to process 4-16 elements simultaneously.

**Contiguous data** (AVX2 can load 256 bits = 8 floats at once):
```rust
// Efficient SIMD load
let ptr = data.as_ptr();
unsafe {
    let vec = _mm256_loadu_ps(ptr);  // Load 8 floats in one instruction
}
```

**Non-contiguous data** (elements scattered):
```rust
// Must gather elements individually (slow)
for i in 0..8 {
    let offset = base_offset + i * stride;
    result[i] = data[offset];
}
```

**Performance difference**: SIMD can be **8-16x faster** for contiguous data.

### 3.2 Cache Efficiency
CPUs load data in **cache lines** (typically 64 bytes).

**Contiguous**: Sequential access maximizes cache hits
```
Cache line loads: [0-63], [64-127], [128-191], ...
Each load gets 16 floats (64 bytes / 4 bytes)
```

**Non-contiguous** (after transpose with large stride):
```
Cache line loads: [0-63], [12-75], [24-87], ...
Each load might only use 1-2 floats before jumping to next stride
Cache utilization: <20%
```

### 3.3 GPU Transfers
GPUs expect contiguous buffers for CUDA `cudaMemcpy`:
```cpp
// Requires contiguous memory
cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
```

Non-contiguous tensors must be copied element-by-element (very slow).

### 3.4 Reshape Requirement
Reshaping reinterprets memory without copying:

**Contiguous** → Reshape is free:
```
[6] → [2, 3] or [3, 2] or [1, 6] (all valid, just change strides)
```

**Non-contiguous** → Can't reshape safely:
```
Transposed [3, 2] with strides [4, 12] can't become [6]
(Element order would be wrong: [1, 4, 2, 5, 3, 6] instead of [1, 2, 3, 4, 5, 6])
```

## 4. Detecting Contiguity

### 4.1 Algorithm
```rust
pub fn is_contiguous(&self) -> bool {
    let element_size = self.dtype.size_in_bytes() as usize;
    let mut expected_stride = element_size;
    
    // Check from innermost to outermost dimension
    for (i, &dim) in self.shape.iter().enumerate().rev() {
        if dim == 0 {
            continue;  // Skip empty dimensions
        }
        
        if self.strides[i] != expected_stride {
            return false;  // Mismatch → non-contiguous
        }
        
        expected_stride *= dim;  // Update for next dimension
    }
    
    true
}
```

### 4.2 Edge Cases

**Empty tensor**:
```rust
Shape: [0, 3]
Strides: anything (doesn't matter, no elements)
Contiguous: YES (by convention)
```

**Scalar (0-D tensor)**:
```rust
Shape: []
Strides: []
Contiguous: YES
```

**1-D tensor**:
```rust
Shape: [10]
Strides: [4]
Contiguous: Always YES (can't be non-contiguous in 1D)
```

### 4.3 Contiguity Examples

| Shape | Strides | Element Size | Contiguous? | Reason |
|-------|---------|--------------|-------------|--------|
| `[3, 4]` | `[16, 4]` | 4 | ✅ YES | `16 = 4*4`, `4 = 4` |
| `[3, 4]` | `[4, 16]` | 4 | ❌ NO | `4 ≠ 4*4` (transposed) |
| `[2, 3, 4]` | `[48, 16, 4]` | 4 | ✅ YES | `4=4, 16=4*4, 48=16*3` |
| `[2, 3, 4]` | `[16, 48, 4]` | 4 | ❌ NO | Permuted |
| `[5]` | `[4]` | 4 | ✅ YES | 1-D always contiguous |
| `[5]` | `[8]` | 4 | ❌ NO | Strided (skipping elements) |

## 5. Operations That Preserve/Break Contiguity

### 5.1 Preserve Contiguity
- **Creating new tensor**: Always contiguous
- **Slicing last dimension**: Usually contiguous
- **Squeeze/Unsqueeze**: Contiguity preserved

**Example (contiguity preserved)**:
```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
// Contiguous: [12, 4]

let sliced = t.slice_dim(1, 0, 2)?;  // Slice columns [0:2]
// Shape: [2, 2], Strides: [12, 4]
// Still contiguous! ✓
```

### 5.2 Break Contiguity
- **Transpose**: Almost always breaks contiguity
- **Permute**: Usually breaks contiguity
- **Slicing non-last dimension**: Can break contiguity
- **Stride-based indexing**: Breaks contiguity

**Example (contiguity broken)**:
```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
// Contiguous: [12, 4]

let transposed = t.transpose_view(0, 1)?;
// Shape: [3, 2], Strides: [4, 12]
// Non-contiguous! ✗
```

### 5.3 Slicing First Dimension
```rust
let t = Tensor::from_f32(vec![1.0; 12], vec![3, 4]);
// Contiguous: strides [16, 4]

let sliced = t.slice(1, 3)?;  // Rows [1:3]
// Shape: [2, 4], Strides: [16, 4], Offset: 16
// Contiguous: YES (just different starting point)
```

**Key**: Slicing adjusts `offset` but doesn't change strides, so as long as original was contiguous and we're not skipping rows, result stays contiguous.

## 6. Materialization: Making Non-Contiguous Contiguous

### 6.1 The Problem
```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let transposed = t.transpose_view(0, 1)?;

// transposed is non-contiguous
// We need contiguous for reshape or SIMD operations
```

### 6.2 The Solution: Copy with Stride-Aware Iteration

```rust
pub fn contiguous(&self) -> Tensor {
    let element_size = self.dtype.size_in_bytes() as usize;
    let expected_strides = Self::compute_strides(&self.shape, element_size);
    let total_bytes = self.numel() * element_size;

    // Fast path: already contiguous, just share memory
    if self.is_contiguous()
        && self.offset == 0
        && self.strides == expected_strides
        && total_bytes == self.data.len()
    {
        return Tensor {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        };
    }

    // Slow path: must copy
    let bytes = self.materialize_bytes();
    Tensor {
        data: Arc::new(bytes),
        shape: self.shape.clone(),
        dtype: self.dtype,
        device: self.device,
    }
}
```

### 6.3 Materialization Algorithm

```rust
fn materialize_bytes(&self) -> Vec<u8> {
    let element_size = self.dtype.size_in_bytes() as usize;
    let numel = self.numel();
    let mut out = vec![0u8; numel * element_size];

    for linear in 0..numel {
        // Convert linear index to multi-dimensional index
        let mut remaining = linear;
        let mut byte_offset = self.offset;

        // Compute strided offset
        for dim in (0..self.shape.len()).rev() {
            let size = self.shape[dim];
            let idx = remaining % size;
            remaining /= size;
            byte_offset += idx * self.strides[dim];
        }

        // Copy this element
        let src = &self.data[byte_offset..byte_offset + element_size];
        let dst_start = linear * element_size;
        out[dst_start..dst_start + element_size].copy_from_slice(src);
    }

    out
}
```

### 6.4 Materialization Example

**Non-Contiguous Transpose**:
```
Original: [1.0, 2.0, 3.0, 4.0]
Shape: [2, 2], Strides: [8, 4]

[1.0, 2.0]
[3.0, 4.0]

Transposed View:
Shape: [2, 2], Strides: [4, 8]

Access pattern (row-major on transposed view):
[0,0]: offset = 0 + 0*4 + 0*8 = 0  → 1.0
[0,1]: offset = 0 + 0*4 + 1*8 = 8  → 3.0
[1,0]: offset = 0 + 1*4 + 0*8 = 4  → 2.0
[1,1]: offset = 0 + 1*4 + 1*8 = 12 → 4.0

Materialized (contiguous):
[1.0, 3.0, 2.0, 4.0]
Shape: [2, 2], Strides: [8, 4]
```

### 6.5 Performance Cost

**Time complexity**: O(numel)
- Must iterate over every element
- For 1M element tensor: ~1-10ms depending on cache

**Space complexity**: O(numel * element_size)
- Allocates entirely new buffer
- For 1M FP32: 4MB allocation

**Trade-off**:
- **Avoid if possible**: Keep views, work with strides
- **Necessary for**: Reshape, SIMD, GPU transfer, external APIs

## 7. Advanced: Partial Contiguity

Some tensors are contiguous in **some dimensions** but not others.

**Example**:
```rust
Shape: [10, 20, 30]
Strides: [600, 4, 120]  // Permuted middle dimensions

Contiguous check:
dim[2]: 4 ≠ 4 ✓
dim[1]: 120 ≠ 4*30 ✗  Non-contiguous!
```

**Opportunity**: Could optimize operations that only iterate over contiguous dimensions.

## 8. Comparison with Other Frameworks

### PyTorch
```python
t = torch.tensor([[1, 2], [3, 4]])
t_t = t.t()  # Transpose (non-contiguous)

t_t.is_contiguous()  # False
t_t_c = t_t.contiguous()  # Materialize
t_t_c.is_contiguous()  # True
```

### NumPy
```python
arr = np.array([[1, 2], [3, 4]])
arr_t = arr.T  # Transpose (non-contiguous)

arr_t.flags['C_CONTIGUOUS']  # False
arr_t_c = np.ascontiguousarray(arr_t)  # Materialize
arr_t_c.flags['C_CONTIGUOUS']  # True
```

### TensorFlow
TensorFlow primarily uses contiguous tensors; transpose creates a new copy (less sophisticated).

## 9. When to Materialize

### ✅ Materialize When:
1. **Reshaping needed**: Non-contiguous can't reshape
2. **SIMD critical**: Performance-critical loops
3. **GPU transfer**: CUDA expects contiguous
4. **External FFI**: C libraries expect contiguous
5. **Multiple uses**: Cheaper to copy once than stride-iterate many times

### ❌ Avoid Materializing When:
1. **One-time use**: Just iterate with strides
2. **View-only workflows**: Slicing → transpose → slice
3. **Memory constrained**: Doubling memory usage unacceptable
4. **Lazy evaluation**: Defer until absolutely necessary

## 10. Optimization Strategies

### 10.1 Lazy Materialization
Don't materialize until an operation requires it:
```rust
impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, FrameworkError> {
        // Could work with views directly via strided iteration
        // OR materialize if SIMD is critical
        
        if !self.is_contiguous() && self.numel() > CONTIGUOUS_THRESHOLD {
            let self_contig = self.contiguous();
            // Use self_contig for SIMD
        }
        // ...
    }
}
```

### 10.2 In-Place Contiguity Check
```rust
// Instead of:
let materialized = view.contiguous();
do_operation(&materialized);

// Do:
if view.is_contiguous() {
    do_operation_fast(&view);
} else {
    let temp = view.contiguous();
    do_operation_fast(&temp);
}
```

### 10.3 Multi-threaded Materialization
For huge tensors, parallelize copying:
```rust
use rayon::prelude::*;

fn materialize_parallel(&self) -> Vec<u8> {
    let element_size = self.dtype.size_in_bytes() as usize;
    let numel = self.numel();
    let mut out = vec![0u8; numel * element_size];
    
    (0..numel).into_par_iter().for_each(|linear| {
        // Compute offset and copy element
        // (parallel iteration over elements)
    });
    
    out
}
```

## 11. Testing Contiguity

```rust
#[test]
fn test_contiguity() {
    let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let v = TensorView::from_tensor(&t).unwrap();
    
    assert!(v.is_contiguous());
    
    let transposed = v.transpose(0, 1).unwrap();
    assert!(!transposed.is_contiguous());
    
    let materialized = transposed.contiguous();
    assert!(TensorView::from_tensor(&materialized).unwrap().is_contiguous());
}

#[test]
fn test_materialization_correctness() {
    let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let v = TensorView::from_tensor(&t).unwrap();
    let transposed = v.transpose(0, 1).unwrap();
    
    let mat = transposed.contiguous();
    assert_eq!(mat.as_f32_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
```

## 12. Common Pitfalls

### Pitfall 1: Assuming All Views Are Non-Contiguous
```rust
// Wrong assumption:
let sliced = tensor.slice(0, 10)?;
let mat = sliced.contiguous();  // Unnecessary copy if already contiguous
```

**Fix**: Check first:
```rust
let mat = if sliced.is_contiguous() {
    Tensor { data: Arc::clone(&sliced.data), ... }
} else {
    sliced.contiguous()
};
```

### Pitfall 2: Repeated Materialization
```rust
// Bad: Materializes twice
let t1 = view.contiguous();
let t2 = view.contiguous();  // Another copy!
```

**Fix**: Reuse:
```rust
let mat = view.contiguous();
let t1 = &mat;
let t2 = &mat;
```

### Pitfall 3: Materializing Large Tensors Unnecessarily
```rust
// 1GB tensor
let huge = Tensor::from_f32(vec![0.0; 250_000_000], vec![10000, 25000]);
let transposed = huge.transpose_view(0, 1)?;
let mat = transposed.contiguous();  // Allocates another 1GB!
```

**Fix**: Use strided ops when possible.

## 13. Glossary
- **Contiguous**: Elements stored sequentially in row-major order
- **Materialization**: Copying non-contiguous view into contiguous buffer
- **Row-major**: Rightmost index varies fastest (C-style)
- **Column-major**: Leftmost index varies fastest (Fortran-style)
- **Fast path**: Already contiguous, just share Arc
- **Slow path**: Must copy with stride-aware iteration

## 14. Summary

**Contiguity** determines whether tensor data is laid out sequentially in memory:

1. **Detection**: Check if `strides[i] == element_size * ∏(shape[i+1..])`
2. **Importance**: SIMD, cache efficiency, GPU transfer, reshape
3. **Preservation**: Most views break contiguity (transpose, permute)
4. **Materialization**: Copy non-contiguous → contiguous when needed
5. **Trade-off**: Memory/time cost vs. performance benefits

**Best Practice**: Defer materialization until an operation truly requires contiguity. Most deep learning ops can work with strided data.

The `contiguous()` method is your escape hatch when stride-aware iteration isn't worth the complexity, but use it judiciously to avoid unnecessary copies.

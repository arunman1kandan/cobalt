# 085: View Implementation Details

## 1. Motivation
Views are the key to efficient tensor manipulation. Instead of copying data when we transpose, slice, or reshape, we create a new "window" into the same memory. This note covers the **actual implementation** of views in Cobalt, including strided indexing, contiguity checks, and materialization.

## 2. Context / Precedence
- **NumPy**: `ndarray.view()` creates views with custom strides
- **PyTorch**: `tensor.view()` for contiguous, `tensor.as_strided()` for general views
- **TensorFlow**: Primarily uses copies; views less common
- **Julia**: `view()` and `@view` macro for zero-copy slicing

## 3. TensorView Structure

### Full Definition
```rust
use std::sync::Arc;
use crate::dtype::DType;
use crate::device::Device;

pub struct TensorView {
    /// Shared reference to underlying data
    data: Arc<Vec<u8>>,
    
    /// Shape of this view
    shape: Vec<usize>,
    
    /// Stride for each dimension (byte offset between elements)
    strides: Vec<usize>,
    
    /// Byte offset into data where this view starts
    offset: usize,
    
    /// Data type of elements
    dtype: DType,
    
    /// Device where data resides
    device: Device,
}
```

### Fields Explained

#### `data: Arc<Vec<u8>>`
- Shared reference to raw bytes
- Multiple views can point to same data
- See [065-shared-memory-arc.md](065-shared-memory-arc.md) for details

#### `shape: Vec<usize>`
- Logical dimensions of this view
- Example: `[3, 2]` for a 3×2 matrix
- Not necessarily the same as the original tensor's shape

#### `strides: Vec<usize>`
- **Byte strides** for each dimension
- `strides[i]` = byte offset to move one step in dimension `i`
- Row-major (C) layout: `strides[i] = element_size * ∏(shape[i+1..])`
- Example for `[3, 2]` with FP32 (4 bytes):
  - `strides[0] = 4 * 2 = 8` (moving one row skips 2 floats = 8 bytes)
  - `strides[1] = 4` (moving one column skips 1 float = 4 bytes)

#### `offset: usize`
- Starting byte position in `data`
- Allows slicing without changing the Arc pointer
- Example: `slice(2, 5)` sets `offset += 2 * stride[0]`

#### `dtype: DType`
- Element data type (FP32, INT32, etc.)
- Determines element size and interpretation

#### `device: Device`
- CPU vs GPU (future)
- Currently always CPU

## 4. Strided Indexing Formula

The **core equation** for accessing element at multi-dimensional index `[i₀, i₁, ..., i_{n-1}]`:

```
byte_offset = offset + Σ(i_k * strides[k]) for k=0 to n-1
```

### Example Calculation

Tensor: `[3, 2]` shape, FP32 dtype
```
Data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
       ↑0   ↑4   ↑8   ↑12  ↑16  ↑20  (byte offsets)

Shape:   [3, 2]
Strides: [8, 4]  (row-major)
Offset:  0
```

Access element `[1, 1]` (second row, second column):
```
byte_offset = 0 + 1*8 + 1*4 = 12
Value at offset 12: 4.0 ✓
```

Access element `[2, 0]` (third row, first column):
```
byte_offset = 0 + 2*8 + 0*4 = 16
Value at offset 16: 5.0 ✓
```

## 5. Computing Strides (Row-Major Layout)

### Algorithm
```rust
fn compute_strides(shape: &[usize], element_size: usize) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = element_size;
    
    // Build from innermost (rightmost) dimension outward
    for &dim in shape.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    
    // Reverse to match shape order
    strides.reverse();
    strides
}
```

### Example Walkthrough

Shape: `[2, 3, 4]`, Element size: `4` bytes (FP32)

**Step 1**: Start from innermost dimension (index 2)
- `dim = 4`, `stride = 4`
- `strides = [4]`, `stride = 4 * 4 = 16`

**Step 2**: Middle dimension (index 1)
- `dim = 3`, `stride = 16`
- `strides = [4, 16]`, `stride = 16 * 3 = 48`

**Step 3**: Outermost dimension (index 0)
- `dim = 2`, `stride = 48`
- `strides = [4, 16, 48]`

**Step 4**: Reverse
- `strides = [48, 16, 4]`

**Result**: `[48, 16, 4]`
- Moving 1 step in dimension 0 (outermost): skip 48 bytes (3×4 elements)
- Moving 1 step in dimension 1: skip 16 bytes (4 elements)
- Moving 1 step in dimension 2 (innermost): skip 4 bytes (1 element)

## 6. Core View Operations

### 6.1 Creating a View from Tensor

```rust
impl TensorView {
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, FrameworkError> {
        let element_size = tensor.dtype.size_in_bytes() as usize;
        let strides = Self::compute_strides(&tensor.shape, element_size);
        
        Ok(TensorView {
            data: Arc::clone(&tensor.data),  // Share memory
            shape: tensor.shape.clone(),
            strides,
            offset: 0,
            dtype: tensor.dtype,
            device: tensor.device,
        })
    }
}
```

**Cost**: O(n) where n = rank (for computing strides)
**Memory**: Just increments Arc refcount

### 6.2 Slicing (Single Dimension)

```rust
pub fn slice(&self, start: usize, end: usize) -> Result<Self, FrameworkError> {
    if start >= end || end > self.shape[0] {
        return Err(FrameworkError::IndexOutOfBounds {
            index: start,
            length: self.shape[0],
        });
    }
    
    let mut new_view = self.clone();
    new_view.offset += start * self.strides[0];  // Adjust starting point
    new_view.shape[0] = end - start;             // Shrink dimension
    Ok(new_view)
}
```

**Example**:
```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
// Rows: [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]

let v = t.view().unwrap();
let sliced = v.slice(1, 3).unwrap();  // Rows 1-2 (indices 1, 2)

// Before: offset=0, shape=[3,2]
// After:  offset=8, shape=[2,2]  (skipped first row = 8 bytes)
```

**Cost**: O(1) - just updates metadata

### 6.3 Slicing (Arbitrary Dimension)

```rust
pub fn slice_dim(&self, dim: usize, start: usize, end: usize) 
    -> Result<Self, FrameworkError> 
{
    if dim >= self.shape.len() {
        return Err(FrameworkError::InvalidDimension { 
            dim, 
            rank: self.shape.len() 
        });
    }
    
    if start >= end || end > self.shape[dim] {
        return Err(FrameworkError::IndexOutOfBounds {
            index: start,
            length: self.shape[dim],
        });
    }
    
    let mut new_view = self.clone();
    new_view.offset += start * self.strides[dim];
    new_view.shape[dim] = end - start;
    Ok(new_view)
}
```

### 6.4 Transpose (Swap Two Dimensions)

```rust
pub fn transpose(&self, dim1: usize, dim2: usize) 
    -> Result<Self, FrameworkError> 
{
    if dim1 >= self.shape.len() || dim2 >= self.shape.len() {
        return Err(FrameworkError::InvalidDimension { 
            dim: std::cmp::max(dim1, dim2), 
            rank: self.shape.len() 
        });
    }
    
    let mut new_view = self.clone();
    new_view.shape.swap(dim1, dim2);    // Swap dimensions
    new_view.strides.swap(dim1, dim2);  // Swap strides
    Ok(new_view)
}
```

**Example**:
```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
// [1.0, 2.0]
// [3.0, 4.0]

let v = t.transpose_view(0, 1).unwrap();
// Shape: [2, 2] → [2, 2] (still square)
// Strides: [8, 4] → [4, 8] (swapped)

// Now accessing [0, 1]:
// Before transpose: offset = 0 + 0*8 + 1*4 = 4  → 2.0
// After transpose:  offset = 0 + 0*4 + 1*8 = 8  → 3.0 ✓
```

**Key**: Transpose **doesn't move data**, just changes how we interpret it via strides.

### 6.5 Permute (Reorder All Dimensions)

```rust
pub fn permute(&self, axes: &[usize]) -> Result<Self, FrameworkError> {
    if axes.len() != self.shape.len() {
        return Err(FrameworkError::ShapeMismatch {
            expected: format!("{} dimensions", self.shape.len()),
            got: format!("{} axes", axes.len()),
        });
    }
    
    // Validate axes
    let mut seen = vec![false; axes.len()];
    for &ax in axes {
        if ax >= axes.len() {
            return Err(FrameworkError::InvalidDimension {
                dim: ax,
                rank: axes.len(),
            });
        }
        if seen[ax] {
            return Err(FrameworkError::DuplicateAxis { axis: ax });
        }
        seen[ax] = true;
    }
    
    let mut new_view = self.clone();
    let old_shape = new_view.shape.clone();
    let old_strides = new_view.strides.clone();
    
    for (i, &ax) in axes.iter().enumerate() {
        new_view.shape[i] = old_shape[ax];
        new_view.strides[i] = old_strides[ax];
    }
    
    Ok(new_view)
}
```

**Example**:
```rust
// Shape [2, 3, 4] with strides [48, 16, 4]
let v = tensor.permute_view(&[2, 0, 1]).unwrap();
// New shape: [4, 2, 3]
// New strides: [4, 48, 16]
```

### 6.6 Squeeze (Remove Size-1 Dimensions)

```rust
pub fn squeeze(&self) -> Result<Self, FrameworkError> {
    let new_shape: Vec<usize> = self.shape.iter()
        .enumerate()
        .filter(|(_, &s)| s != 1)
        .map(|(_, &s)| s)
        .collect();
    
    let new_strides: Vec<usize> = self.shape.iter()
        .enumerate()
        .filter(|(_, &s)| s != 1)
        .map(|(i, _)| self.strides[i])
        .collect();
    
    Ok(TensorView {
        data: Arc::clone(&self.data),
        shape: new_shape,
        strides: new_strides,
        offset: self.offset,
        dtype: self.dtype,
        device: self.device,
    })
}
```

**Example**:
```rust
// Shape [1, 3, 1, 2] → squeeze → [3, 2]
```

### 6.7 Unsqueeze (Add Size-1 Dimension)

```rust
pub fn unsqueeze(&self, dim: usize) -> Result<Self, FrameworkError> {
    if dim > self.shape.len() {
        return Err(FrameworkError::InvalidDimension {
            dim,
            rank: self.shape.len() + 1,
        });
    }
    
    let mut new_shape = self.shape.clone();
    let mut new_strides = self.strides.clone();
    
    new_shape.insert(dim, 1);
    new_strides.insert(dim, if dim < self.strides.len() { 
        self.strides[dim] 
    } else { 
        self.dtype.size_in_bytes() as usize
    });
    
    Ok(TensorView {
        data: Arc::clone(&self.data),
        shape: new_shape,
        strides: new_strides,
        offset: self.offset,
        dtype: self.dtype,
        device: self.device,
    })
}
```

**Example**:
```rust
// Shape [3, 2] → unsqueeze(0) → [1, 3, 2]
// Shape [3, 2] → unsqueeze(2) → [3, 2, 1]
```

## 7. Contiguity

### 7.1 What is Contiguity?

A tensor is **contiguous** if elements are laid out sequentially in memory following row-major order.

**Contiguous**:
```
Data: [1, 2, 3, 4, 5, 6]
Shape: [2, 3]
Strides: [12, 4] (FP32)
```

**Non-Contiguous** (after transpose):
```
Data: [1, 2, 3, 4, 5, 6]
Shape: [3, 2]
Strides: [4, 12]  ← Not row-major!
```

### 7.2 Checking Contiguity

```rust
pub fn is_contiguous(&self) -> bool {
    let element_size = self.dtype.size_in_bytes() as usize;
    let mut expected_stride = element_size;
    
    for (i, &dim) in self.shape.iter().enumerate().rev() {
        if dim == 0 {
            continue;
        }
        if self.strides[i] != expected_stride {
            return false;
        }
        expected_stride *= dim;
    }
    true
}
```

**Algorithm**:
1. Start from innermost dimension
2. Check if `stride[i] == expected_stride`
3. Update `expected_stride *= shape[i]`
4. If any mismatch → not contiguous

### 7.3 Why Contiguity Matters

**Operations that require contiguity**:
- `reshape()`: Can only reinterpret if memory is sequential
- SIMD optimizations: Vector loads need contiguous data
- GPU transfers: Efficient copy requires contiguous buffers

**Operations that work on non-contiguous**:
- Slicing: Just adjust offset/shape
- Elementwise ops: Can iterate with strides
- Transpose: Changes strides, doesn't care about contiguity

## 8. Reshape (Requires Contiguity)

```rust
pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, FrameworkError> {
    let new_numel: usize = new_shape.iter().product();
    if new_numel != self.numel() {
        return Err(FrameworkError::ShapeMismatch {
            expected: format!("{} elements", self.numel()),
            got: format!("{} elements", new_numel),
        });
    }

    if !self.is_contiguous() || self.offset != 0 {
        return Err(FrameworkError::UnsupportedOp("reshape on non-contiguous view"));
    }

    let element_size = self.dtype.size_in_bytes() as usize;
    let strides = Self::compute_strides(new_shape, element_size);

    Ok(TensorView {
        data: Arc::clone(&self.data),
        shape: new_shape.to_vec(),
        strides,
        offset: 0,
        dtype: self.dtype,
        device: self.device,
    })
}
```

**Key Restrictions**:
1. Same number of elements
2. Must be contiguous
3. Offset must be 0 (full view of original data)

## 9. Materialization (contiguous())

When a view is non-contiguous and we need contiguous layout (e.g., for reshape or efficient iteration), we **materialize** it by copying data into a new buffer.

```rust
pub fn contiguous(&self) -> Tensor {
    let element_size = self.dtype.size_in_bytes() as usize;
    let expected_strides = Self::compute_strides(&self.shape, element_size);
    let total_bytes = self.numel() * element_size;

    // Fast path: already contiguous
    if self.is_contiguous()
        && self.offset == 0
        && self.strides == expected_strides
        && total_bytes == self.data.len()
    {
        return Tensor {
            data: Arc::clone(&self.data),  // Just share
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        };
    }

    // Slow path: copy with stride-aware iteration
    let bytes = self.materialize_bytes();
    Tensor {
        data: Arc::new(bytes),
        shape: self.shape.clone(),
        dtype: self.dtype,
        device: self.device,
    }
}

fn materialize_bytes(&self) -> Vec<u8> {
    let element_size = self.dtype.size_in_bytes() as usize;
    let numel = self.numel();
    let mut out = vec![0u8; numel * element_size];

    for linear in 0..numel {
        // Convert linear index to multi-dimensional index
        let mut remaining = linear;
        let mut byte_offset = self.offset;

        for dim in (0..self.shape.len()).rev() {
            let size = self.shape[dim];
            let idx = remaining % size;
            remaining /= size;
            byte_offset += idx * self.strides[dim];
        }

        // Copy element bytes
        let src = &self.data[byte_offset..byte_offset + element_size];
        let dst_start = linear * element_size;
        out[dst_start..dst_start + element_size].copy_from_slice(src);
    }

    out
}
```

**Example**:
```rust
let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let transposed = t.transpose_view(0, 1).unwrap();
// Non-contiguous, strides=[4, 8]

let materialized = transposed.contiguous();
// Contiguous, data=[1.0, 3.0, 2.0, 4.0], strides=[8, 4]
```

## 10. Flatten

```rust
pub fn flatten(&self) -> Result<Self, FrameworkError> {
    self.reshape(&[self.numel()])
}
```

Simple wrapper around reshape. Requires contiguity.

## 11. Integration with Tensor

The `Tensor` struct exposes convenient view helpers:

```rust
impl Tensor {
    pub fn view(&self) -> Result<TensorView, FrameworkError> {
        TensorView::from_tensor(self)
    }

    pub fn slice(&self, start: usize, end: usize) -> Result<TensorView, FrameworkError> {
        self.view()?.slice(start, end)
    }

    pub fn transpose_view(&self, dim1: usize, dim2: usize) -> Result<TensorView, FrameworkError> {
        self.view()?.transpose(dim1, dim2)
    }

    pub fn reshape_view(&self, new_shape: &[usize]) -> Result<TensorView, FrameworkError> {
        self.view()?.reshape(new_shape)
    }

    pub fn flatten_view(&self) -> Result<TensorView, FrameworkError> {
        self.view()?.flatten()
    }
}
```

## 12. Performance Characteristics

| Operation | Time Complexity | Memory |
|-----------|-----------------|---------|
| `from_tensor()` | O(rank) | O(1) Arc clone |
| `slice()` | O(1) | O(rank) metadata clone |
| `transpose()` | O(1) | O(rank) metadata clone |
| `permute()` | O(rank) | O(rank) metadata clone |
| `squeeze()` | O(rank) | O(rank) metadata clone |
| `unsqueeze()` | O(1) | O(rank) metadata clone |
| `reshape()` | O(rank) | O(rank) metadata clone |
| `contiguous()` | O(numel) if copy needed | O(numel) if copy needed |
| `is_contiguous()` | O(rank) | O(1) |

**Key Insight**: All view operations except `contiguous()` are extremely fast because they only manipulate metadata.

## 13. Common Patterns

### Pattern 1: Slice + Transpose + Materialize
```rust
let t = Tensor::from_f32(data, vec![100, 100]);
let sliced = t.slice(10, 90)?;           // O(1)
let transposed = sliced.transpose(0, 1)?; // O(1)
let mat = transposed.contiguous();        // O(80*100) copy
```

### Pattern 2: Batch Processing with Views
```rust
let batch = Tensor::from_f32(data, vec![32, 3, 224, 224]);  // [B, C, H, W]

for i in 0..32 {
    let img_view = batch.slice(i, i+1)?;  // O(1), shape [1, 3, 224, 224]
    // Process img_view...
}
```

### Pattern 3: Channel-wise Operations
```rust
let img = Tensor::from_f32(data, vec![3, 224, 224]);  // [C, H, W]

let red_channel = img.slice_dim(0, 0, 1)?;    // [1, 224, 224]
let green_channel = img.slice_dim(0, 1, 2)?;  // [1, 224, 224]
let blue_channel = img.slice_dim(0, 2, 3)?;   // [1, 224, 224]
```

## 14. Debugging Views

Print view metadata:
```rust
impl std::fmt::Debug for TensorView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("dtype", &self.dtype)
            .field("contiguous", &self.is_contiguous())
            .field("numel", &self.numel())
            .finish()
    }
}
```

## 15. Error Handling

View operations can fail:
- **IndexOutOfBounds**: Slice exceeds dimension size
- **InvalidDimension**: Dimension index >= rank
- **DuplicateAxis**: Permute with repeated axes
- **ShapeMismatch**: Reshape with wrong element count
- **UnsupportedOp**: Reshape on non-contiguous view

All errors are descriptive and include context (indices, shapes, etc.).

## 16. Summary

TensorView implementation provides:

1. **Zero-copy operations**: Slice, transpose, permute via metadata manipulation
2. **Strided indexing**: Access elements via `offset + Σ(i_k * strides[k])`
3. **Contiguity tracking**: Detect if memory is row-major sequential
4. **Materialization**: Copy non-contiguous views when needed
5. **Type safety**: Rust's type system prevents aliasing bugs

**Core Philosophy**: Views are cheap (O(1) to O(rank)), copies are expensive (O(numel)). Defer materialization until absolutely necessary.

This enables efficient data manipulation in deep learning workloads without the memory overhead of naive copying.

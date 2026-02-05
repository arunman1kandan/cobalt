# Phase 1: Views & Slicing Implementation

## Overview
Phase 1 focuses on implementing **zero-copy tensor views**, slicing, and reshaping operations. This enables efficient data reinterpretation without memory copies, crucial for large tensors.

## Architecture

### TensorView Struct
```
TensorView {
    data: Arc<Vec<u8>>,    // Shared reference to underlying data
    shape: Vec<usize>,      // Current shape of this view
    strides: Vec<usize>,    // Byte strides per dimension
    offset: usize,          // Starting offset in data
    dtype: DType,           // Element data type
    device: Device,         // Device for storage
}
```

### Strided Indexing Formula
For element at index `[i₀, i₁, ..., i_n]`:
```
byte_offset = offset + Σ(i_k * strides[k])
```

This allows:
- **Slicing**: Change shape and offset, keep strides
- **Transposing**: Swap shape and stride dimensions
- **Reshaping**: Reinterpret data with new shape (if contiguous)
- **Broadcasting**: Expand shapes with zero stride

## Implementation Status

### Phase 1A: Core Views (COMPLETED ✓)
- [x] TensorView struct with Arc<Vec<u8>> for shared memory
- [x] slice() - Single dimension slicing
- [x] slice_dim() - Multi-dimension slicing
- [x] transpose() - Zero-copy dimension swap
- [x] permute() - Arbitrary axis reordering
- [x] squeeze() - Remove size-1 dimensions
- [x] unsqueeze() - Add size-1 dimensions
- [x] is_contiguous() - Check memory layout
- [x] Basic view tests

### Phase 1B: View Operations (COMPLETED ✓)
- [x] Reshape with contiguity checks
- [x] Flatten operations
- [x] Contiguous materialization (copy when needed)
- [x] Tensor view helper methods (slice/transpose/reshape)

### Phase 1C: Performance (OPTIONAL)
- [ ] Benchmark: view creation speed (O(1))
- [ ] Benchmark: slicing performance
- [ ] Benchmark: vs. copying data
- [ ] Memory profiling with Arc reference counting

### Phase 1D: Integration (COMPLETED ✓)
- [x] Tensor view helpers for easy creation
- [x] Contiguity checks and materialization path
- [x] Documentation and examples

## Key Concepts

### Zero-Copy Views
A view is metadata (shape, strides, offset) over shared storage:
```rust
let t = Tensor::from_f32([1,2,3,4,5,6], vec![3,2]);
let view = TensorView::from_tensor(&t)?;  // shares data
let sliced = view.slice(1, 3)?;            // offset changes only
let transposed = sliced.transpose(0,1)?;   // strides change only
```

### Memory Sharing with Arc
```rust
// Both views reference same data
let v1 = view.slice(0, 5)?;
let v2 = view.slice(2, 7)?;
// Arc::strong_count(&v1.data) == 2
```

### Contiguity Checking
A view is contiguous if:
```
stride[i] == stride[i+1] * shape[i+1]  for all i
```

Non-contiguous views must be materialized (copied) before certain operations.

## API Usage Examples

### Basic Slicing
```rust
let t = Tensor::from_f32(vec![1..12], vec![3, 4]);
let view = TensorView::from_tensor(&t)?;

// First row
let row0 = view.slice(0, 1)?;              // shape: [1, 4]

// Rows 1-2
let rows = view.slice(1, 3)?;              // shape: [2, 4]

// Specific dimension slicing
let col_range = view.slice_dim(1, 1, 3)?;  // shape: [3, 2]
```

### Transposition
```rust
let t = Tensor::from_f32(vec![1..7], vec![2, 3]);
let view = TensorView::from_tensor(&t)?;

// Swap axes
let transposed = view.transpose(0, 1)?;    // shape: [3, 2]

// Multiple permutations
let permuted = view.permute(&[1, 0])?;    // same as transpose(0,1)
```

### Shape Manipulation
```rust
let t = Tensor::from_f32(vec![1,2,3,4], vec![2, 2]);
let view = TensorView::from_tensor(&t)?;

// Remove size-1 dimensions
let unsqueezed = view.unsqueeze(0)?;      // shape: [1, 2, 2]
let squeezed = unsqueezed.squeeze()?;     // shape: [2, 2]

// Reshape (requires contiguity)
// let reshaped = view.reshape(&[4])?;    // shape: [4]
```

## Testing Strategy

### Unit Tests
- ✓ View creation from tensor
- ✓ Slicing correctness
- ✓ Transpose correctness
- ✓ Shape manipulation
- ✓ Stride computation

### Property-Based Tests
- [ ] All view operations preserve data integrity
- [ ] Shared memory verification
- [ ] Shape consistency checks

### Performance Tests
- [ ] View creation is O(1)
- [ ] Slicing is O(1)
- [ ] Memory overhead is minimal

## Future Enhancements

### Phase 1 Extensions
1. **Negative strides**: Support backward iteration (for reverse())
2. **Advanced indexing**: Fancy indexing with integer arrays
3. **Broadcasting views**: Dimensions with 0 stride
4. **Broadcasting with views**: All operations work on views automatically

### Phase 1 to Phase 2 Transition
Once views are stable:
- Start Phase 2: Broadcasting and advanced indexing
- Start Phase 3: Automatic differentiation with tape
- Keep Phase 0 & 1: Always available as foundation

## Building Phase 1

### Run tests
```bash
cargo test views -- --nocapture
```

### Check memory usage
```bash
cargo build --release
# Test with large tensors to verify Arc works correctly
```

### Benchmark views vs copies
```bash
cargo run --release
# Add benchmarks comparing:
# - View creation time
# - Slicing time
# - Memory overhead of Arc
```

## Notes for Implementation

### Current State
- TensorView struct designed and tested
- Strides computation correct
- Basic operations (slice, transpose, squeeze, unsqueeze) working
- Error handling in place

### Next Steps
1. Add reshape() with contiguity checking
2. Add flatten() convenience function
3. Add tests for complex view operations
4. Performance benchmarking
5. Documentation with examples

### Potential Issues & Solutions

**Issue**: Non-contiguous views in operations
**Solution**: Operations detect non-contiguous views and materialize if needed

**Issue**: Arc overhead for small tensors
**Solution**: Small tensors copied directly; views only beneficial for large tensors

**Issue**: Memory lifetime of underlying data
**Solution**: Arc guarantees data exists while views reference it

## Related Documentation
- See [notes/080-views-and-slicing.md](../notes/080-views-and-slicing.md) for detailed architecture
- See [PHASE0_COMPLETE.md](PHASE0_COMPLETE.md) for Phase 0 foundation
- See [INDEX.md](INDEX.md) for navigation

## Summary

Phase 1 implements the foundational zero-copy tensor view system, enabling efficient slicing, transposition, and reshaping without data copies. This is essential for:
1. **Memory efficiency**: Large tensors shared via Arc
2. **Computational efficiency**: O(1) view operations
3. **Flexibility**: Multiple interpretations of same data
4. **Foundation for broadcasting**: Views with zero strides for expanded dimensions

Current Progress: **100% complete** (core views, reshape/flatten, materialization, and docs)

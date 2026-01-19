# 080: Views and Slicing

## 1. Motivation
Deep Learning models handle massive datasets (GBs/TBs). Copying data is the cardinal sin of performance.
If we want to:
1.  Look at the left half of an image.
2.  Transpose a matrix.
3.  Flatten a volume.
We should be able to do this without moving a single byte of memory. **Views** allow us to create new Tensors that point to existing storage with different interpretation rules.

## 2. Context / Precedence
*   **C Pointers**: The original "View". `float*`.
*   **NumPy Slicing**: `arr[::2]` (Strided view).
*   **PyTorch**: `tensor.view()` vs `tensor.reshape()`. Distinct concepts.

## 3. Intuition
**The Magic Window**:
You have a giant poster (Storage). You have a small picture frame (Tensor Structure).
*   **Slicing**: You move the frame to a different spot.
*   **Transposing**: You rotate the frame 90 degrees (conceptually).
*   **Striding**: You look through a picket fence (skipping columns).

## 4. Formal Definition
A Tensor View is a tuple $(D, S, W, O)$:
*   $D$: Pointer to Data Storage.
*   $S$: Shape vector.
*   $W$: Stride vector.
*   $O$: Storage Offset (Start index).

Two tensors $A$ and $B$ are **Shared/Aliased** if $D_A == D_B$.

## 5. Mathematical Deep Dive
**The Strided Indexing Equation (General Form)**:
$$ \text{ptr}(i_0, \dots, i_n) = D + O + \sum_{k=0}^n i_k \cdot w_k $$

**Transposition Algebra**:
To transpose dimensions $k$ and $l$, we simply swap $s_k, s_l$ and $w_k, w_l$. No data moves.
Strides allow us to represent non-contiguous data layouts (like Column-Major) inside a Row-Major system.

## 6. Computation / Implementation Details
*   **Reference Counting**: The underlying `Storage` is typically an `Arc<Vec<u8>>` (Atomic Reference Counted).
    *   Tensor A created (Refcount 1).
    *   Tensor B = A.view() (Refcount 2).
    *   Tensor A dropped (Refcount 1). Storage stays alive until B is dropped.
*   **Negative Strides** (Python): `arr[::-1]` requires negative strides, which Rust's `Vec` (unsigned indexing) handles poorly. Usually emulated with an offset adjustment.

## 7. Minimal Code

### View Struct (Rust)
```rust
struct TensorView {
    storage: Arc<Vec<f32>>,
    shape: Vec<usize>,
    strides: Vec<usize>, 
    offset: usize,
}

impl TensorView {
    fn slice(&self, start: usize, end: usize) -> Self {
        let mut new_view = self.clone();
        new_view.offset += start * self.strides[0];
        new_view.shape[0] = end - start;
        new_view
    }
}
```

## 8. Practical Behavior
*   **Mutability Hazards**: If Tensor A and Tensor B share memory, writing to A changes B! This is dangerous but powerful.
*   **Performance**: Creating a view is $O(1)$ roughly 50ns.

## 9. Tuning / Debugging Tips
*   **Contiguity**: `A.transpose()` makes a tensor **non-contiguous**.
    *   Fast: `A.sum()` (jumpy memory access, but works).
    *   Impossible: `A.view([new_shape])` (cannot reshape non-contiguous memory).
    *   Fix: `A.contiguous()` (Allocates new memory, copies data, resets strides).

## 10. Historical Notes
The separation of **Storage** (Blob) from **Tensor** (Metadata) was a key architectural evolution in Torch7, allowing it to efficiently handle the shared memory needed for RNNs and LSTMs (Time-steps sharing weights).

## 11. Variants / Related Forms
*   **Cow (Copy On Write)**: Share memory until one writes, then clone. (Not typically used in DL due to unpredictability).
*   **Memory Mapping (mmap)**: Viewing a file on disk as a Tensor in RAM (used for datasets larger than RAM).

## 12. Examples / Exercises
**Exercise**: Transpose Logic.
Input: `Shape=[2, 3]`, `Strides=[3, 1]`.
Data: `[1, 2, 3, 4, 5, 6]` (Row Major).
Transpose:
New Shape: `[3, 2]`.
New Strides: `[1, 3]`.
Read (0, 1) in new tensor:
$0 \cdot 1 + 1 \cdot 3 = 3$. Value at index 3 is `4`. Correct.

## 13. Failure Cases / Limitations
*   **Vectorization**: SIMD requires contiguous loads. Iterating over a strided view requires "Gather" instructions or scalar loops, which are much slower.

## 14. Applications
*   **Multi-Head Attention**: Splitting `[Batch, Seq, 768]` into `[Batch, Seq, 12, 64]`.
*   **Sliding Windows**: Convolution patches are just views.

## 15. Connections to Other Concepts
*   **Computer Graphics**: Texture mapping and strided vertex buffers use identical logic.
*   **Database**: Columnar stores vs Row stores are just checking Stride=(1, N) vs Stride=(N, 1).

## 16. Frontier / Research Angle (Optional)
**Functorch**: Making `vmap` (Vectorizing Map) functional requires robust view handling to treat a batch of examples as a single tensor without logic changes.

## 17. Glossary of Terms
*   **Aliasing**: When two variables point to the same memory.
*   **Contiguous**: When `stride[i] == size[i+1] * stride[i+1]`.
*   **Storage**: The heap-allocated byte buffer.

## 18. References / Further Reading
*   [PyTorch Internals: Contiguity](https://pytorch.org/blog/pytorch-internals/)
*   [Strided layouts in Numpy](https://numpy.org/doc/stable/reference/arrays.ndarray.html)

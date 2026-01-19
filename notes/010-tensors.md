# 010: The Tensor

## 1. Motivation
In Deep Learning, we deal with massive amounts of data: images, audio waveforms, text embeddings, and batches of probabilities. We need a single, unified data structure that can:
1.  Store homogeneous numerical data efficiently (dense packing).
2.  Represent data of arbitrary dimensionality (scalars, vectors, matrices, n-D hypercubes).
3.  Support high-performance arithmetic operations on hardware accelerators (CPUs, GPUs).

The **Tensor** is this universal structure. It abstracts away the complexity of memory management, allowing researchers to focus on the shape and flow of data.

## 2. Context / Precedence
*   **Physics/Mathematics**: Originated in continuum mechanics (Stress Tensors) and Ricci Calculus. A tensor is technically an object invariant under coordinate transformations.
*   **Computer Science**: Re-appropriated to mean "N-dimensional array".
*   **Predecessors**:
    *   **Fortran Arrays**: The grandfather of numerical computing. Column-major by default.
    *   **NumPy (`ndarray`)**: The gold standard in Python. Introduced strides and broadcasting.
    *   **Torch/TensorFlow**: Added automatic differentiation and GPU support to the `ndarray` concept.

## 3. Intuition
Think of a Tensor as a **View** over a **Container**.
*   **The Container (Storage)**: A single, long, flat shelf of books. The books are locked in place.
*   **The View (Shape/Strides)**: A pair of "magic glasses" that makes you see those books arranged in a grid, a stack of grids, or a line.

You can change the glasses (Reshape) without moving the books.

## 4. Formal Definition
A Tensor $T$ of rank $R$ is defined by:
1.  **Data**: A sequence of scalars $D = \{d_0, d_1, \dots, d_{N-1}\}$ where $d_i \in \mathbb{S}$ (e.g., $\mathbb{R}$).
2.  **Shape**: A tuple $S = (s_0, s_1, \dots, s_{R-1})$ where $s_i \in \mathbb{Z}^+$ represents the size of dimension $i$.
3.  **Layout Mapping**: A function $L: \mathbb{Z}^R \to \mathbb{Z}$ that maps a multi-index $(i_0, \dots, i_{R-1})$ to a linear index in $D$.

## 5. Mathematical Deep Dive
The core mathematical concept governing modern tensors is the **Strided Layout Linear Combination**.

For a standard Row-Major (C-Style) layout, the linear index $k$ for a coordinate $(i_0, i_1, \dots, i_{R-1})$ is given by the dot product of the indices and the **Strides**.

$$ k = \sum_{j=0}^{R-1} i_j \cdot \text{stride}_j $$

Where the stride for dimension $j$, denoted $w_j$, is the product of the sizes of all subsequent dimensions:

$$ w_j = \prod_{k=j+1}^{R-1} s_k $$
(Base case: $w_{R-1} = 1$).

**Example:**
Shape $S = [2, 3]$.
Strides $W = [3, 1]$.
Index $(1, 2)$ -> $1 \cdot 3 + 2 \cdot 1 = 5$.

## 6. Computation / Implementation Details
*   **Memory Layout**: Cobalt uses **Row-Major** (C-contiguous) layout. This means the last dimension varies the fastest. This is cache-friendly for row-wise reduction ops.
*   **Data Types**: Cobalt uses a "Byte Bag" approach (`Vec<u8>`) in the storage layer to allow dynamic reinterpretation, though usually, we cast this to `&[f32]` for computation.
*   **Alignment**: To use SIMD (AVX2/AVX512), memory ideally should be aligned to 32 (AVX2) or 64 (AVX512) bytes. Rust's `Vec` aligns to the element size, so explicit alignment handling is often needed for high performance.

## 7. Minimal Code

### Rust Definition (Cobalt)
z```rust
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<u8>, // Type-erased storage
    pub dtype: DType,
    pub device: Device,
}
```

### Python/NumPy Equivalent
```python
import numpy as np
# Create a 2x3 tensor of float32s
t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(t.strides) # (12, 4) -> Steps in BYTES, not elements
```

## 8. Practical Behavior
*   **Creation**: Tensors are usually initialized with zeros, ones, or random noise.
*   **Immutability**: In functional frameworks (JAX), tensors are immutable. In Cobalt/PyTorch, they are mutable.
*   **Printing**: Rank > 2 tensors are printed as slices. The last dimension is printed horizontally.

## 9. Tuning / Debugging Tips
*   **Check Shapes First**: 99% of errors are shape mismatches. Always print `.shape` when a crash occurs.
*   **Contiguity Check**: If you perform a `transpose()`, the tensor becomes non-contiguous. Some operations (like `.view()`) require contiguity. In PyTorch, you call `.contiguous()`, which forces a memory copy.
*   **NaN Debugging**: If a tensor contains `NaN`, it will poison all future operations. Check for NaNs early.

## 10. Historical Notes
The term "Tensor" in ML was popularized by Google's **TensorFlow** (2015). Before that, "Multi-dimensional Arrays" or "Matrices" were more common terms in the Theano/Caffe era. The mathematical tensor (Ricci calculus) is quite different; ML tensors are strictly coordinate-dependent arrays unless explicitly designed otherwise (e.g., Geometric Deep Learning).

## 11. Variants / Related Forms
*   **Sparse Tensors**: Store only non-zero elements (indices + values). Essential for Graph Neural Networks.
*   **Ragged Tensors**: Dimensions can vary in length (e.g., batch of sentences of different lengths).
*   **Quantized Tensors**: Store integers (`int8`) but act like floats (via scale + zero_point) for inference efficiency.

## 12. Examples / Exercises
**Exercise:** Calculate the Flat Index.
*   Shape: `[2, 3, 4]`
*   Coordinate: `[1, 2, 1]`
*   Layout: Row-Major.

*Solution:*
Strides: `[12, 4, 1]`
Index: $1 \cdot 12 + 2 \cdot 4 + 1 \cdot 1 = 12 + 8 + 1 = 21$.

## 13. Failure Cases / Limitations
*   **OOM (Out of Memory)**: A `[10000, 10000]` float32 matrix is 400MB. A `[10000, 10000, 10000]` tensor is 4TB. It's easy to blow up RAM.
*   **Integer Overflow**: If a tensor is too large (>$2^{63}$ elements), simple indexing math can overflow `i64`.

## 14. Applications
*   **Images**: `[Batch, 3, Height, Width]` (NCHW) or `[Batch, Height, Width, 3]` (NHWC).
*   **Audio**: `[Batch, Channels, Time]`.
*   **NLP**: `[Batch, Sequence_Length, Embedding_Dim]`.

## 15. Connections to Other Concepts
*   **Linear Algebra**: Matrices (Rank 2) are the primary subsystem.
*   **Manifold Hypothesis**: High-dimensional tensors in DL are thought to lie on lower-dimensional manifolds.
*   **Memory Management**: Paging, Virtual Memory, and Cache Lines directly affect Tensor performance.

## 16. Frontier / Research Angle (Optional)
**Named Tensors**: Instead of `[32, 3, 224, 224]`, we define `[batch, channels, height, width]`. This prevents semantic errors (e.g., accidentally adding height to width). Proposed in PyTorch, but not yet standard.

## 17. Glossary of Terms
*   **Rank**: Number of dimensions.
*   **Shape**: Size of each dimension.
*   **Stride**: Steps to skip to get to the next element in a dimension.
*   **Contiguous**: Memory is arranged without gaps in the logical order.
*   **Scalar**: Rank 0 Tensor.

## 18. References / Further Reading
*   [NumPy Internals: Memory Interpretation](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)
*   [PyTorch Internals: Strides](https://pytorch.org/docs/stable/tensor_attributes.html)
*   [The Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/)

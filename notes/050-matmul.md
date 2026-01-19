# 050: Matrix Multiplication

## 1. Motivation
The Matrix Multiplication (GEMM - General Matrix Multiply) is the single most important algorithm in Deep Learning. It consumes >99% of FLOPs in Training LLMs.
Why? Because Neural Networks are defined as layers of linear transformations: $y = Wx + b$. This $Wx$ is a matrix multiply.

## 2. Context / Precedence
*   **BLAS Level 3**: The standard `dgemm` (Double Precision General Matrix Multiply) allows vendors (Intel MKL, NVIDIA cuBLAS) to optimize the underlying assembly without changing the mathematical definition.
*   **Einsum Notation**: `ik,kj->ij`.

## 3. Intuition
**The Shopping Receipt**:
You have a list of items to buy (Input Features) and their prices at different stores (Weights).
*   **Row 1 (User 1)**: `[2 Apples, 5 Bananas]`
*   **Col 1 (Store A)**: `[$1/Apple, $0.5/Banana]`
*   **Dot Product**: $(2 \times 1) + (5 \times 0.5) = 4.5$.
*   **Matrix Mul**: Doing this for All Users (Rows) and All Stores (Cols) simultaneously.

## 4. Formal Definition
Given Matrix $A \in \mathbb{R}^{M \times K}$ and Matrix $B \in \mathbb{R}^{K \times N}$, the product $C = AB$ where $C \in \mathbb{R}^{M \times N}$ is defined as:

$$ C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj} $$

Time Complexity: $O(M \cdot N \cdot K)$ (Cubic).
Space Complexity: $O(M \cdot N)$ (Quadratic).

## 5. Mathematical Deep Dive
**Arithmetic Intensity**:
The ratio of FLOPs to Bytes of Memory Accessed.
$$ I = \frac{2MNK}{MK + KN + MN} $$
For large square matrices ($N=M=K$), $I \approx \frac{2N^3}{3N^2} \approx \frac{2}{3} N$.
As $N$ grows, Intensity grows linearly. This means MatMul becomes **Compute Bound** (limited by CPU speed), unlike Elementwise ops which are **Memory Bound** (limited by RAM speed).

## 6. Computation / Implementation Details
*   **Naive ($O(N^3)$)**: Three nested loops. Horribly slow due to cache misses.
*   **Tiling / Blocking**: Divide the matrix into small $32 \times 32$ tiles that fit in L1 Cache. Compute the product of tiles.
*   **SIMD Packing**: Re-arrange data in memory to load 8 contiguous floats into AVX registers.
*   **Strassen Algorithm**: Recursively lowers complexity to $O(N^{2.8})$, but rarely used in DL due to numerical instability and overhead.

## 7. Minimal Code

### Naive Implementation (Rust)
```rust
pub fn matmul_naive(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2);
    
    let mut c_data = vec![0.0; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a.get(i, p) * b.get(p, j);
            }
            c_data[i * n + j] = sum;
        }
    }
    // ... return Tensor
}
```

## 8. Practical Behavior
*   **Cubic Scaling**: Doubling the matrix size ($1024 \to 2048$) makes it run $8\times$ slower.
*   **Warm-up**: First call might be slow if libraries (MKL/cuBLAS) perform lazy initialization or autotuning.

## 9. Tuning / Debugging Tips
*   **Transposition**: `A @ B.T` is often faster than `A @ B` because `B.T` allows accessing rows contiguously (if B is row-major), whereas accessing columns of `B` causes cache thrashing.
*   **Batching**: Always prefer `[Batch, M, K] @ [K, N]` over a loop of `[M, K] @ [K, N]`. It allows larger tiles.

## 10. Historical Notes
In 2012, AlexNet won ImageNet largely because they figured out how to run MatMul efficiently on two GTX 580 GPUs. The entire history of Deep Learning is the history of making MatMul faster.

## 11. Variants / Related Forms
*   **Vector-Matrix Multiply (GEMV)**: $y = Ax$. Memory bound.
*   **Batched MatMul (BMM)**: $[B, N, M] \times [B, M, K] = [B, N, K]$.
*   **Conv2d via Im2Col**: Convolution is implemented by turning the image patches into a Matrix ($Im2Col$) and doing a MatMul.

## 12. Examples / Exercises
**Exercise**: Calculate Output Shape.
$A: [32, 128]$ (Batch size 32, Hidden Dim 128)
$B: [128, 10]$ (Weights to 10 classes)
$C = A \times B$.
*Result*: $[32, 10]$.
Note: The inner dimension (128) disappears.

## 13. Failure Cases / Limitations
*   **Stack Overflow**: Not the error, but literal Recursion limits if using recursive tiling strategies.
*   **Precision Loss**: Adding millions of small numbers can result in floating point error accumulation. Kahan Summation is sometimes used.

## 14. Applications
*   **Fully Connected Layers (Linear)**: The standard building block.
*   **Attention ($Q \cdot K^T$)**: The core of Transformers is a MatMul between Queries and Keys.

## 15. Connections to Other Concepts
*   **Graph Theory**: Adjacency matrix multiplication counts paths between nodes.
*   **Physics**: Rotation matrices.

## 16. Frontier / Research Angle (Optional)
**1-bit LLMs (BitNet 1.58)**: Research into replacing $W \cdot x$ multiplication with just Addition (by restricting weights to $\{-1, 0, 1\}$). This eliminates the expensive Multiplier hardware requirements.

## 17. Glossary of Terms
*   **GEMM**: General Matrix Multiply.
*   **Dot Product**: Sum of products of elements.
*   **Transposition**: Flipping a matrix over its diagonal.

## 18. References / Further Reading
*   [GotoBLAS: Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_revision.pdf)
*   [Why GEMM is at the heart of Deep Learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

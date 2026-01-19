# Matrix Multiplication

## Purpose

Implement 2D matrix multiplication with shape signature:
[A, B] x [B, C] -> [A, C]

**Visual Representation:**
```
     [ B x C ]
      +-----+
      |     |
      |  B  |
[ A x B ] |     |
+-------+ +-----+
|   A   | |  C  |
+-------+ +-----+
    =
   [ A x C ]
```


This operation is foundational for modern deep learning and numerical computing. It underpins:
- Dense layers / Linear(x)
- Self-attention mechanisms
- Backpropagation for MLPs and Transformers
- Convolution lowering (im2col)
- Optimization kernels in training and inference

## Phase 0 Constraints

Initial implementation constraints:
- 2D only
- Naive algorithm: O(N^3)
- CPU execution
- Contiguous row-major storage (implicit)

This phase prioritizes correctness and API stability over performance.

## Future Roadmap

Phase 1:
- Batched matmul
- Broadcasting
- BLAS backend integration (OpenBLAS, MKL, Accelerate)
- Parallel CPU kernels

Phase 2:
- CUDA kernels
- Tensor Core awareness (FP16/BF16)
- Mixed-precision math
- Stream scheduling

Phase 3:
- JIT code generation
- Kernel fusion and tiling
- Runtime dispatch based on hardware characteristics

## Notes

Matrix multiplication is typically memory-bandwidth constrained for small matrices and compute-bound for large ones. High-performance implementations use:
- Tiling/Blocking for cache locality
- SIMD for inner loops
- Multi-core parallelism
- Specialized low-precision kernels for ML workloads

As the tensor library grows, matmul becomes the performance lighthouse for backend architecture.

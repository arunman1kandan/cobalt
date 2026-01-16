#  Shapes and Math

## Purpose
Define shape rules for tensor operations such as:
- addition
- multiplication
- matrix multiplication (2D)
- broadcasting (future phase)
- convolution (future phase)

## Basic Terminology
Shape is a list of dimensions, e.g.:
- `[N]` 1D vector
- `[N, M]` 2D matrix
- `[B, C, H, W]` 4D image batch

## Addition Rules (Phase 0)
Shapes must match:

```
[A, B] + [A, B] OK
[A, B] + [B, A] ERROR
```

Broadcasting will be added in future.

## Elementwise Multiplication Rules
Same shape rules as addition.

## Matrix Multiply (MatMul)
2D operands:

```
[A, B] x [B, C] -> [A, C]
```

Example:

```
3x4 x 4x5 -> 3x5
```

Requirements:
- 2D only (Phase 0)
- Dimension match on middle dims

## Why MatMul Matters
It is the core of:
- Linear layers
- Attention mechanism
- Weight updates
- Convolution lowering techniques
- Backpropagation math
- GPU kernels

In deep learning, Conv2d is usually compiled into MatMul indirectly for speed.

## Phase Breakdown
Phase 0:
- No broadcasting
- 2D-only MatMul

Phase 1:
- Broadcasting for add/mul

Phase 2:
- Batched matmul
- Image ops

Phase 3:
- Autograd compatibility

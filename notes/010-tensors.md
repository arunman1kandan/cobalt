# Tensor Design Notes

## Purpose
Define the core data structure that represents multi-dimensional numerical arrays used in all deep learning operations.

Every neural network operation (matmul, conv2d, add, relu, softmax, cross entropy, etc.) operates over Tensors.

Images, labels, weights, activations, gradients, all become Tensors.

## Requirements (Phase 0.1: CPU-only, no autodiff)
- Store contiguous data in 1D `Vec<f32>`
- Maintain shape as a `Vec<usize>`
- Allow construction from:
  - scalars
  - vectors
  - 2D arrays (for matmul)
  - later: images
- Basic operations:
  - add
  - mul
  - matmul (2D)
- Debug printing
- Shape checking

Future phases will add:
- Strides
- Broadcasting
- Autodiff metadata
- GPU device placement
- Memory pools and views

## Why Flattened Memory?
Using a flat `Vec<f32>` is standard because:
1. Less fragmented memory
2. Simpler GPU interoperability
3. Enables stride-based slicing

Example:
A 2D tensor with shape [3, 4] is stored as:

``` 
[ 12 contiguous values ]
```

## Shape Representation
`Vec<usize>` was chosen due to:
- dynamic flexibility
- easy indexing
- common in Rust and ML frameworks

Example and meaning:
- `[]` scalar
- `[N]` vector
- `[N,M]` matrix
- `[B,C,H,W]` image batch

## Validation Rules
Tensor creation must check:
- data.len() equals product(shape)

## Phase 0.2: Strides
Strides allow:
- slicing
- views
- transposes
- efficient data movement

Skipped initially for clarity.

## Phase 1.0: Autodiff Metadata
Tensors will later store:
- parents
- op type
- gradient storage

This produces dynamic computation graphs.

Example:

$$
y = xw + b
$$

produces graph nodes:
x  w  b  → mul → add → y

## Implementation Order
1. Tensor struct
2. Constructors
3. Debug display
4. Shape utilities
5. add
6. mul
7. matmul
8. tests
9. docs

This forms the foundation before autograd.

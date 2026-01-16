# Elementwise Operations

Elementwise operations apply a function independently to each corresponding element of one or more tensors. They form the basic vocabulary of numerical computing and deep learning.

## Phase 0 Scope

Initial operators:
- add(a, b)
- mul(a, b)

These define the baseline for more advanced kernels and composite operations.

## Planned Extensions

Future expansions may include:
- Broadcasting
- Fused kernels (e.g. a * b + c without intermediate allocation)
- In-place operations
- Parallel execution (multi-core)
- SIMD / vectorization
- GPU kernels

## Shape Constraints (Phase 0)

Current requirement:
shape(a) == shape(b)

Broadcasting will relax this constraint in later phases.

## Mathematical Definition

Given tensors a and b with identical shape and index set i:

add(a, b)_i = a_i + b_i
mul(a, b)_i = a_i * b_i

These serve as computational primitives for:
- Linear algebra kernels
- Activation functions
- Optimizer update rules
- Gradient computations
- Convolution internals

## Computational Notes

Elementwise arithmetic is:
- Embarrassingly parallel
- Branch-free (for basic arithmetic)
- Typically memory-bound, not compute-bound

This makes it a natural target for:
- SIMD
- Multi-core parallelism
- Kernel fusion
- GPU execution

High-performance tensor libraries optimize memory layout and fuse multiple elementwise ops to reduce allocation overhead and improve cache behavior.

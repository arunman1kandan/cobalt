# Cobalt
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-orange)

Cobalt is a tiny deep learning framework built from scratch in Rust for learning and exploration. The project focuses on understanding tensor operations, numerical computing, and the systems-level foundations of modern machine learning.

This is not intended as a production framework — it is a learning engine and experimental playground for low-level ML ideas.

## Goals (WIP)

- Understand how tensor libraries work internally
- Implement fundamental numerical primitives
- Document design and reasoning clearly
- Experiment with performance (SIMD, parallel, GPU)
- Build systems intuition beyond high-level APIs

## Current Features (Phase 0-1) ✅ COMPLETE

- **Tensor Core**: Multi-dimensional arrays with type-erased storage
- **Data Types**: 16 dtypes (FP32/64/16/BF16, INT/UINT 8/16/32/64, BOOL)
- **Elementwise Ops**: Add, Mul with full broadcasting
- **Matrix Operations**: MatMul (2D, naive O(n³))
- **Activations**: ReLU, Softmax (numerically stable)
- **Broadcasting**: NumPy-compatible with zero-copy stride manipulation
- **Views & Slicing**: Transpose, reshape, flatten, squeeze/unsqueeze, contiguous materialization
- **SIMD**: AVX2/AVX512 optimization for FP32 operations
- **Testing**: 32 comprehensive tests (100% passing)
- **Documentation**: Extensive notes under `/notes`

**Status**: Phase 0 and Phase 1 (views/slicing) complete with full test coverage.

## Roadmap

### Phase 0 (now) ✅ COMPLETE

- ✅ Elementwise ops (add, mul)
- ✅ ReLU and softmax activations
- ✅ Naive matmul
- ✅ Broadcasting support
- ✅ Multi-dtype support (FP32, FP64, INT32, INT64)
- ✅ Comprehensive test suite (32 tests)
- ✅ SIMD optimization (AVX2/AVX512)
- ✅ Documentation-first development

### Phase 1 (complete)

- ✅ Views and slicing (transpose, reshape, flatten, indexing helpers)

### Phase 1.5 (next)
- Reduction operations (sum, mean, max, min)
- Batched matmul
- Optimized tiled/blocked matmul
- Parallel CPU kernels (Rayon)
- More activations (GELU, Sigmoid, Tanh)
- Fused kernels (e.g., Add+ReLU)
- Initial benchmarks

### Phase 2

- Autograd system
- MLP layers and losses
- Mixed precision support

### Phase 3

- BLAS / GPU backends
- Kernel fusion and JIT concepts
- Tensor Core awareness

## Directory Structure
```
cobalt/
├── notes/
│   ├── intro.md
│   ├── 010-tensors.md
│   ├── 020-shapes-and-math.md
│   ├── 030-ops-architecture.md
│   ├── 040-elementwise.md
│   ├── 050-matmul.md
│   └── 060-activations-softmax.md
├── src/
│   ├── ops/
│   │   ├── activation.rs
│   │   ├── elementwise.rs
│   │   ├── matmul.rs
│   │   ├── mod.rs
│   │   └── softmax.rs
│   ├── main.rs
│   └── tensor.rs
├── target/
├── .gitignore
├── Cargo.lock
├── Cargo.toml
├── channellog.md
└── readme.md
```

## Quick Example

```rust
use cobalt::Tensor;

fn main() {
    let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let c = a.add(&b).unwrap();
    println!("Add: {:?}", c.as_f32_slice());

    let v = a.transpose_view(0, 1).unwrap();
    let v_mat = v.contiguous();
    println!("View transpose (materialized): {:?}", v_mat.as_f32_slice());
}
```

## Design Principles

- Bottom-up learning
- Documentation-first
- Correctness before speed
- Performance later (SIMD, threads, GPU)
- Small, focused components

## Why Rust?

Rust provides:

- Memory safety without GC
- Precise control over data layout
- Borrow rules useful for parallel kernels
- Zero-cost abstractions
- GPU/FFI potential
- Strong type system for tensor APIs

These align with ML systems where tensors = memory + structured compute.

## Inspirations

Conceptual influences include:

- PyTorch (ATen + autograd)
- NumPy
- JAX
- tinygrad
- cuBLAS / BLIS kernel styles
- Rust crates like `ndarray` and `tch-rs`

## Status

Experimental and evolving. Expect refactors and iteration.

## Author

**Arun Manikandan**

*Fueled by caffeine and late-night curiosity.*

*"Understanding comes from building, not just using."*
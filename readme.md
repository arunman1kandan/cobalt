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

## Current Features (Phase 0)

- Tensor type (CPU)
- Elementwise ops (add, mul, ReLU)
- Softmax (last dimension)
- Naive 2D matmul (O(n^3))
- Structured design notes under `/notes`

**Focus:** correctness and conceptual clarity, not performance. Although might evolve into focusing on performance.

## Roadmap

### Phase 0 (now)

- Elementwise ops
- ReLU and softmax
- Naive matmul
- Documentation-first development

### Phase 1

- Broadcasting
- Batched matmul
- Parallel CPU kernels
- Fused kernels
- Initial tests and benchmarks

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
// Note: This is the target API design - implementation in progress

fn main() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let c = a.add(&b);
    println!("{:?}", c);
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
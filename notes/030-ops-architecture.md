# Operations Architecture

## Overview
Operations (Ops) are the verbs of the framework. They take Tensors as input and produce Tensors as output.

## The Dispatch System
Cobalt uses **Dynamic Runtime Dispatch** to handle different datatypes (`FP32`, `INT32`, etc.).

1. **Frontend (User API):** The user calls `tensor.add(&other)`. The `Tensor` struct is untyped (holds raw bytes).
2. **Dispatcher (`ops` layer):** The function checks `self.dtype`.
    - If `FP32`, it routes to `backend::cpu::add::add::<f32>`.
    - If `INT32`, it routes to `backend::cpu::add::add::<i32>`.
3. **Backend (Kernel):** The specific implementation (e.g., AVX2 optimized) is executed using concrete Rust types.

This allows us to maintain a clean Python-like API while getting C-like performance where it counts.


## Purpose
Define how mathematical operations are structured within the framework,
separate from the Tensor data container.

This separation enables:
- backend dispatch (CPU, CUDA, Vulkan, etc)
- autograd and graph tracing
- device-specific kernel optimizations
- clean layering and testability

## Design Principle
Tensor should:
- store data
- store shape
- provide ergonomic wrappers
- remain backend-agnostic

Ops should:
- contain all numerical logic
- optionally dispatch to device kernels
- obey shape rules
- maintain mathematical correctness

## Why Not Put Ops Inside Tensor?
Mixing storage + compute leads to:
- tight coupling
- poor backend extensibility
- tangled abstractions

PyTorch & JAX both separate data vs ops.

## Dispatch Model (Phase 0)
Current dispatch path:

```
Tensor::add() -> ops::elementwise::add()
Tensor::matmul() -> ops::matmul::matmul()
Tensor::relu() -> ops::activation::relu()
```

## Future Dispatch (Phase 3)
```
ops::matmul -> backend::matmul -> cpu/cuda/etc
``` 

Backend pattern (future):

```rust
trait Backend {
fn matmul(&self, ...)
fn add(&self, ...)
}
```

## Summary
This phase sets up CPU ops in a form compatible with:
- autograd
- CUDA backend
- batched operations
- image layers

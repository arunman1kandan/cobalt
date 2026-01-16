# Ops Architecture

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

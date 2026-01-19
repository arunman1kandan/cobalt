/// Operations module.
///
/// This module exposes the high-level API for tensor operations.
/// It delegates the actual computation to the architecture-specific backend (e.g., `backend::cpu`).

pub mod elementwise;
pub mod matmul;
pub mod activation;
pub mod softmax;

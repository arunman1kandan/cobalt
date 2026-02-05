//! Cobalt: A tiny deep learning framework built from scratch in Rust.
//!
//! This library provides tensor operations, automatic differentiation,
//! and neural network building blocks for learning and experimentation.

pub mod dtype;
pub mod device;
pub mod tensor;
pub mod backend;
pub mod ops;
pub mod errors;
pub mod broadcast;

#[cfg(test)]
mod tests;

// Re-export common types
pub use tensor::Tensor;
pub use dtype::DType;
pub use device::Device;
pub use errors::FrameworkError;

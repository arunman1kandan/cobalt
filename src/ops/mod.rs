/// This file contains ergonomic wrappers around operations defined in the `ops` module.
/// These methods are implemented on the `Tensor` struct and forward calls to the corresponding

pub mod elementwise;
pub mod matmul;
pub mod activation;
pub mod softmax;
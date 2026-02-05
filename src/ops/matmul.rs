use crate::tensor::Tensor;
use crate::errors::FrameworkError;

/// Performs Matrix Multiplication.
///
/// Supports 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    crate::backend::cpu::matmul::matmul(a, b)
}

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::errors::FrameworkError;
use crate::backend::cpu::*;

/// Performs Matrix Multiplication.
///
/// Supports broadcasting for batched matmul (e.g., [B, M, K] @ [B, K, N] -> [B, M, N]).
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    unimplemented!("migrating to backend SIMD")

}

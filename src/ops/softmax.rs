use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::errors::FrameworkError;
use crate::backend::cpu::*;

/// Applies the Softmax function along the last dimension.
///
/// `Softmax(x_i) = exp(x_i) / sum(exp(x_j))`
pub fn softmax(a: &Tensor) -> Result<Tensor, FrameworkError> {
    unimplemented!("migrating to backend SIMD")
}

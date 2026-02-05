use crate::tensor::Tensor;
use crate::errors::FrameworkError;

/// Applies the Softmax function along the last dimension.
///
/// `Softmax(x_i) = exp(x_i) / sum(exp(x_j))`
pub fn softmax(a: &Tensor) -> Result<Tensor, FrameworkError> {
    crate::backend::cpu::softmax::softmax(a)
}

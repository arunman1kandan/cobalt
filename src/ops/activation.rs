use crate::tensor::Tensor;
use crate::errors::FrameworkError;

/// Applies the Rectified Linear Unit (ReLU) activation function.
///
/// `f(x) = max(0, x)`
pub fn relu(a: &Tensor) -> Result<Tensor, FrameworkError> {
    crate::backend::cpu::relu::relu(a)
}

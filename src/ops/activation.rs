use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::errors::FrameworkError;
use crate::backend::cpu::*;

/// Applies the Rectified Linear Unit (ReLU) activation function.
///
/// `f(x) = max(0, x)`
pub fn relu(a: &Tensor) -> Result<Tensor, FrameworkError> {
    unimplemented!("migrating to backend SIMD")
}

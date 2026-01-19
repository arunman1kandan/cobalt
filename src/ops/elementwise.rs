use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::device::Device;
use crate::errors::FrameworkError;
use crate::broadcast::broadcast_shapes;
use crate::backend::cpu::*;

// ========================================================================
// elementwise ADD with broadcasting + dtype/device/error handling
// ========================================================================

/// Performs elementwise addition with broadcasting.
///
/// # Arguments
/// * `a` - First input tensor.
/// * `b` - Second input tensor.
///
/// # Returns
/// A new Tensor containing the result of `a + b`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    crate::backend::cpu::add::add(a, b)
}


/// Performs elementwise multiplication with broadcasting.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    unimplemented!("migrating to backend SIMD")
}
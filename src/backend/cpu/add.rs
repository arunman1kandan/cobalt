use crate::backend::cpu::isa::*;
use crate::backend::cpu::{add_scalar, add_avx2, add_avx512};
use crate::tensor::Tensor;
use crate::errors::FrameworkError;

/// CPU backend entry point for addition.
///
/// This function coordinates the selection of the best kernel for the hardware.
/// 1. If operands are FP32 and AVX is available, it uses specialized kernels.
/// 2. Otherwise, it falls back to the generic scalar implementation.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    // Attempt vectorized kernels for FP32
    if a.dtype == crate::dtype::DType::FP32 && b.dtype == crate::dtype::DType::FP32 {
        unsafe {
            if has_avx512() {
                return add_avx512::add_avx512(a, b);
            }
            if has_avx2() {
                return add_avx2::add_avx2(a, b);
            }
        }
    }
    
    // Fallback or handle other types via scalar dispatch
    add_scalar::add_scalar_dispatch(a, b)
}

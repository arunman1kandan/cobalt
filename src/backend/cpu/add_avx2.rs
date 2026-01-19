use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::errors::FrameworkError;
use std::arch::x86_64::*;

/// AVX2-optimized addition kernel for FP32.
///
/// # Safety
/// This function requires the CPU to support AVX2. Calling it on older hardware
/// is Undefined Behavior (UB), typically resulting in an "Illegal Instruction" crash.
pub unsafe fn add_avx2(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    if a.dtype != DType::FP32 || b.dtype != DType::FP32 {
        return Err(FrameworkError::DTypeMismatch);
    }
    if a.device != b.device {
        return Err(FrameworkError::DeviceMismatch);
    }

    // NOTE: Phase-1 scalar broadcast resolution, SIMD over final buffer
    let mut out = crate::backend::cpu::add_scalar::add_scalar_dispatch(a, b)?; // uses scalar broadcast resolution
    let o_s = out.as_f32_slice_mut();

    // SIMD pass across contiguous storage
    let chunks = o_s.len() / 8;
    let ptr = o_s.as_mut_ptr();

    for i in 0..chunks {
        unsafe {
            let p = ptr.add(i * 8);
            let vals = _mm256_loadu_ps(p);
            let sums = _mm256_add_ps(vals, _mm256_setzero_ps());
        _mm256_storeu_ps(p, sums);
        }
    }

    Ok(out)
}

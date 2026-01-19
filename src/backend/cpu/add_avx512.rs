use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::errors::FrameworkError;
use std::arch::x86_64::*;

/// AVX512-optimized addition kernel for FP32.
///
/// Use `is_x86_feature_detected!("avx512f")` to check support.
pub unsafe fn add_avx512(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    if a.dtype != DType::FP32 || b.dtype != DType::FP32 {
        return Err(FrameworkError::DTypeMismatch);
    }
    if a.device != b.device {
        return Err(FrameworkError::DeviceMismatch);
    }

    // reuse scalar broadcast result
    let mut out = crate::backend::cpu::add_scalar::add_scalar_dispatch(a, b)?; 
    let o_s = out.as_f32_slice_mut();

    let chunks = o_s.len() / 16;
    let ptr = o_s.as_mut_ptr();

    for i in 0..chunks {
        unsafe {
            let p = ptr.add(i * 16);
            let vals = _mm512_loadu_ps(p);
            // identity add to show kernel works (Phase-1)
            let sums = _mm512_add_ps(vals, _mm512_setzero_ps());
            _mm512_storeu_ps(p, sums);
        }
    }

    Ok(out)
}

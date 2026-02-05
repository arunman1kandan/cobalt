use crate::tensor::{Tensor, Element};
use crate::errors::FrameworkError;
use crate::dtype::DType;
use crate::broadcast::broadcast_shapes;

/// CPU backend entry point for multiplication.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    // Device check
    if a.device != b.device {
        return Err(FrameworkError::DeviceMismatch);
    }

    // Dtype dispatch
    match (a.dtype, b.dtype) {
        (DType::FP32, DType::FP32) => mul_typed::<f32>(a, b),
        (DType::FP64, DType::FP64) => mul_typed::<f64>(a, b),
        (DType::FP16, DType::FP16) => mul_typed::<half::f16>(a, b),
        (DType::BF16, DType::BF16) => mul_typed::<half::bf16>(a, b),
        (DType::INT32, DType::INT32) => mul_typed::<i32>(a, b),
        (DType::INT64, DType::INT64) => mul_typed::<i64>(a, b),
        (DType::INT16, DType::INT16) => mul_typed::<i16>(a, b),
        (DType::INT8, DType::INT8) => mul_typed::<i8>(a, b),
        (DType::UINT8, DType::UINT8) => mul_typed::<u8>(a, b),
        (DType::UINT16, DType::UINT16) => mul_typed::<u16>(a, b),
        (DType::UINT32, DType::UINT32) => mul_typed::<u32>(a, b),
        (DType::UINT64, DType::UINT64) => mul_typed::<u64>(a, b),
        (DType::BOOL, DType::BOOL) => Err(FrameworkError::UnsupportedDType("BOOL doesn't support multiplication".to_string())),
        (DType::FP8E4M3, _) | (_, DType::FP8E4M3) |
        (DType::FP8E5M2, _) | (_, DType::FP8E5M2) |
        (DType::INT4, _) | (_, DType::INT4) => {
            Err(FrameworkError::UnsupportedDType(format!("dtype pair {:?}, {:?} not yet implemented", a.dtype, b.dtype)))
        }
        _ => Err(FrameworkError::DTypeMismatch),
    }
}

fn mul_typed<T: Element + std::ops::Mul<Output = T>>(
    a: &Tensor,
    b: &Tensor,
) -> Result<Tensor, FrameworkError> {
    let a_data = a.as_slice::<T>();
    let b_data = b.as_slice::<T>();

    // Handle broadcasting
    if a.shape == b.shape {
        // Same shape - simple elementwise
        let mut out = Tensor::new_raw(a.shape.clone(), T::DTYPE, a.device.clone());
        let out_data = out.as_slice_mut::<T>();
        
        for i in 0..a_data.len() {
            out_data[i] = a_data[i] * b_data[i];
        }
        
        Ok(out)
    } else {
        // Broadcasting required
        let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
        let mut out = Tensor::new_raw(out_shape.clone(), T::DTYPE, a.device.clone());
        let out_data = out.as_slice_mut::<T>();

        // Compute strides
        let a_strides = compute_strides(&a.shape);
        let b_strides = compute_strides(&b.shape);
        let out_strides = compute_strides(&out_shape);

        // Broadcast strides (set to 0 for size-1 dimensions)
        let a_broadcast_strides = broadcast_strides(&a.shape, &out_shape, &a_strides);
        let b_broadcast_strides = broadcast_strides(&b.shape, &out_shape, &b_strides);

        let numel = out_data.len();
        for i in 0..numel {
            let a_idx = compute_broadcast_index(i, &out_shape, &out_strides, &a_broadcast_strides);
            let b_idx = compute_broadcast_index(i, &out_shape, &out_strides, &b_broadcast_strides);
            out_data[i] = a_data[a_idx] * b_data[b_idx];
        }

        Ok(out)
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn broadcast_strides(src_shape: &[usize], target_shape: &[usize], src_strides: &[usize]) -> Vec<usize> {
    let mut result = vec![0; target_shape.len()];
    let offset = target_shape.len() - src_shape.len();
    
    for i in 0..src_shape.len() {
        if src_shape[i] == 1 {
            result[i + offset] = 0; // Broadcast dimension
        } else {
            result[i + offset] = src_strides[i];
        }
    }
    result
}

fn compute_broadcast_index(
    linear_idx: usize,
    out_shape: &[usize],
    out_strides: &[usize],
    broadcast_strides: &[usize],
) -> usize {
    let mut idx = 0;
    let mut remaining = linear_idx;
    
    for i in 0..out_shape.len() {
        let coord = remaining / out_strides[i];
        remaining %= out_strides[i];
        idx += coord * broadcast_strides[i];
    }
    
    idx
}

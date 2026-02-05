use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::errors::FrameworkError;
use crate::tensor::Element;

/// Dispatches binary addition to the appropriate type-specific implementation.
///
/// This function acts as a switch statement on the runtime `dtype`. It matches the
/// enum variant and instantiates the generic `add_scalar_impl<T>`.
///
/// # Arguments
/// * `a` - First input tensor
/// * `b` - Second input tensor (must have same dtype)
pub fn add_scalar_dispatch(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    if a.dtype != b.dtype {
        return Err(FrameworkError::DTypeMismatch);
    }
    
    match a.dtype {
        DType::FP32 => add_scalar_impl::<f32>(a, b),
        DType::FP64 => add_scalar_impl::<f64>(a, b),
        DType::FP16 => add_scalar_impl::<half::f16>(a, b),
        DType::BF16 => add_scalar_impl::<half::bf16>(a, b),
        DType::INT32 => add_scalar_impl::<i32>(a, b),
        DType::INT64 => add_scalar_impl::<i64>(a, b),
        DType::INT16 => add_scalar_impl::<i16>(a, b),
        DType::INT8 => add_scalar_impl::<i8>(a, b),
        DType::UINT8 => add_scalar_impl::<u8>(a, b),
        DType::UINT16 => add_scalar_impl::<u16>(a, b),
        DType::UINT32 => add_scalar_impl::<u32>(a, b),
        DType::UINT64 => add_scalar_impl::<u64>(a, b),
        DType::BOOL => Err(FrameworkError::UnsupportedDType("BOOL type doesn't support addition".to_string())),
        DType::FP8E4M3 | DType::FP8E5M2 | DType::INT4 => {
            Err(FrameworkError::UnsupportedDType(format!("{:?} not yet implemented", a.dtype)))
        }
    }
}

/// Generic implementation of scalar addition.
///
/// This implementation is instantiated for each supported type (f32, i32, etc.)
/// by the compiler. It handles:
/// 1. Broadcast check (Phase-1) to determine output shape.
/// 2. Iteration over the multi-dimensional shape to apply elementwise add.
fn add_scalar_impl<T: Element + std::ops::Add<Output = T>>(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    if a.device != b.device {
        return Err(FrameworkError::DeviceMismatch);
    }
    let shape = crate::broadcast::broadcast_shapes(&a.shape, &b.shape)?;
    let mut out = Tensor::new_raw(shape.clone(), T::DTYPE, a.device);

    let out_numel = out.numel();
    let o_s = out.as_slice_mut::<T>();
    let a_s = a.as_slice::<T>();
    let b_s = b.as_slice::<T>();

    // Compute strides for output shape
    let out_strides = compute_strides(&shape);
    
    // Compute broadcast-aware strides for inputs
    let a_strides = compute_broadcast_strides(&a.shape, &shape);
    let b_strides = compute_broadcast_strides(&b.shape, &shape);

    for i in 0..out_numel {
        let a_idx = compute_linear_index(i, &out_strides, &a_strides);
        let b_idx = compute_linear_index(i, &out_strides, &b_strides);
        o_s[i] = a_s[a_idx] + b_s[b_idx];
    }

    Ok(out)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn compute_broadcast_strides(src_shape: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; target_shape.len()];
    let offset = target_shape.len() - src_shape.len();
    
    // Compute normal strides for source shape
    let mut src_strides = vec![1; src_shape.len()];
    for i in (0..src_shape.len().saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }
    
    // Map to target shape with broadcasting (stride=0 for size-1 dims)
    for i in 0..src_shape.len() {
        if src_shape[i] == 1 {
            strides[i + offset] = 0; // Broadcast dimension
        } else {
            strides[i + offset] = src_strides[i];
        }
    }
    
    strides
}

fn compute_linear_index(
    out_idx: usize,
    out_strides: &[usize],
    broadcast_strides: &[usize],
) -> usize {
    let mut idx = 0;
    let mut remaining = out_idx;
    
    for i in 0..out_strides.len() {
        let coord = remaining / out_strides[i];
        remaining %= out_strides[i];
        idx += coord * broadcast_strides[i];
    }
    
    idx
}

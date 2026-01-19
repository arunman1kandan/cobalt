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
        DType::INT32 => add_scalar_impl::<i32>(a, b),
        DType::INT64 => add_scalar_impl::<i64>(a, b),
        DType::UINT8 => add_scalar_impl::<u8>(a, b),
         // For bool, we might not want to support add, or define it as OR/XOR? Or just standard add (1+1=1 or 2?)
         // Rust bool doesn't implement Add. We'll skip BOOL for now or cast?
         // Let's Skip BOOL for add for now.
        _ => Err(FrameworkError::DTypeMismatch), // Or unimplemented
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
    let a_shape = &a.shape;
    let b_shape = &b.shape;

    for i in 0..out_numel {
        let mut idx = vec![0; shape.len()];
        let mut tmp = i;
        for d in (0..shape.len()).rev() {
            idx[d] = tmp % shape[d];
            tmp /= shape[d];
        }

        let mut ai = 0;
        let mut stride = 1;
        for d in (0..shape.len()).rev() {
            let dim = if a_shape.len() > d {
                if a_shape[d] == 1 { 0 } else { idx[d] }
            } else { 0 };
            ai += dim * stride;
            stride *= if a_shape.len() > d { a_shape[d] } else { 1 };
        }

        let mut bi = 0;
        let mut stride2 = 1;
        for d in (0..shape.len()).rev() {
            let dim = if b_shape.len() > d {
                if b_shape[d] == 1 { 0 } else { idx[d] }
            } else { 0 };
            bi += dim * stride2;
            stride2 *= if b_shape.len() > d { b_shape[d] } else { 1 };
        }

        o_s[i] = a_s[ai] + b_s[bi];
    }

    Ok(out)
}

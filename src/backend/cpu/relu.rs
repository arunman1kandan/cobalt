use crate::tensor::{Tensor, Element};
use crate::errors::FrameworkError;
use crate::dtype::DType;

/// CPU backend entry point for ReLU activation.
pub fn relu(a: &Tensor) -> Result<Tensor, FrameworkError> {
    // Dtype dispatch
    match a.dtype {
        DType::FP32 => relu_typed::<f32>(a),
        DType::FP64 => relu_typed::<f64>(a),
        DType::FP16 => relu_typed::<half::f16>(a),
        DType::BF16 => relu_typed::<half::bf16>(a),
        DType::INT32 => relu_typed::<i32>(a),
        DType::INT64 => relu_typed::<i64>(a),
        DType::INT16 => relu_typed::<i16>(a),
        DType::INT8 => relu_typed::<i8>(a),
        DType::UINT8 => relu_typed::<u8>(a),
        DType::UINT16 => relu_typed::<u16>(a),
        DType::UINT32 => relu_typed::<u32>(a),
        DType::UINT64 => relu_typed::<u64>(a),
        _ => Err(FrameworkError::UnsupportedDType(format!(
            "relu not supported for {:?}",
            a.dtype
        ))),
    }
}

fn relu_typed<T: Element + PartialOrd + Default>(a: &Tensor) -> Result<Tensor, FrameworkError> {
    let a_data = a.as_slice::<T>();
    let mut out = Tensor::new_raw(a.shape.clone(), T::DTYPE, a.device.clone());
    let out_data = out.as_slice_mut::<T>();

    let zero = T::default();
    
    for i in 0..a_data.len() {
        out_data[i] = if a_data[i] > zero {
            a_data[i]
        } else {
            zero
        };
    }

    Ok(out)
}

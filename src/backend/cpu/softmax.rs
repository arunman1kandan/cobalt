use crate::tensor::{Tensor, Element};
use crate::errors::FrameworkError;
use crate::dtype::DType;

/// CPU backend entry point for Softmax activation.
/// Applies softmax along the last dimension.
pub fn softmax(a: &Tensor) -> Result<Tensor, FrameworkError> {
    // Dtype dispatch - only floating point types make sense for softmax
    match a.dtype {
        DType::FP32 => softmax_typed::<f32>(a),
        DType::FP64 => softmax_typed::<f64>(a),
        DType::FP16 => softmax_f16(a),
        DType::BF16 => softmax_bf16(a),
        _ => Err(FrameworkError::UnsupportedDType(format!(
            "softmax only supported for floating point types, got {:?}",
            a.dtype
        ))),
    }
}

fn softmax_f16(a: &Tensor) -> Result<Tensor, FrameworkError> {
    use half::f16;
    
    if a.shape.is_empty() {
        return Err(FrameworkError::ShapeMismatch {
            expected: "non-empty tensor".to_string(),
            got: "scalar".to_string(),
        });
    }
    
    let a_data = a.as_slice::<f16>();
    let last_dim = a.shape[a.shape.len() - 1];
    let num_batches = a.numel() / last_dim;
    
    let mut result = Vec::with_capacity(a.numel());
    
    for batch in 0..num_batches {
        let start_idx = batch * last_dim;
        let batch_data = &a_data[start_idx..start_idx + last_dim];
        
        // Convert to f32 for computation
        let f32_data: Vec<f32> = batch_data.iter().map(|x| x.to_f32()).collect();
        let max_val = f32_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = f32_data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        
        for &exp_val in &exp_vals {
            result.push(f16::from_f32(exp_val / sum_exp));
        }
    }
    
    Ok(Tensor::from_slice(&result, a.shape.clone()))
}

fn softmax_bf16(a: &Tensor) -> Result<Tensor, FrameworkError> {
    use half::bf16;
    
    if a.shape.is_empty() {
        return Err(FrameworkError::ShapeMismatch {
            expected: "non-empty tensor".to_string(),
            got: "scalar".to_string(),
        });
    }
    
    let a_data = a.as_slice::<bf16>();
    let last_dim = a.shape[a.shape.len() - 1];
    let num_batches = a.numel() / last_dim;
    
    let mut result = Vec::with_capacity(a.numel());
    
    for batch in 0..num_batches {
        let start_idx = batch * last_dim;
        let batch_data = &a_data[start_idx..start_idx + last_dim];
        
        // Convert to f32 for computation
        let f32_data: Vec<f32> = batch_data.iter().map(|x| x.to_f32()).collect();
        let max_val = f32_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = f32_data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        
        for &exp_val in &exp_vals {
            result.push(bf16::from_f32(exp_val / sum_exp));
        }
    }
    
    Ok(Tensor::from_slice(&result, a.shape.clone()))
}

fn softmax_typed<T>(a: &Tensor) -> Result<Tensor, FrameworkError>
where
    T: Element + num_traits::Float,
{
    if a.shape.is_empty() {
        return Err(FrameworkError::ShapeMismatch {
            expected: "non-empty tensor".to_string(),
            got: "scalar".to_string(),
        });
    }

    let a_data = a.as_slice::<T>();
    let mut out = Tensor::new_raw(a.shape.clone(), T::DTYPE, a.device.clone());
    let out_data = out.as_slice_mut::<T>();

    let last_dim = a.shape[a.shape.len() - 1];
    let num_batches = a.numel() / last_dim;

    // Process each batch (row for 2D, or sequence for higher dims)
    for batch in 0..num_batches {
        let offset = batch * last_dim;
        let slice = &a_data[offset..offset + last_dim];

        // Step 1: Find max for numerical stability
        let mut max_val = slice[0];
        for &val in slice.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }

        // Step 2: Compute exp(x - max) and sum
        let mut sum = T::zero();
        for i in 0..last_dim {
            let exp_val = (slice[i] - max_val).exp();
            out_data[offset + i] = exp_val;
            sum = sum + exp_val;
        }

        // Step 3: Normalize
        for i in 0..last_dim {
            out_data[offset + i] = out_data[offset + i] / sum;
        }
    }

    Ok(out)
}

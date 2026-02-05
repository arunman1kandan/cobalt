use crate::tensor::{Tensor, Element};
use crate::errors::FrameworkError;
use crate::dtype::DType;

/// CPU backend entry point for matrix multiplication.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, FrameworkError> {
    // Device check
    if a.device != b.device {
        return Err(FrameworkError::DeviceMismatch);
    }

    // Dtype dispatch
    match (a.dtype, b.dtype) {
        (DType::FP32, DType::FP32) => matmul_typed::<f32>(a, b),
        (DType::FP64, DType::FP64) => matmul_typed::<f64>(a, b),
        (DType::FP16, DType::FP16) => matmul_typed::<half::f16>(a, b),
        (DType::BF16, DType::BF16) => matmul_typed::<half::bf16>(a, b),
        _ => Err(FrameworkError::UnsupportedDType(format!(
            "matmul only supported for floating-point types, got {:?} and {:?}",
            a.dtype, b.dtype
        ))),
    }
}

fn matmul_typed<T: Element + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>(
    a: &Tensor,
    b: &Tensor,
) -> Result<Tensor, FrameworkError>
where
    T: Default,
{
    // Support 2D matmul for now
    if a.rank() != 2 || b.rank() != 2 {
        return Err(FrameworkError::ShapeMismatch {
            expected: "2D tensors".to_string(),
            got: format!("{}D and {}D", a.rank(), b.rank()),
        });
    }

    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];

    if k != k2 {
        return Err(FrameworkError::ShapeMismatch {
            expected: format!("Inner dimensions to match, got {} and {}", k, k2),
            got: format!("[{}, {}] @ [{}, {}]", m, k, k2, n),
        });
    }

    let a_data = a.as_slice::<T>();
    let b_data = b.as_slice::<T>();

    let mut out = Tensor::new_raw(vec![m, n], T::DTYPE, a.device.clone());
    let out_data = out.as_slice_mut::<T>();

    // Naive O(n^3) implementation
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for p in 0..k {
                let a_val = a_data[i * k + p];
                let b_val = b_data[p * n + j];
                sum = sum + (a_val * b_val);
            }
            out_data[i * n + j] = sum;
        }
    }

    Ok(out)
}

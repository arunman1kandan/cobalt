use crate::errors::FrameworkError;

/// Computes the resulting shape after broadcasting two shapes according to NumPy rules.
///
/// Use alignment from the right. A dimension is compatible if:
/// 1. They are equal.
/// 2. One of them is 1.
///
/// # Arguments
/// * `a` - Shape of the first tensor
/// * `b` - Shape of the second tensor
///
/// # Returns
/// * `Ok(Vec<usize>)` - The broadcasted output shape.
/// * `Err(FrameworkError::BroadcastMismatch)` - If shapes are incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, FrameworkError> {
    // align right
    let mut out = Vec::new();
    let mut i = a.len() as isize - 1;
    let mut j = b.len() as isize - 1;

    while i >= 0 || j >= 0 {
        let dim_a = if i >= 0 { a[i as usize] } else { 1 };
        let dim_b = if j >= 0 { b[j as usize] } else { 1 };

        if dim_a == dim_b {
            out.push(dim_a);
        } else if dim_a == 1 {
            out.push(dim_b);
        } else if dim_b == 1 {
            out.push(dim_a);
        } else {
            return Err(FrameworkError::BroadcastMismatch {
                a: a.to_vec(),
                b: b.to_vec(),
            });
        }

        i -= 1;
        j -= 1;
    }

    out.reverse();
    Ok(out)
}

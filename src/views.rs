/// View-based tensor operations (Phase 1)
/// 
/// This module implements zero-copy tensor views, slicing, and reshaping.
/// Views allow efficient data reinterpretation without memory copies.

use crate::tensor::Tensor;
use crate::errors::FrameworkError;
use crate::dtype::DType;
use crate::device::Device;
use std::sync::Arc;

/// TensorView: A read-only window into tensor data with custom strides
/// Allows zero-copy slicing, transposition, and reshaping
pub struct TensorView {
    /// Shared reference to underlying data
    data: Arc<Vec<u8>>,
    /// Shape of this view
    shape: Vec<usize>,
    /// Stride for each dimension (byte offset between elements)
    strides: Vec<usize>,
    /// Byte offset into data where this view starts
    offset: usize,
    /// Data type of elements
    dtype: DType,
    /// Device where data resides
    device: Device,
}

impl TensorView {
    /// Create a view from a tensor (shares underlying storage)
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, FrameworkError> {
        // For now, tensors are always contiguous
        // In future, we'll track strides in Tensor itself
        let element_size = tensor.dtype.size_in_bytes() as usize;
        let strides = Self::compute_strides(&tensor.shape, element_size);

        Ok(TensorView {
            data: Arc::clone(&tensor.data),
            shape: tensor.shape.clone(),
            strides,
            offset: 0,
            dtype: tensor.dtype,
            device: tensor.device,
        })
    }
    
    /// Compute strides for a given shape (assumes row-major C layout)
    fn compute_strides(shape: &[usize], element_size: usize) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = element_size;
        
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        
        strides.reverse();
        strides
    }
    
    /// Slice along first dimension
    /// Example: tensor[start..end] keeps all other dimensions
    pub fn slice(&self, start: usize, end: usize) -> Result<Self, FrameworkError> {
        if start >= end || end > self.shape[0] {
            return Err(FrameworkError::IndexOutOfBounds {
                index: start,
                length: self.shape[0],
            });
        }
        
        let mut new_view = self.clone();
        new_view.offset += start * self.strides[0];
        new_view.shape[0] = end - start;
        Ok(new_view)
    }
    
    /// Slice a specific dimension
    pub fn slice_dim(&self, dim: usize, start: usize, end: usize) 
        -> Result<Self, FrameworkError> 
    {
        if dim >= self.shape.len() {
            return Err(FrameworkError::InvalidDimension { dim, rank: self.shape.len() });
        }
        
        if start >= end || end > self.shape[dim] {
            return Err(FrameworkError::IndexOutOfBounds {
                index: start,
                length: self.shape[dim],
            });
        }
        
        let mut new_view = self.clone();
        new_view.offset += start * self.strides[dim];
        new_view.shape[dim] = end - start;
        Ok(new_view)
    }
    
    /// Transpose two dimensions (zero-copy)
    pub fn transpose(&self, dim1: usize, dim2: usize) 
        -> Result<Self, FrameworkError> 
    {
        if dim1 >= self.shape.len() || dim2 >= self.shape.len() {
            return Err(FrameworkError::InvalidDimension { 
                dim: std::cmp::max(dim1, dim2), 
                rank: self.shape.len() 
            });
        }
        
        let mut new_view = self.clone();
        new_view.shape.swap(dim1, dim2);
        new_view.strides.swap(dim1, dim2);
        Ok(new_view)
    }
    
    /// Permute dimensions (reorder axes)
    pub fn permute(&self, axes: &[usize]) -> Result<Self, FrameworkError> {
        if axes.len() != self.shape.len() {
            return Err(FrameworkError::ShapeMismatch {
                expected: format!("{} dimensions", self.shape.len()),
                got: format!("{} axes", axes.len()),
            });
        }
        
        // Check all axes are valid and unique
        let mut seen = vec![false; axes.len()];
        for &ax in axes {
            if ax >= axes.len() {
                return Err(FrameworkError::InvalidDimension {
                    dim: ax,
                    rank: axes.len(),
                });
            }
            if seen[ax] {
                return Err(FrameworkError::DuplicateAxis { axis: ax });
            }
            seen[ax] = true;
        }
        
        let mut new_view = self.clone();
        let old_shape = new_view.shape.clone();
        let old_strides = new_view.strides.clone();
        
        for (i, &ax) in axes.iter().enumerate() {
            new_view.shape[i] = old_shape[ax];
            new_view.strides[i] = old_strides[ax];
        }
        
        Ok(new_view)
    }
    
    /// Squeeze: remove dimensions of size 1
    pub fn squeeze(&self) -> Result<Self, FrameworkError> {
        let new_shape: Vec<usize> = self.shape.iter()
            .enumerate()
            .filter(|(_, &s)| s != 1)
            .map(|(_, &s)| s)
            .collect();
        
        let new_strides: Vec<usize> = self.shape.iter()
            .enumerate()
            .filter(|(_, &s)| s != 1)
            .map(|(i, _)| self.strides[i])
            .collect();
        
        Ok(TensorView {
            data: Arc::clone(&self.data),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
        })
    }
    
    /// Unsqueeze: add a dimension of size 1 at position
    pub fn unsqueeze(&self, dim: usize) -> Result<Self, FrameworkError> {
        if dim > self.shape.len() {
            return Err(FrameworkError::InvalidDimension {
                dim,
                rank: self.shape.len() + 1,
            });
        }
        
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        
        new_shape.insert(dim, 1);
        new_strides.insert(dim, if dim < self.strides.len() { 
            self.strides[dim] 
        } else { 
            self.dtype.size_in_bytes() as usize
        });
        
        Ok(TensorView {
            data: Arc::clone(&self.data),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
        })
    }

    /// Reshape this view (requires contiguity)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, FrameworkError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(FrameworkError::ShapeMismatch {
                expected: format!("{} elements", self.numel()),
                got: format!("{} elements", new_numel),
            });
        }

        if !self.is_contiguous() || self.offset != 0 {
            return Err(FrameworkError::UnsupportedOp("reshape on non-contiguous view"));
        }

        let element_size = self.dtype.size_in_bytes() as usize;
        let strides = Self::compute_strides(new_shape, element_size);

        Ok(TensorView {
            data: Arc::clone(&self.data),
            shape: new_shape.to_vec(),
            strides,
            offset: 0,
            dtype: self.dtype,
            device: self.device,
        })
    }

    /// Flatten this view into 1D (requires contiguity)
    pub fn flatten(&self) -> Result<Self, FrameworkError> {
        self.reshape(&[self.numel()])
    }

    /// Materialize this view into a contiguous Tensor (copies if needed)
    pub fn contiguous(&self) -> Tensor {
        let element_size = self.dtype.size_in_bytes() as usize;
        let expected_strides = Self::compute_strides(&self.shape, element_size);
        let total_bytes = self.numel() * element_size;

        if self.is_contiguous()
            && self.offset == 0
            && self.strides == expected_strides
            && total_bytes == self.data.len()
        {
            return Tensor {
                data: Arc::clone(&self.data),
                shape: self.shape.clone(),
                dtype: self.dtype,
                device: self.device,
            };
        }

        let bytes = self.materialize_bytes();
        Tensor {
            data: Arc::new(bytes),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
        }
    }

    fn materialize_bytes(&self) -> Vec<u8> {
        let element_size = self.dtype.size_in_bytes() as usize;
        let numel = self.numel();
        let mut out = vec![0u8; numel * element_size];

        for linear in 0..numel {
            let mut remaining = linear;
            let mut byte_offset = self.offset;

            for dim in (0..self.shape.len()).rev() {
                let size = self.shape[dim];
                let idx = remaining % size;
                remaining /= size;
                byte_offset += idx * self.strides[dim];
            }

            let src = &self.data[byte_offset..byte_offset + element_size];
            let dst_start = linear * element_size;
            out[dst_start..dst_start + element_size].copy_from_slice(src);
        }

        out
    }
    
    /// Check if this view is contiguous (standard row-major order)
    pub fn is_contiguous(&self) -> bool {
        let element_size = self.dtype.size_in_bytes() as usize;
        let mut expected_stride = element_size;
        for (i, &dim) in self.shape.iter().enumerate().rev() {
            if dim == 0 {
                continue;
            }
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }
    
    /// Get the shape of this view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the strides of this view
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Get the offset into the underlying storage
    pub fn offset(&self) -> usize {
        self.offset
    }
    
    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }
}

impl Clone for TensorView {
    fn clone(&self) -> Self {
        TensorView {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
        }
    }
}

// Potential future additions would go here as we expand Phase 1

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    
    #[test]
    fn test_view_from_tensor() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = TensorView::from_tensor(&t).unwrap();
        
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.numel(), 4);
        assert!(view.is_contiguous());
    }
    
    #[test]
    fn test_view_slice() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let view = TensorView::from_tensor(&t).unwrap();
        let sliced = view.slice(1, 3).unwrap();
        
        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.numel(), 4);
    }
    
    #[test]
    fn test_view_transpose() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = TensorView::from_tensor(&t).unwrap();
        let transposed = view.transpose(0, 1).unwrap();
        
        assert_eq!(transposed.shape(), &[2, 2]);
        // After transpose, strides should be swapped
        assert_ne!(view.strides(), transposed.strides());
    }
    
    #[test]
    fn test_view_squeeze() {
        let t = Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]);
        let view = TensorView::from_tensor(&t).unwrap();
        let squeezed = view.squeeze().unwrap();
        
        assert_eq!(squeezed.shape(), &[2]);
        assert_eq!(squeezed.numel(), 2);
    }
    
    #[test]
    fn test_view_unsqueeze() {
        let t = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
        let view = TensorView::from_tensor(&t).unwrap();
        let unsqueezed = view.unsqueeze(1).unwrap();
        
        assert_eq!(unsqueezed.shape(), &[2, 1]);
        assert_eq!(unsqueezed.numel(), 2);
    }

    #[test]
    fn test_view_reshape_flatten() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let view = TensorView::from_tensor(&t).unwrap();

        let reshaped = view.reshape(&[4]).unwrap();
        assert_eq!(reshaped.shape(), &[4]);

        let flattened = view.flatten().unwrap();
        assert_eq!(flattened.shape(), &[4]);
    }

    #[test]
    fn test_view_contiguous_materialize() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let view = TensorView::from_tensor(&t).unwrap();
        let transposed = view.transpose(0, 1).unwrap();

        let materialized = transposed.contiguous();
        let data = materialized.as_slice::<f32>();
        assert_eq!(materialized.shape, vec![3, 2]);
        assert_eq!(data, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}

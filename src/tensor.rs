use std::fmt;
use crate::dtype::DType;
use crate::device::Device;
use std::mem;

/// Private trait for types that can be elements of a Tensor.
/// This connects compile-time Rust types (f32, i32) to runtime DTypes.
pub trait Element: Copy + Clone + 'static + std::fmt::Debug + PartialEq {
    const DTYPE: DType;
}

impl Element for f32 { const DTYPE: DType = DType::FP32; }
impl Element for f64 { const DTYPE: DType = DType::FP64; }
impl Element for i32 { const DTYPE: DType = DType::INT32; }
impl Element for i64 { const DTYPE: DType = DType::INT64; }
impl Element for u8 { const DTYPE: DType = DType::UINT8; }
impl Element for i8 { const DTYPE: DType = DType::INT8; }
impl Element for bool { const DTYPE: DType = DType::BOOL; }

/// A multi-dimensional array containing elements of a single data type.
///
/// Cobalt Tensors are "untyped" (non-generic) structs wrapping a raw byte buffer.
/// This allows a single `Tensor` type to hold any data type at runtime, similar to PyTorch.
///
/// # Internals
/// - `data`: Raw `Vec<u8>` storage.
/// - `dtype`: Runtime tag indicating how to interpret the bytes.
/// - `shape`: Dimensions of the tensor.
#[derive(Clone)]
pub struct Tensor {
    /// Raw memory buffer. Layout is typically row-major (C-contiguous).
    pub data: Vec<u8>,     // raw byte buffer
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
}

impl Tensor {
    pub fn new_raw(shape: Vec<usize>, dtype: DType, device: Device) -> Self {
        let numel: usize = shape.iter().product();
        let bytes = numel * dtype.size_in_bytes();
        Self {
            data: vec![0u8; bytes],
            shape,
            dtype,
            device,
        }
    }

    /// Create a new Tensor from a slice of typed data.
    ///
    /// This copies the data into the internal byte buffer.
    ///
    /// # Arguments
    /// * `data` - A slice of generic elements (must implement `Element` trait).
    /// * `shape` - A vector defining the dimensions.
    ///
    /// # Panics
    /// Panics if the number of elements in `data` does not match the product of `shape`.
    pub fn from_slice<T: Element>(data: &[T], shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert!(numel == data.len(), "Tensor::from_slice shape mismatch");

        let size = std::mem::size_of::<T>();
        // Create raw bytes from the slice
        let ptr = data.as_ptr() as *const u8;
        let bytes_len = numel * size;
        let mut bytes = Vec::with_capacity(bytes_len);
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, bytes_len);
            bytes.extend_from_slice(slice);
        }

        Self {
            data: bytes,
            shape,
            dtype: T::DTYPE,
            device: Device::CPU,
        }
    }
    
    // Legacy helper (optional, or remove if we update call sites)
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::from_slice(&data, shape)
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// View the tensor data as a typed slice.
    ///
    /// # Safety
    /// This uses `unsafe` to cast raw bytes to type `T`. We verify `dtype` matches `T::DTYPE`
    /// to ensure type safety.
    pub fn as_slice<T: Element>(&self) -> &[T] {
        assert!(self.dtype == T::DTYPE, "dtype mismatch");
        let ptr = self.data.as_ptr() as *const T;
        let len = self.numel();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// View the tensor data as a mutable typed slice.
    pub fn as_slice_mut<T: Element>(&mut self) -> &mut [T] {
        assert!(self.dtype == T::DTYPE, "dtype mismatch");
        let ptr = self.data.as_mut_ptr() as *mut T;
        let len = self.numel();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        self.as_slice::<f32>()
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        self.as_slice_mut::<f32>()
    }
}

// ====================== Ops Wrappers ======================

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor, crate::errors::FrameworkError> {
        crate::ops::elementwise::add(self, other)
}


    pub fn mul(&self, other: &Tensor) -> Result<Tensor, crate::errors::FrameworkError> {
        crate::ops::elementwise::mul(self, other)
}


    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, crate::errors::FrameworkError> {
        crate::ops::matmul::matmul(self, other)
}

    pub fn relu(&self) -> Result<Tensor, crate::errors::FrameworkError> {
        crate::ops::activation::relu(self)
}

    pub fn softmax(&self) -> Result<Tensor, crate::errors::FrameworkError> {
        crate::ops::softmax::softmax(self)
}

}


impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dtype == DType::FP32 {
            write!(
                f,
                "Tensor(shape={:?}, dtype={:?}, device={:?}, data={:?})",
                self.shape,
                self.dtype,
                self.device,
                self.as_f32_slice()
            )
        } else {
            write!(
                f,
                "Tensor(shape={:?}, dtype={:?}, device={:?}, data=[raw bytes])",
                self.shape,
                self.dtype,
                self.device
            )
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dtype {
            DType::FP32 => {
                let data = self.as_slice::<f32>();
                f.debug_struct("Tensor").field("shape", &self.shape).field("dtype", &self.dtype).field("data", &data).finish()
            },
            DType::FP64 => {
                let data = self.as_slice::<f64>();
                f.debug_struct("Tensor").field("shape", &self.shape).field("dtype", &self.dtype).field("data", &data).finish()
            },
            DType::INT32 => {
                 let data = self.as_slice::<i32>();
                 f.debug_struct("Tensor").field("shape", &self.shape).field("dtype", &self.dtype).field("data", &data).finish()
            },
            DType::INT64 => {
                 let data = self.as_slice::<i64>();
                 f.debug_struct("Tensor").field("shape", &self.shape).field("dtype", &self.dtype).field("data", &data).finish()
            },
             DType::UINT8 => {
                 let data = self.as_slice::<u8>();
                 f.debug_struct("Tensor").field("shape", &self.shape).field("dtype", &self.dtype).field("data", &data).finish()
            },
             DType::BOOL => {
                 let data = self.as_slice::<bool>();
                 f.debug_struct("Tensor").field("shape", &self.shape).field("dtype", &self.dtype).field("data", &data).finish()
            },
            _ => {
                f.debug_struct("Tensor")
                .field("shape", &self.shape)
                .field("dtype", &self.dtype)
                .field("device", &self.device)
                .field("data(bytes)", &format!("{} bytes", self.data.len()))
                .finish()
            }
        }
    }
}


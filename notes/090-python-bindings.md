# 090: Python Bindings

## 1. Motivation
Deep Learning is a dual-language discipline.
*   **System Layer (C++/Rust)**: Needs low latency, manual memory management, and SIMD access.
*   **User Layer (Python)**: Needs high flexibility, REPL support, and a vast ecosystem (Matplotlib, Pandas).
We need a way to expose the blazing speed of Rust to the usability of Python without incurring massive overhead.

## 2. Context / Precedence
*   **C-Extensions (CPython)**: The strict C API (`Python.h`) that allows creating modules.
*   **Boost.Python / pybind11**: C++ libraries that simplify binding. Used effectively by PyTorch.
*   **PyO3**: The standard Rust crate for binding to CPython. It leverages Rust's macros to auto-generate the C-API boilerplate.

## 3. Intuition
**The Diplomat**:
*   **Python (The Town Square)**: Rules are loose, everything is an Object, garbage collection happens whenever.
*   **Rust (The Fortress)**: Rules are strict (Borrow Checker), memory is precise.
*   **PyO3 (The Embassy)**: Translates objects at the border.
    *   `PyObject`: A box that Python gives to Rust. "Here, hold this, don't look inside."
    *   `IntoPy`: Rust wrapping a gift for Python.

## 4. Formal Definition
A **Binding** is a mapping from a Rust function $f_R: T \to U$ to a Python function $f_P: \text{PyObject} \to \text{PyObject}$.
The wrapper handles:
1.  **Unboxing**: Checking types of input arguments.
2.  **Conversion**: `PyInt` $\to$ `i64`.
3.  **Execution**: Calling $f_R$.
4.  **Boxing**: `Result<T, E>` $\to$ `PyObject` (or raising Python Exception).

## 5. Mathematical Deep Dive
**Reference Counting**:
CPython uses Reference Counting for memory management.
Every object has an `ob_refcnt` field.
*   `Py_INCREF(obj)`: $+1$.
*   `Py_DECREF(obj)`: $-1$. If 0, free memory.
**PyO3** uses RAII (Resource Acquisition Is Initialization) to handle this. When a Rust `Py<T>` drops, it automatically decrements the refcount.

## 6. Computation / Implementation Details
*   **The GIL (Global Interpreter Lock)**: Only one thread can execute Python bytecode at a time. Rust code usually runs *outside* the GIL (`py.allow_threads`).
*   **FFI Overhead**: Calling a C-function from Python takes ~50-100ns. This is why we don't write `add(scalar, scalar)` in Rust effectively. We want to move *Bulk Data* (Tensors) so the overhead is amortized.
*   **Buffer Protocol**: A C-level protocol that allows Python (NumPy) to see the raw memory of a Rust `Vec<u8>` without copying.

## 7. Minimal Code

### Cargo.toml
```toml
[lib]
crate-type = ["cdylib"]
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
```

### Rust Implementation
```rust
use pyo3::prelude::*;

#[pyclass]
struct Tensor {
    inner: Vec<f32>,
}

#[pymethods]
impl Tensor {
    #[new]
    fn new(data: Vec<f32>) -> Self { Tensor { inner: data } }
    
    fn sum(&self) -> f32 {
        self.inner.iter().sum()
    }
}

#[pymodule]
fn cobalt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    Ok(())
}
```

## 8. Practical Behavior
*   **Release vs Debug**: Python extensions compiled in Debug mode are horrifically slow. Always use `maturin develop --release`.
*   **Errors**: Rust panics become `PyRuntimeError`. Rust `Result::Err` becomes raised Exceptions.

## 9. Tuning / Debugging Tips
*   **`unsafe`**: Almost all FFI is inherently `unsafe` because C pointers can be null or dangling. PyO3 hides this, but raw pointer access (for Buffer Protocol) requires care.
*   **Leaks**: If you create a Reference Cycle (Rust holding Python holding Rust), simple RefCounting won't clean it up. You need Python's Cyclic GC.

## 10. Historical Notes
The success of PyTorch over TensorFlow 1.x was largely due to its "Python First" philosophy. It felt like a Python extension, whereas TF looked like a C++ engine that happened to have Python bindings.

## 11. Variants / Related Forms
*   **ctypes / CFFI**: Calling dynamic libraries from Python without compiled extensions (slower).
*   **WASM**: Compiling Rust to WebAssembly to run in Browser JS (similar concept, different host).

## 12. Examples / Exercises
**Exercise**: Zero-Copy Transfer.
How to give a Rust `Vec<f32>` to NumPy?
1. Obtain the raw pointer `ptr` and `len`.
2. Use `unsafe { PyArray1::from_raw_array(py, ptr, len) }`.
3. Give ownership of the standard layout to Python (Capsule).

## 13. Failure Cases / Limitations
*   **Segfaults**: If Rust code writes to memory that Python thinks it owns (or vice versa) without synchronization.
*   **Pickling**: Rust structs are not automatically pickle-able. You must implement `__getstate__` and `__setstate__`.

## 14. Applications
*   **HuggingFace Tokenizers**: Written in Rust for speed, exposed to Python.
*   **Polars**: DataFrame library written in Rust.

## 15. Connections to Other Concepts
*   **Operating Systems**: Dynamic Linking (`.so`, `.dll`).
*   **Foreign Function Interface (FFI)**: The general concept of languages talking to each other.

## 16. Frontier / Research Angle (Optional)
**Modular AI**: Using ABI-stable interfaces (like `abi_stable` crate) so that plugins compiled with different Rust versions can still talk to each other, decoupling the Python version from the Engine version.

## 17. Glossary of Terms
*   **Wheel (.whl)**: A ZIP archive containing the compiled Python extension.
*   **Maturin**: Build system for Rust-Python extensions.
*   **GIL**: Global Interpreter Lock.

## 18. References / Further Reading
*   [PyO3 User Guide](https://pyo3.rs/)
*   [Python C-API Documentation](https://docs.python.org/3/c-api/)

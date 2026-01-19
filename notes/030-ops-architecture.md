# 030: Operations Architecture

## 1. Motivation
A Deep Learning framework must expose a high-level, dynamic API (like Python) while executing low-level, static machine code (like AVX assembly) for performance. The **Ops Architecture** is the bridge that connects these two worlds. It must handle:
1.  **Type Erasure**: Users don't want to type `Add<Float32>`. They just want `Add`.
2.  **Device Abstraction**: The same code should run on CPU, GPU, and TPU.
3.  **Extensibility**: Adding new operations shouldn't require rewriting the core.

## 2. Context / Precedence
*   **PyTorch (ATen)**: Uses a sophisticated "Dispatcher" based on C++ templates and code generation (`gen_aten.py`). It maps `Tensor` -> `Type/Device` -> `Kernel`.
*   **TensorFlow / XLA**: Uses a Graph-based approach where ops are nodes in a computation graph, compiled entirely before execution (Static).
*   **JAX**: Uses "trace-based" dispatch (jit) to build an XLA expression.

## 3. Intuition
Think of the Ops Architecture as a **Traffic Control Tower**.
*   **The Planes (Tensors)**: Arrive with cargo (Data) and a flight plan (DType/Device).
*   **The Controller (Dispatcher)**: Looks at the flight plan. "You are a specialized Cargo plane (FP32)? Go to Runway 1 (AVX2)." "You are a small private jet (INT8)? Go to Runway 2 (Scalar)."
*   **The Runway (Kernel)**: The actual place where the work happens.

## 4. Formal Definition
An Operation $Op$ is a function mapping a tuple of Input Tensors to Output Tensors.
$$ Op: (T_1, \dots, T_n) \to (O_1, \dots, O_m) $$

The **Dispatch Function** $D$ selects the concrete implementation $K$ based on metadata $M$ (dtype, device, layout).
$$ K = D(Op, M_{T_1}, \dots, M_{T_n}) $$

## 5. Mathematical Deep Dive
**The Expression Problem**:
In Programming language theory, we want to extend two dimensions:
1.  **Data Types** (FP16, BF16, FP32...)
2.  **Operations** (Add, Sub, Conv...)

Object-Oriented approaches make adding types easy but ops hard. Functional approaches make ops easy but types hard.
Cobalt uses **Enum-Based Dispatch** (Sum Types), which makes adding Ops easy (new functions) and adding Types explicit (modifying the Enum match).

## 6. Computation / Implementation Details
*   **The Match Statement**: In Rust, dispatch is usually a giant `match` statement on the `DType` enum.
    ```rust
    match (lhs.dtype, rhs.dtype) {
        (FP32, FP32) => add_f32(lhs, rhs),
        (INT32, INT32) => add_i32(lhs, rhs),
        _ => Err(DTypeMismatch),
    }
    ```
*   **Monomorphization**: The compiler generates a unique copy of the kernel function for each concrete type (`add<f32>`, `add<i32>`). This allows aggressive inlining and vectorization specific to that type.
*   **V-Tables vs Enums**: We avoid V-Tables (dynamic dispatch traits) inside tight loops because they prevent inlining. We dispatch *once* at the top level, then run a fast loop.

## 7. Minimal Code

### User API (Type Erased)
```rust
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    // 1. Check constraints
    if a.shape != b.shape { panic!("Shape mismatch"); }
    
    // 2. Dispatch
    match a.dtype {
        DType::FP32 => {
            // 3. Re-interpret bytes as generic type T
            let a_data = a.data_as::<f32>();
            let b_data = b.data_as::<f32>();
            // 4. Call concrete kernel
            let out_data = kernels::add_slice(a_data, b_data);
            Tensor::new(out_data, a.shape)
        },
        _ => unimplemented!("DType not supported")
    }
}
```

## 8. Practical Behavior
*   **Overhead**: The dispatch cost is roughly 5-10 nanoseconds. For a tensor with 1,000,000 elements, this is negligible. For a scalar tensor (1 element), this is expensive.
*   **Panic Safety**: The dispatcher is the primary line of defense. It panics or errors *before* unsafe memory access occurs.

## 9. Tuning / Debugging Tips
*   **Unsupported DTypes**: The most common error is implementing an Op for `FP32` but forgetting `INT32`. The specific match arm will land in `unimplemented!()`.
*   **Cross-Device Errors**: Dispatchers must verify that all inputs are on the same device. Passing a CPU tensor to a GPU kernel results in a segfault.

## 10. Historical Notes
Early frameworks like **Theano** did strictly symbolic dispatch (compiling C code strings). **Torch7** (Lua) used strict naming conventions (`torch.FloatTensor`, `torch.CudaTensor`). Modern PyTorch consolidates everything into a single `Tensor` class with internal flags.

## 11. Variants / Related Forms
*   **Double Dispatch**: When the operation depends on two types (e.g., typically we don't support `Float` + `Int`, but some languages do type promotion).
*   **Multiple Dispatch (Julia)**: The language itself handles the dispatch logic, arguably making it the "ideal" language for DL.

## 12. Examples / Exercises
**Exercise**: Design a `Cast` operation.
*   Input: `Tensor` (FP32)
*   Output: `Tensor` (INT32)
*   Logic: The dispatcher must accept `dtype=FP32` but call a kernel that produces `dtype=INT32`.
    *   `match src_dtype { FP32 => { cast_to_int(src_data) } }`

## 13. Failure Cases / Limitations
*   **Binary Bloat**: Supporting $N$ types and $M$ ops leads to $N \times M$ compiled kernels. This increases the binary size significantly.
*   **Combinatorial Explosion**: If we support Mixed Precision (FP16 + FP32), the dispatch table grows quadratically.

## 14. Applications
*   **Quantization**: Dispatching to `QInt8` kernels allows models to run 4x smaller and faster on mobile devices.
*   **Device Switching**: Moving a model to GPU `model.to("cuda")` essentially changes the dispatch path for all future operations.

## 15. Connections to Other Concepts
*   **Compilers**: This is essentially a JIT (Just-In-Time) compiler's job, but approximated manually in Rust.
*   **Generics**: Rust's generic system is the compile-time mechanism that powers the runtime dispatch.

## 16. Frontier / Research Angle (Optional)
**JIT Dispatch**: Instead of pre-compiling all kernels, generate them on the fly (like OpenAI Triton). This avoids binary bloat and allows custom fused kernels (`hardswish` + `mem_opt`).

## 17. Glossary of Terms
*   **Dispatch**: Selecting which function to call at runtime.
*   **Kernel**: The tight inner loop that does the math.
*   **Type Erasure**: Hiding the concrete type (`f32`) behind a generic handle (`Tensor`).

## 18. References / Further Reading
*   [PyTorch Internals: The Dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
*   [Rust Book: Enums and Pattern Matching](https://doc.rust-lang.org/book/ch06-00-enums.html)

# 020: Shapes and Math

## 1. Motivation
Deep Learning assumes that operations can be composed. To compose operations safely, we need a rigorous system of rules that defines "compatibility". Just as Type Systems (like Rust's) check compatibility of data types, **Shape Calculus** checks the compatibility of tensor dimensions.

## 2. Context / Precedence
*   **Linear Algebra**: Defines rules for Matrix multiplication ($[N, M] \times [M, K]$).
*   **Einstein Summation (Einsum)**: A concise notation for tensor contraction that generalizes matrix math.
*   **APL / J / K**: Array-oriented languages that pioneered rank polymorphism (functions that work on any dimensionality).

## 3. Intuition
Think of Shapes as **Socket Types** and Operations as **Connectors**.
*   **Elementwise (`+`)**: Requires a "Twin Plug". Both sides must be identical.
*   **MatMul (`@`)**: Requires a "Male-Female Adapter". The output of the left (`M`) must fit the input of the right (`M`).
*   **Broadcasting**: A "Universal Adapter" that stretches a small plug to fit a large socket.

## 4. Formal Definition
Let $S_A = (a_0, \dots, a_{n-1})$ and $S_B = (b_0, \dots, b_{m-1})$.

*   **Equality Compatibility**: $S_A \sim S_B \iff n=m \land \forall i, a_i = b_i$.
*   **Contraction Compatibility (Dim $k$ with $l$)**: Allowed if $a_k = b_l$.
*   **Broadcast Compatibility**: (See Note 070).

## 5. Mathematical Deep Dive
**Rank Polymorphism**:
A function $f: \mathbb{R} \to \mathbb{R}$ (scalar function) can be lifted to operate on tensors $T$ of any rank.
$$ f(T)_{i,j,\dots} = f(T_{i,j,\dots}) $$

**The Shape Algebra of MatMul**:
Operation: $C = A \times B$
Shape Rule:
$$ \text{shape}(A) = [\dots, m, n] $$
$$ \text{shape}(B) = [\dots, n, k] $$
$$ \text{shape}(C) = [\dots, m, k] $$
(Where $\dots$ represents batch dimensions which must be broadcast-compatible).

## 6. Computation / Implementation Details
*   **Shape Storage**: Typically `SmallVec<[usize; 4]>` or `Vec<usize>`. Since most tensors have rank <= 4, stack allocation helps.
*   **Check overhead**: Shape checks happen at runtime in Cobalt (and PyTorch). This adds a small overhead (ns) compared to Static Shapes (like in specialized compilers XLA), but enables dynamic graphs.

## 7. Minimal Code

### Rust (Cobalt)
```rust
fn check_shape_match(a: &Tensor, b: &Tensor) -> Result<(), Error> {
    if a.shape != b.shape {
        return Err(Error::ShapeMismatch(a.shape.clone(), b.shape.clone()));
    }
    Ok(())
}
```

### Python (Debugging)
```python
import torch
a = torch.randn(32, 100)
b = torch.randn(32, 100)
# c = torch.matmul(a, b) # RuntimeError: size mismatch, [32, 100] x [32, 100]
c = torch.matmul(a, b.T) # OK: [32, 100] x [100, 32] -> [32, 32]
```

## 8. Practical Behavior
*   **Reduction Ops**: Operations like `sum(dim=1)` *remove* a dimension. Shape goes from $[32, 10] \to [32]$.
*   **KeepDim**: Often useful to keep rank: `sum(dim=1, keepdim=True)` -> Shape $[32, 1]$.

## 9. Tuning / Debugging Tips
*   **Dimensional Analysis**: If you are unsure of a bug, write down the shapes on paper.
    *   Input: $[B, T, C]$
    *   Linear: $[C, H]$
    *   Output: $[B, T, H]$
*   **The "-1" Trick**: In `reshape`, using `-1` asks the framework to infer the missing dimension. `reshape(new_shape=[-1, 10])`. Use sparingly as it hides bugs.

## 10. Historical Notes
The notation of shape lists `[N, M]` became standard with NumPy. Previous math software often treated everything as matrices, leading to awkward handling of higher-order tensors (e.g., having to flatten an image into a vector).

## 11. Variants / Related Forms
*   **Symbolic Shapes**: Frameworks like SymPy or compiler stacks (MLIR) track shapes as symbols ($N, C, H, W$) rather than concrete numbers, allowing for Ahead-of-Time compilation.
*   **Dynamic Shapes**: Where a dimension is unknown until runtime (e.g., text generation length).

## 12. Examples / Exercises
**Exercise**: Convolution Output Shape
Equation: $O = \lfloor \frac{I - K + 2P}{S} \rfloor + 1$
*   Input ($I$): 224
*   Kernel ($K$): 3
*   Padding ($P$): 1
*   Stride ($S$): 2

*Result*: $\lfloor \frac{224 - 3 + 2}{2} \rfloor + 1 = \lfloor 111.5 \rfloor + 1 = 111 + 1 = 112$.

## 13. Failure Cases / Limitations
*   **Ambiguous Operations**: Operations like `view()` can fail if the stride logic implies non-contiguous memory, requiring a `reshape()` or `contiguous()` call first.
*   **Silent Broadcasting**: Sometimes broadcasting happens when you didn't intend it (adding a `[32]` bias to a `[32, 32]` matrix row-wise instead of column-wise), leading to logically wrong but runnable code.

## 14. Applications
*   **Reshaping for Attention**: Transforming `[Batch, Seq, Head * Dim]` into `[Batch, Seq, Head, Dim]` to compute attention per-head.
*   **Flattening**: Converting dimensions `[Batch, C, H, W]` to `[Batch, C*H*W]` before a fully connected classifier.

## 15. Connections to Other Concepts
*   **Type Theory**: Shapes can be viewed as "Dependent Types" (types that depend on values).
*   **Group Theory**: Permutations of axes (Transposes) form a symmetric group.

## 16. Frontier / Research Angle (Optional)
**Compile-Time Shape Safety**: Languages like Dex or Rust crates (like `dfdx`) try to enforce shape constraints at compile time using advanced generic const expressions, preventing runtime crashes entirely.

## 17. Glossary of Terms
*   **Rank**: Number of axes.
*   **Dimension / Axis**: A specific direction in the tensor.
*   **Reshape**: Changing logical shape without changing data order.
*   **Permute / Transpose**: Swapping dimensions (changes stride order).
*   **Flatten**: reducing a tensor to 1D.

## 18. References / Further Reading
*   [A Guide to NumPy (Travis Oliphant)](https://web.mit.edu/dvp/Public/numpybook.pdf)
*   [The Tensor Contraction Engine](https://en.wikipedia.org/wiki/Tensor_contraction)

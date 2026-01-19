# 060: Activations and Softmax

## 1. Motivation
Linear transformations ($Wx + b$) can only represent linear relationships. Stacking 100 linear layers is mathematically equivalent to 1 linear layer.
To learn complex patterns (curves, decision boundaries), we need **Non-Linearity**. Activation functions introduce "kinks" in the geometry of the data.
Softmax is motivated by the need to interpret raw primitive outputs (logits) as a valid Probability Distribution.

## 2. Context / Precedence
*   **Biological Inspiration**: Neurons fire (Action Potential) only when input exceeds a threshold (All-or-Nothing).
*   **Sigmoid Era**: Early nets used Sigmoid, but suffered from Vanishing Gradients.
*   **ReLU Revolution (2011)**: Changed everything by fixing the gradient problem.

## 3. Intuition
**ReLU (The Filter)**:
It says "No negativity allowed."
*   Input 5 $\to$ Output 5.
*   Input -5 $\to$ Output 0.
This simple clipping allows partial networks to "turn off" for certain inputs, creating sparsity.

**Softmax (The Pie Chart)**:
It forces a competition.
If Output A gets bigger, Output B *must* get smaller, because the total pie (100%) is fixed.

## 4. Formal Definition
**ReLU**:
$$ f(x) = \max(0, x) $$

**Softmax**:
Given a vector $z = [z_1, \dots, z_K]$:
$$ \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} $$

## 5. Mathematical Deep Dive
**Softmax Derivative**:
The Jacobian matrix of Softmax is complex. However, when combined with **Cross-Entropy Loss**, the derivative simplifies beautifully to:
$$ \frac{\partial L}{\partial z_i} = p_i - y_i $$
(Prediction - Target). This property makes training remarkably stable.

**Temperature in Softmax**:
$$ \sigma(z/T)_i $$
*   $T \to 0$: Argmax (One-Hot).
*   $T \to \infty$: Uniform Distribution.

## 6. Computation / Implementation Details
*   **Numerical Stability (Log-Sum-Exp Trick)**:
    Calculating $e^{1000}$ results in `Infinity`.
    We use the identity:
    $$ \frac{e^{z_i}}{\sum e^{z_j}} = \frac{e^{z_i - C}}{\sum e^{z_j - C}} $$
    Where $C = \max(z)$. This shifts all values to be $\le 0$ before exponentiation, preventing overflow.
*   **Vectorization**: Softmax requires a Reduction (`max`, `sum`) followed by Elementwise Ops (`exp`, `div`).

## 7. Minimal Code

### ReLU (Rust)
```rust
fn relu(x: &[f32], out: &mut [f32]) {
    for (i, v) in x.iter().enumerate() {
        out[i] = v.max(0.0);
    }
}
```

### Stable Softmax (Python)
```python
def softmax(x):
    # x is [N]
    c = np.max(x)
    exp_x = np.exp(x - c) # Stability shift
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x
```

## 8. Practical Behavior
*   **Dead ReLUs**: If a neuron's weights update such that it always outputs negative values for all inputs, it produces 0 gradient forever. It "dies". LeakyReLU fixes this.
*   **Softmax Saturation**: If logits are very large, Softmax outputs are strictly 1.0 or 0.0.

## 9. Tuning / Debugging Tips
*   **Logits vs Probs**: Always prefer working with Logits (raw scores) for Loss calculation (`CrossEntropyLoss` takes logits). Taking `log(softmax(x))` is numerically unstable if softmax outputs 0.
*   **GELU**: In transformers, GELU (Gaussian Error Linear Unit) is preferred over ReLU. It's a smoother curve.

## 10. Historical Notes
The discovery that ReLU (simplest possible function) outperformed Sigmoid (complex exponential) was counter-intuitive but proved that **optimization landscape** matters more than biological plausibility.

## 11. Variants / Related Forms
*   **LeakyReLU**: $f(x) = \max(\alpha x, x)$. allows small negative gradients.
*   **GeLU**: $x \Phi(x)$. Used in BERT/GPT.
*   **Swish**: $x \cdot \sigma(x)$.

## 12. Examples / Exercises
**Exercise**: Compute Softmax.
Input: $[2.0, 1.0, 0.1]$
1.  Exp: $[7.389, 2.718, 1.105]$
2.  Sum: $11.212$
3.  Div: $[0.659, 0.242, 0.099]$
Sum check: $0.659+0.242+0.099 = 1.0$.

## 13. Failure Cases / Limitations
*   **Exploding Gradients**: Unbounded functions like ReLU can allow activations to grow infinitely if weights are not initialized properly (He Initialization).

## 14. Applications
*   **Hidden Layers**: ReLU/GELU.
*   **Final Layer (Binary)**: Sigmoid.
*   **Final Layer (Multi-Class)**: Softmax.

## 15. Connections to Other Concepts
*   **Information Theory**: Softmax output represents a categorical distribution maximizing entropy under constraints.
*   **Physics**: Boltzmann Distribution ($e^{-E/kT}$).

## 16. Frontier / Research Angle (Optional)
**Squared ReLU**: Google used `ReLU^2` in recent LLMs (PaLM). It gives slightly stronger gradients.

## 17. Glossary of Terms
*   **Logits**: Raw, unnormalized scores output by the last linear layer.
*   **Saturation**: When the gradient becomes close to 0 (flat regions of the curve).

## 18. References / Further Reading
*   [Deep Sparse Rectifier Neural Networks (Glorot et al, 2011)](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
*   [Visualizing Activations](https://distill.pub/)

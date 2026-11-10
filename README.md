# Deep Learning Architectures (dla)

### Core Project Roadmap

This repository bypasses high-level wrappers like Hugging Face or PyTorch Lightning to build deep learning components completely from scratch using raw tensors. By implementing the underlying linear algebra and automatic differentiation loops directly, this project showcases a rigorous understanding of both structural engineering and mathematical statistics.

---

### Phase 1: Under-the-Hood Foundations

Before scaling up to complex architectures, every deep learning model relies on a fundamental optimization engine. The table and code snippets below demonstrate how PyTorch abstracts basic scalar calculus into multi-layer systems.

| Low-Level Scalar Concept | High-Level Neural Network Equivalent | Core Engineering Purpose |
| :--- | :--- | :--- |
| `w = torch.tensor(..., requires_grad=True)` | `model = nn.Sequential(...)` | Explicitly declares tracking for trainable weights. |
| `loss = (w - 3).square()` | `loss = criterion(predictions, y)` | Formulates the objective error function against a target goal. |
| `loss.backward()` | `loss.backward()` | Computes analytical partial derivatives via the chain rule. |
| `w -= 0.1 * w.grad` | `optimizer.step()` | Steps the model parameters in the direction of lower loss. |
| `w.grad.zero_()` | `optimizer.zero_grad()` | Flushes accumulated gradients before the next iteration. |

<details>
<summary><b>View Minimal PyTorch Foundations Code Examples</b></summary>

#### Single Scalar Variable Optimization
```python
import torch

w = torch.tensor(0.0, requires_grad=True)

for step in range(20):
    loss = (w - 3).square()
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
        w.grad.zero_()
    print(f"Step {step}: w = {w.item():.4f}, Loss = {loss.item():.4f}")
```

#### Multi-Layer Neural Network Equivalence
```python
import torch
import torch.nn as nn
import torch.optim as optim

X, y = torch.randn(100, 5), torch.randn(100, 1)

model = nn.Sequential(
    nn.Linear(5, 12),
    nn.ReLU(),
    nn.Linear(12, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```
</details>

---

### Phase 2: The Three Pillars of Implementation

#### 1. Custom Transformer Block (Raw Tensors)
Instead of a multi-layer GPT clone, this focuses on engineering a singular, isolated **Transformer Encoder Block** from fundamental math principles.
* **Multi-Head Attention (MHA):** Hand-rolled linear mappings for Q, K, and V. It accounts for explicit tensor dimension tracking `(B, T, C) -> (B, H, T, d_k)` and manual scaling of attention scores. Includes causal and padding masking by injecting -∞ values before the softmax function.
* **Custom LayerNorm & Pre-LN Layout:** Implements mean and variance tracking over the last dimension with numerical stability (ε), learnable γ scale, and β bias. Features a Pre-LN structural layout (x → Norm → Sublayer → Residual) to protect training loops from exploding or vanishing gradients.
* **Feed-Forward Network (FFN):** A structured sub-layer mapping `Inner Dim = 4 * d_model` leveraging GELU/ReLU activations, complete with explicit residual connections and dropout layers.

```python
# Core block execution architecture
x = x + Dropout(MHA(LN(x)))
x = x + Dropout(FFN(LN(x)))
```

#### 2. Advanced Loss Functions & Gradient Analysis
This pillar demonstrates a deep statistics background by bypassing automated autograd for specialized loss curves.
* **Focal Loss Implementation:** Formulates specialized classification loss to handle heavy class imbalances: 
  \[FL(p_t) = -\alpha(1 - p_t)^\gamma \log(p_t)\]
* **Mathematical Verification:** The `forward` pass is carefully written in log-space for explicit floating-point stability. 
* **Gradient Checking:** Features custom analytical closed-form derivations of partial derivatives with respect to logits, verified line-by-line via `torch.autograd.grad`.

#### 3. The Bare-Metal Training Loop
A singular, transparent training function written completely without training-class wrappers or boilerplate abstractions.
* **State Management:** Strict toggle logic across `model.train()` and `model.eval()` contexts alongside `torch.no_grad()` checking.
* **Instability Defense:** Active logging of raw `grad_norm` metrics ahead of explicit gradient clipping to actively analyze training health.
* **Optimization Scheduling:** Powered by an AdamW optimizer alongside a custom Cosine Learning Rate Warmup schedule to securely bound massive Transformer logit spikes.

```python
# Raw explicit optimization cycle
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Monitor instability and clip bounds
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```
### The mathematics behind it all is covered in the [Mathematics.md](./Mathematics.md)
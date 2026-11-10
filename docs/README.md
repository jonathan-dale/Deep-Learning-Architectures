## Understanding the Mechanics of what happens Under-the-Hood

Deep learning architectures can look complex, but they all rely on the exact same loop:  
**Forward Pass $\rightarrow$ Loss Calculation $\rightarrow$ Backpropagation $\rightarrow$ Weight Update**.

Here is how that process evolves from a single scalar parameter up to a multi-layer neural network.

### 1. The Bare Minimum: Optimizing a Single Variable
This minimal snippet isolates how PyTorch uses its autograd engine to track a variable, calculate a gradient using the chain rule, and update it using basic gradient descent without any abstractions.

```python
import torch

# Initialize a single weight tracking gradients
w = torch.tensor(0.0, requires_grad=True)

for step in range(20):
    # 1. Forward Pass / Objective function
    loss = (w - 3).square()
    
    # 2. Backpropagation (computes derivative of loss w.r.t w)
    loss.backward()
    
    # 3. Weight Update (manually step against the gradient)
    with torch.no_grad():
        w -= 0.1 * w.grad
        w.grad.zero_()  # Clear gradients for the next iteration
        
    print(f"Step {step}: w = {w.item():.4f}, Loss = {loss.item():.4f}")
```

### What this illustrates:
- **`requires_grad=True`**: Enables automatic differentiation (autograd) to track operations on tensors.
- **`loss.backward()`**: Computes $\frac{\partial \text{loss}}{\partial w}$ analytically via the chain rule.
- **`w -= 0.1 * w.grad`**: The basic optimizer step (similar to `torch.optim.SGD`).

---

### 2. Scaling Up: A Multi-Layer Neural Network
When building actual deep learning models, we swap manual tracking for PyTorch modules (`nn.Linear`) and automated optimizers (`optim.SGD`). The underlying mathematics remain exactly the same as the single-variable example above.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulated dataset: 100 samples, 5 features each
X = torch.randn(100, 5)
y = torch.randn(100, 1)

# A simple 2-layer Neural Network MLP
model = nn.Sequential(
    nn.Linear(5, 12),   # Input Layer -> Hidden Layer
    nn.ReLU(),          # Activation Function
    nn.Linear(12, 1)    # Hidden Layer -> Output Layer
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# The Training Loop
for epoch in range(20):
    # 1. Forward Pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # 2. Backpropagation
    optimizer.zero_grad()  # Replaces 'w.grad.zero_()'
    loss.backward()        # Calculates gradients across all layers
    
    # 3. Weight Update
    optimizer.step()       # Replaces 'w -= lr * w.grad'
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### Side-by-Side Comparison

| Low-Level Scalar Concept | High-Level Neural Network Equivalent | Purpose |
| :--- | :--- | :--- |
| `w = torch.tensor(..., requires_grad=True)` | `model = nn.Sequential(...)` | Defines the trainable parameters/weights. |
| `loss = (w - 3).square()` | `loss = criterion(predictions, y)` | Measures model error against a target goal. |
| `loss.backward()` | `loss.backward()` | Calculates partial derivatives via the chain rule. |
| `w -= 0.1 * w.grad` | `optimizer.step()` | Updates weights in the direction of lower loss. |
| `w.grad.zero_()` | `optimizer.zero_grad()` | Flushes old gradients so they don't accumulate. |



### Environment setup

Before you run the code blocks above, open a python interpreter, and run this environment setup script; if it works then copy and paste the code blocks above to see it in action.

#### Open up a python interperter
```shell
uv sync
source .venv/bin/activate
uv run python
Python 3.13.15 (main, Aug 14 2026, 15:24:40) [Clang 22.1.3 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
# Run the code below to test the environment first....
# Copy the code block into the python interpreter at the >>> prompt
```

### A simple test
This is to ensure that the environment is set up for this project:
```python
"""
This is a simple environment test for PyTorch.
Copy this into the python interpreter, verify the output shape matches (2, 4, 6)
"""
import torch
x = torch.randn(2, 4, 6, requires_grad=True)
y = x.mean()
y.backward()
print(x.grad.shape) # should be (2, 4, 6)
```


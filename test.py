"""
This is a simple environment test for PyTorch.
Run it with:
`uv run python3 test.py`
"""
import torch
x = torch.randn(2, 4, 6, requires_grad=True)
y = x.mean()
y.backward()
print(x.grad.shape) # should be (2, 4, 6)

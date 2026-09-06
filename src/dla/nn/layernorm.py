"""Layer Norm
Returns:
    torch.Tensor: The normalized tensor acros the last (hidden) dimension.
    
    Compute the mean and variance for each final axis vector.
    Normalize with sqrt(var(x) + eps).
    Apply learned elementwise scale (gamma) and shift (beta).
    Return an output with the same shape as the input x.
     
    
    gamma: a trainable per-feature scale, conventionally called weight in PyTorch.
    beta: a trainable per-feature shift, conventionally called bias.
    eps: prevents numerical problems when the variance is very small (i.e. devision by zero).
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape)) # weights/scale
        self.beta = nn.Parameter(torch.zeros(normalized_shape)) # biasses/shift


    # New: optimized for speed and memory efficiency by calculating var/mean together
    # PyTorch calls forward() whenever you invoke the module as layer_norm(x).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(
            x, 
            dim=-1, 
            keepdim=True, 
            correction=0
        )
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


    # # Original layer_norm calculate mu, sigma, and variance explicitly
    # def layer_norm(self, x: torch.Tensor) -> torch.Tensor:
    #     n = x.size(-1)
    #     mu = torch.sum(x, dim=-1, keepdim=True) / n
    #     sigma = (torch.sum((x - mu) ** 2, dim=-1, keepdim=True) / n) ** 0.5
    #     variance = sigma ** 2
    #     x_hat = (x - mu) / (variance + self.eps) ** 0.5
    #     return self.gamma * x_hat + self.beta


    # # PyTorch calls forward() whenever you invoke the module as layer_norm(x).
    # # Using forward directly instead of a separately named layer_norm method is stylistic; both work.    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.layer_norm(x)

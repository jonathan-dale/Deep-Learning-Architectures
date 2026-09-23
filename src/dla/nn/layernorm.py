"""Layer Norm
Returns:
    torch.Tensor: The normalized tensor acros the last (hidden) dimension.
    Unlike BatchNorm, LayerNorm does not use statistics aggregated across the batch; each individual token vector 
    gets normalized independently.
    
    Compute the mean and variance for each final axis vector.
    Normalize with sqrt(var(x) + eps).
    Apply learned elementwise scale (gamma) and shift (beta).
    Return an output with the same shape as the input x.
     
    
    gamma: a trainable per-feature scale, conventionally called weight in PyTorch.
    beta: a trainable per-feature shift, conventionally called bias.
    eps: prevents numerical problems when the variance is very small (i.e. devision by zero).
    
    Unlike BatchNorm, LayerNorm dose not use statistics aggerated across a batch, instead each
    token vector gets normalized independently. PyTorches nn.LayerNorm applies the normalization 
    over the training dimemsions to learn the affine parameters.
    
    For example, if x.shape == (8, 512, 768), then LayerNorm computes:
    8 * 512 = 4,096 independent means,
    4,096 independent variances,
    each based on the 768 hidden features of one token.
    
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape)) # weights/scale
        self.beta = nn.Parameter(torch.zeros(normalized_shape)) # biasses/shift


    # PyTorch calls forward() whenever you invoke the module as layer_norm(x).
    # Shorter and avoids computing the standard deviation (sigma) and then squaring it.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var, mean = torch.var_mean(
            x, 
            dim=-1, 
            keepdim=True, 
            correction=0 # correction=0 uses the population/biased variance used by PyTorch’s nn.LayerNorm
        )
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


    # # Original forward pass calculate mu, sigma, and variance explicitly
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     n = x.size(-1)
    #     mu = torch.sum(x, dim=-1, keepdim=True) / n
    #     sigma = (torch.sum((x - mu) ** 2, dim=-1, keepdim=True) / n) ** 0.5
    #     variance = sigma ** 2
    #     x_hat = (x - mu) / (variance + self.eps) ** 0.5
    #     return self.gamma * x_hat + self.beta


import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        # RMSNorm only only uses gamma (weights/scale), beta (biasses/shift) is dropped for speed and to save memory conservation.
        self.gamma = nn.Parameter(torch.ones(self.normalized_shape))


    """
    First calculate the mean of the squared values along the last dimension
    Take the square root to get the RMS
    Normalize and scale by gamma
    """
    # def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
    #     means = torch.mean(x.pow(2), dim=-1, keepdim=True)
    #     inverted_rms = torch.rsqrt(means + self.eps)
    #     return self.gamma * x * inverted_rms
    
    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        inverted_rms = torch.rsqrt(
                x.pow(2).mean(dim=-1, keepdim=True) + self.eps
            )
        return x * inverted_rms * self.gamma
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rms_norm(x)
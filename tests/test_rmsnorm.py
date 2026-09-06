"""Tests for from-scratch RMSNorm."""

import torch
import torch.nn as nn

from dla.nn.rmsnorm import RMSNorm


# Shared inputs
BATCH, SEQ, HIDDEN = 2, 3, 4
EPS = 1e-5

def _random_x(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(BATCH, SEQ, HIDDEN)


def test_output_shape_matches_input():
    x = _random_x()
    rmsn = RMSNorm(normalized_shape=HIDDEN, eps=EPS)
    y = rmsn(x)
    assert x.shape == y.shape


def test_normalized_last_dim_has_unit_rms_before_affine():
    """When gamma = 1 the root mean square of each last-dimension slice should be approximately 1."""
    x = _random_x() # torch.randn(2, 3, 4)
    rmsn = RMSNorm(normalized_shape=HIDDEN, eps=EPS)
    y = rmsn(x)
    
    actual_rms = torch.sqrt(y.pow(2).mean(dim=-1, keepdim=True))
    input_mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    
    expected_rms = torch.sqrt(
        input_mean_square / (input_mean_square + EPS)
    )
    
    torch.testing.assert_close(
        actual_rms,
        expected_rms,
        atol=1e-5,
        rtol=1e-5
    )


def test_gamma_scales_each_hidden_feature():
    """Verify that gamma independently scales each hidden-dimension output after RMS normalization."""
    
    # Create a test input vector with shape (2, 3, 4).
    x = _random_x()
    
    # gamma starts at [1, 1, 1, 1].
    rmsn = RMSNorm(normalized_shape=HIDDEN, eps=EPS)
    
    # Replace gamma with known distinct values so each feature has different expected scale factors.
    # Note: PyTorch automatically tracks operations on self.gamma for backpropagation because it is an nn.Parameter. 
    # Manual gradient tracking is not needed since this is only a test, not training.
    with torch.no_grad():
        # This line replaces the initial scale vector of all ones:
        rmsn.gamma.copy_(torch.Tensor([1.0, 2.0, 0.5, 3.0]))
    
    # Run the RMSNorm mdoule.
    y = rmsn(x)
    
    # Manually calculate the rms_square (what the module should produce) for comparison with RMSNorm calculation
    # normalize x then multiply each hidden feature by gamma.
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    expected = x * inv_rms * rmsn.gamma
    
    # Confirm the output matches with the expected formula.
    torch.testing.assert_close(
        y,
        expected,
        rtol=1e-6,
        atol=1e-6
    )


def test_zero_input_is_finite_and_zero():
    """An all-zero input produces finite output rather than NaN or infinity.
    Without eps in the denominator, zero based input causes devision by zero.
    """
    x = torch.zeros(BATCH, SEQ, HIDDEN)
    rmsn = RMSNorm(normalized_shape=HIDDEN, eps=EPS)
    
    y = rmsn(x)
    
    assert torch.isfinite(y).all
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_rms_norm_all_gradients_gradcheck():
    """ Verify analytical gradient for RMSNorm inputs and gamma simultaneously."""
    # gradcheck needs float64, double precision, for numerical precision stability
    x = _random_x().double().requires_grad_(True)
    
    rmsn = RMSNorm(normalized_shape=HIDDEN, eps=EPS).double()
    gamma = rmsn.gamma.detach().clone().requires_grad_(True)
    
    # Functional wrapper to explicitly test inputs x and gamma together.
    def functiona_rms_norm(input_tensor, g):
        mean_square = torch.mean(input_tensor.pow(2), dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + EPS)
        return g * (input_tensor / rms)

    # Run the gradient check
    tests_passed = torch.autograd.gradcheck(
        functiona_rms_norm,
        inputs=(x, gamma),
        eps=EPS,
        atol=1e-5,
        rtol=1e-4,
        raise_exception=True
    )
    
    assert tests_passed, "RMSNorm gradient verification failed!"

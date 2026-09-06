"""Tests for from-scratch LayerNorm.

A useful pattern for numeric modules:

1. Shape: the tensor that comes out has the shape you expect.
2. Invariants: properties that must hold even if the numbers look messy
   (here: last-dim mean ≈ 0 and std ≈ 1 before the affine scale/shift).
3. Oracle: match a trusted implementation (`torch.nn.LayerNorm`) on random
   inputs. This is the main correctness check.
4. Gradients: `backward()` produces a grad with the same shape as the input.

Run from the repo root:

    uv run pytest tests/test_layernorm.py -q

"""


from math import gamma
import pytest
import torch
import torch.nn as nn

from torch import float64
from dla.nn.layernorm import LayerNorm


# Shared random input: (batch, seq, hidden). LayerNorm is usually over hidden.
BATCH, SEQ, NORMALIZED_SHAPE = 2, 4, 8
EPS = 1e-5


def _random_x(
    batch: int = BATCH,
    seq: int = SEQ,
    dim: int = NORMALIZED_SHAPE,
    seed: int = 0
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, seq, dim)


# shape test
def test_output_shape_matches_input():
    x = _random_x()
    # Build the module (call the constructor), then call the module on x
    # specifically name the constructor "ln = LayerNorm(.....)", then call it on x "y = ln(x)"
    # This one line will used in the rest of the file: 
    # y = LayerNorm(normalized_shape=HIDDEN, eps=EPS)(x)
    ln = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=EPS)
    y = ln(x)
    assert y.shape == x.shape, "Output shape should match input shape failed!"


def test_layer_norm_last_dim_is_normalized():
    """With gamma=1, beta=0, and eps=0, each last-dim slice has mean 0 and std 1.
    With eps>0 the population std is sqrt(var)/sqrt(var+eps), slightly below 1.
    """
    x = _random_x()
    y = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=0.0)(x)

    mean = y.mean(dim=-1)
    std = y.std(dim=-1, unbiased=False)

    # test a tensor of all zeros with the exact shape and data type as "mean"
    # test a tensor of all ones with the exact shape and data type as "std"
    # atol=1e-5: absolute tolerance, allows for very small differences (up to 0.00001) due to floating point rounding errors.
    # rtol=1e-5: relative tolerance, scales the allowed difference based on the size of the larger values.
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(std, torch.ones_like(std), atol=1e-5, rtol=1e-5)


def test_layer_norm_with_different_dims():
    for dim in [1, 2, 3, 4, 5, 16]:
        x = _random_x(BATCH, SEQ, dim)
        # dim=1 has variance of 0, so using eps=0 makes devision by 0 return NaN. Keep a small eps > 0 for that case.
        y = LayerNorm(normalized_shape=dim, eps=0.0 if dim > 1 else EPS)(x)
        
        assert y.shape == x.shape
        mean = y.mean(dim=-1)        
        torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=1e-5)
        if dim == 1:
            torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-5, rtol=1e-5)
        else:
            std = y.std(dim=-1, unbiased=False)
            torch.testing.assert_close(std, torch.ones_like(std), rtol=1e-5, atol=1e-5)


def test_layer_norm_matches_pytorch_layernorm():
    """Random tensors vs nn.LayerNorm over the last dimension."""
    x = _random_x()
    ours = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=EPS)
    ref = nn.LayerNorm(NORMALIZED_SHAPE, eps=EPS, elementwise_affine=False)
    torch.testing.assert_close(ours(x), ref(x), atol=1e-5, rtol=1e-5)


def test_hand_computed_vector():
    """Tiny example you can check with a calculator.

    x = [1, 2, 3]
    mean = 2
    var  = ((-1)^2 + 0^2 + 1^2) / 3 = 2/3
    y    = (x - 2) / sqrt(2/3 + eps)
    """
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = LayerNorm(normalized_shape=3, eps=EPS)(x)

    mean = torch.tensor(2.0)
    var = torch.tensor(2.0 / 3.0)
    expected = (x - mean) / torch.sqrt(var + EPS)

    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)


def test_affine_scale_and_shift_are_applied():
    x = _random_x()
    x_norm = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=EPS)(x)
    
    ln = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=EPS)
    gamma = torch.arange(1, NORMALIZED_SHAPE + 1, dtype=x.dtype)
    beta = torch.linspace(-1.0, 1.0, NORMALIZED_SHAPE)
    with torch.no_grad():
        ln.gamma.copy_(gamma)
        ln.beta.copy_(beta)

    torch.testing.assert_close(ln(x), gamma * x_norm + beta, atol=1e-5, rtol=1e-5)


def test_normalization_is_only_over_last_dimention():
    x = _random_x()
    ln = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=0.0)
    y = ln(x)
    
    other_axis_mean = y.mean(dim=0)
    assert not torch.allclose(
        other_axis_mean,
        torch.zeros_like(other_axis_mean),
        atol=1e-3,
    )
    
    x_perturbed = x.clone()
    x_perturbed[0] += 1000.0
    torch.testing.assert_close(y[1], ln(x_perturbed)[1], atol=1e-6, rtol=1e-6)


def test_layer_norm_accepts_different_input_ranks():
    ln = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=0.0)
    shapes = [(NORMALIZED_SHAPE,), (BATCH, NORMALIZED_SHAPE), (BATCH, SEQ, NORMALIZED_SHAPE), (2, 2, 3, NORMALIZED_SHAPE)]
    # shapes = [(8,), (2, 8), (2, 4, 8), (2, 2, 3, 8)]
    for shape in shapes:
        print(f"shape = {shape}")
        x = torch.randn(*shape)
        y = ln(x)
        assert y.shape == x.shape
        mean = y.mean(dim=-1)
        std = y.std(dim=-1, unbiased=False)
        torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(std, torch.ones_like(std), atol=1e-5, rtol=1e-5)


def test_constant_last_dim_is_finite_and_zero():
    x = torch.full((BATCH, SEQ, NORMALIZED_SHAPE), 7.0)
    y = LayerNorm(normalized_shape=NORMALIZED_SHAPE, eps=EPS)(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-5, rtol=1e-5)


def test_mismatched_feature_dim_raises():
    ln = LayerNorm(normalized_shape=NORMALIZED_SHAPE)
    x = torch.randn(BATCH, SEQ, NORMALIZED_SHAPE + 3)
    with pytest.raises(RuntimeError):
        ln(x)


def test_gamma_and_beta_receive_gradients():
    x = _random_x()
    ln = LayerNorm(normalized_shape=NORMALIZED_SHAPE)
    ln(x).sum().backward()
    
    assert ln.gamma.grad is not None
    assert ln.beta.grad is not None
    assert ln.gamma.grad.abs().sum() > 0
    assert ln.beta.grad.abs().sum() > 0





"""
To test the backward pass, I am using a built in PyTorch utility called `torch.autograd.gradcheck`.
It takes a custom layer, runs the forward pass, and computes the analytical gradients using autograd. 

Next the inputs are slightly shifted by a tiny value, epsilon=1e-5, to calculate numerical gradients 
using finite differences. Finally, it compares both results to ensure they match up to a defined precision.

What happens under the hood at execution time:
    1. `gradcheck` calculates the partial deritives of L w.r.t. x, gamma, and beta
    using the autograd graph generated by the _layer_norm().

    2. Then adjustes every ellement in the imput vector by a small step (1e-6).

    3. Validates the math matches the output of the algorithim. (See Mathematics.md)

    4. `gradcheck` raises a detailed error showing exactaly where the numerical and
    analytical gradients diverged.


A note regarding Precision:
    Neural nets usually run on standard 32-bit floats (float32). However, numerical gradient checking requires 
    incredibly high precision (float64) to prevent small floating point rounding errors from failing the test. 
    The input layers are casted to .double() here due to this precision requirement.

"""
def test_layer_norm_backward_gradcheck():
    """Verify LayerNorm analytical gradients against numerical gradients using gradcheck."""
    
    # 1. Set up percision using double percision (float64) for numerical stability. 
    # Since gradcheck in computationally heavy, this is intentionally small.
    batch, seq, norm_shape = 2, 3, 4
    
    # 2. Initalize an input vercotor with double precision and requires_grad=True.
    torch.manual_seed(32)
    x = torch.randn(batch, seq, norm_shape, dtype=torch.float64, requires_grad=True)
    
    # 3. Initalize the layer and convert its parameters to double precision.
    ln = LayerNorm(normalized_shape=norm_shape, eps=EPS).double()
    
    # 4. Define a wrapper function for gradcheck
    def run_layer_norm(input_tensor):
        return ln(input_tensor)
    
    # 5. Run the gradient check
    # eps: the steps size for finite differences
    # atol/rtol: absolute/relative tolerance for computing numerical vs analytical gradients
    tests_passed = torch.autograd.gradcheck(
        run_layer_norm,
        inputs=(x,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
        raise_exception=True,
    )

    assert tests_passed, "LayerNorm backward pass tesst failed!"



"""
By default torch.autograd.gradcheck only checks the inputs you explicitly passed 
in to the inputs tuple. However, gamma and beta are internal parameters of the nn.Module,
standard gradcheck on the module forward pass can sometimes skip them.

In order to explicitly test the gradients for the inputs, gamma, and beta all at the 
same run, you can pass a custom function to gradcheck that accepts the parameters as 
direct inputs. 

"""
def test_layer_norm_all_gradients_gradcheck():
    """ Verify all gradients for inputs, gamma, beta, simultaneously."""
    batch, seq, norm_shape = 2, 3, 4
    eps = 1e-5
    
    torch.manual_seed(32)
    # 1. Inputs require gradient
    x = torch.randn(batch, seq, norm_shape, dtype=float64, requires_grad=True)
    
    # 2. Instantiate the layer and extract the parameters
    ln = LayerNorm(normalized_shape=norm_shape, eps=eps).double()
    
    # Force the internal parameters to require gradients for this check
    gamma = ln.gamma.detach().clone().requires_grad_(True)
    beta = ln.beta.detach().clone().requires_grad_(True)
    
    # 3. Create a functional wrapper that accepts parameters as arguements 
    # This forces gradcheck to evaluate and test gradients for all three tensors
    def functional_layer_norm(input_tensor, g, b):
        # Manually preform the normalization step using the passed in g and b
        std, mean = torch.std_mean(input_tensor, dim=-1, keepdim=True, unbiased=False)
        x_hat = (input_tensor - mean) / (std + eps)
        return g * x_hat + b
    
    
    # 4. Run grad check on all three tensors (x, gamma, beta)
    tests_passed = torch.autograd.gradcheck(
        functional_layer_norm,
        inputs=(x, gamma, beta),
        eps=1e-5,
        atol=1e-4,
        rtol=1e-4,
        raise_exception=True
    )
    
    assert tests_passed, "LayerNorm weight or input gradient failed!"
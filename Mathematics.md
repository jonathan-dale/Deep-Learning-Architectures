# Mathemaics Behind LayerNorm and RMSNorm
#### An explanation of the mathematics of normalization in deep learning.

While normalization does not guarantee that activations, weights, or gradients will always remain well-behaved throughout training, it rescales intermediate representations to give their feature values a more controlled scale and improved conditioning, which ultimately enhances numerical behavior and optimization stability.

> Normalization of the variables computed during the forwrd pass through a neural network removes the need for the network to deal with extremely large or extremely small values.

Bishop, Christopher M., and Hugh Bishop. [Deep Learning: Foundations and Concepts. Springer, 2023](https://link.springer.com/book/10.1007/978-3-031-45468-4)



## LayerNormalization Mathematics

#### An example to start:

For a tensor `x` with shape `(batch, tokens, hidden)` and a layer configured as `nn.LayerNorm(hidden)`, LayerNorm computes statistics independently for each token vector acros its hidden-feature dimension.

For example if:

```python
x.shape == (8, 512, 768)
#            ↑    ↑    ↑
#          batch tokens features
```
You have 8 sequences in the batch, each sequence has 512 tokens, and each token is represented by 768 numbers which make up its hidden-state vector.
Then LayerNorm will compute one mean and one variance for each of the `8 x 512 = 4,096` token vectors. Each statistic is computed from the 768 hidden-feature values of that token only.

Every one of these 4,096 tokens has a separate vector of 768 feature values and LayerNorm performs the same operation 4,096 separate times:

```text
Token 1:  normalize its 768 values
Token 2:  normalize its 768 values
Token 3:  normalize its 768 values
...
Token 4,096: normalize its 768 values
```

With this, LayerNorm computes:
- 8 × 512 = 4,096 separate means,
- 4,096 separate variances,
- each based on the 768 hidden features of one token.

### What happens for one token
Pick one token—for example, token 10 in sequence 3:

```python
x[3, 10].shape  # (768,)
```

It might look conceptually like:
```python
[0.3, -1.1, 2.0, 0.7, ..., 768 total values]
```

LayerNorm looks only at those 768 values and then calculates:
- Their average, or the mean, $\mu$.
- How spread out they are, or the standard deviation, $\sigma$.
- It rescales them so they are on a more consistent numeric scale.   

During this calculation, it does not look at:
- Other tokens in the sequence.
- Tokens in other batch examples.
- Values from other sequences.
So token 10 gets normalized independently from token 11, even though they are adjacent words.

Although LayerNorm calculates separate statistics for every token vector, the same learned $\gamma$ and $\beta$ vectors are applied to every token position and every batch example. Their gradients are therefore summed over all batch and sequence positions, however the learned affine parameters are typically shared across every token and batch item:

```python
gamma.shape == (768,)
beta.shape == (768,)
```

### A tiny example
Imagine this smaller tensor:

```python
x.shape == (2, 3, 4)
#            2 sequences
#            3 tokens each
#            4 hidden features per token
```

The first sequence might contain three token vectors:
```python
sequence_0 = [
    [2.0, 4.0, 6.0, 8.0],     # token 0
    [1.0, 1.5, 2.0, 2.5],     # token 1
    [-3.0, 0.0, 3.0, 6.0],    # token 2
]
```

LayerNorm processes them like this:
```text
[2.0, 4.0, 6.0, 8.0]      → calculate statistics from these 4 numbers only
[1.0, 1.5, 2.0, 2.5]      → calculate statistics from these 4 numbers only
[-3.0, 0.0, 3.0, 6.0]     → calculate statistics from these 4 numbers only
```

For the first token:
```python
token = torch.tensor([2.0, 4.0, 6.0, 8.0])
```

Its mean is:
$$(2+4+6+8)/4=5$$

LayerNorm subtracts 5 from every number, then divides by a measure of their spread:
```text
Original:       [ 2,  4,  6,  8]
Minus mean:     [-3, -1,  1,  3]
Rescaled:       roughly [-1.34, -0.45, 0.45, 1.34]
```
> Notice how the final values are centered around zero.

LayerNorm calculates a separate mean for each row, subtracts that row’s mean, and then divides by that row’s spread. LayerNorm computes its statistics across the hidden units of a single input representation—not across all tokens together

## Token-by-Token Results 
| Token   | Original values  | Mean | After subtracting its own mean |
| ------- | ---------------- | ---- | ------------------------------ |
| Token 0 | [2, 4, 6, 8]     | 5.0  | [-3, -1, 1, 3]                 |
| Token 1 | [1, 1.5, 2, 2.5] | 1.75 | [-0.75, -0.25, 0.25, 0.75]     |
| Token 2 | [-3, 0, 3, 6]    | 1.5  | [-4.5, -1.5, 1.5, 4.5]         |


After subtracting the mean, every normalized token vector has mean zero in exact arithmetic.

```text
Token 0: (-3 + -1 + 1 + 3) / 4 = 0
Token 1: (-0.75 + -0.25 + 0.25 + 0.75) / 4 = 0
Token 2: (-4.5 + -1.5 + 1.5 + 4.5) / 4 = 0
```

Then LayerNorm divides each row by its own standard deviation plus a tiny value of eps, ${\sqrt{\sigma + \epsilon}}$. Ignoring $\epsilon$ the normalized values have unit variance; with $\epsilon$ included, their variance is slightly below one, which is usually negligible when the input variance is much larger than $\epsilon$. This small value, $\epsilon$, is included in the denominator for numerical stability; it prevent division by zero when the variance is close to zero.

Because $\hat{x}$ is centered around zero:

```python
x_hat = (x - mean) / torch.sqrt(var + self.eps)
```

So for each token vector:

```python
x_hat[b, t].mean()
```

will be extremely close to 0.0.

But the final output from your full LayerNorm layer is:

```python
return self.gamma * x_hat + self.beta
```

After multiplying by gamma and adding beta, the output is not guaranteed to still average to zero.

At initialization, the parameters are:

```python
gamma = 1
beta = 0
```

so initially:

```python
output == x_hat
```

and every token vector is centered near zero.

During training, however, the model learns individual values for gamma and beta:

```python
gamma = [0.8, 1.2, 0.5, 1.7, ...]
beta  = [0.1, -0.3, 0.4, 0.0, ...]
```

Those learned scale and shift parameters let the model choose a different preferred distribution for each hidden feature. Therefore
```python
x_hat.mean(dim=-1)  # approximately 0 for each token
output.mean(dim=-1) # not necessarily 0 after gamma and beta
```







---

#### One nuance
Normalization over the hidden size is the standard Transformer convention, however it is not a requirement of every neural-network architecture. For example, computer-vision architectures may normalize over a specific channel and/or spatial dimensions depending on the model.


---

# Backward Pass Mathematics
## LayerNorm
To implement or verify a custom backward pass, we use the multivariate chain rule to compute the gradients of a scalar loss $L$ with respect to our inputs ($x$) and learnable parameters ($\gamma$, $\beta$).

For a single feature vector $x$ of dimension $D$ (where $D = \text{HIDDEN}$), the forward equations are:
1. **Mean:** $\mu = \frac{1}{D} \sum_{i=1}^D x_i$
2. **Variance:** $\sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$
3. **Normalized Input:** $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4. **Output:** $y_i = \gamma_i \hat{x}_i + \beta_i$

During the backward pass, we receive the upstream gradient from the layer above: $\frac{\partial L}{\partial y_i}$.

### 1. Gradients for Learnable Parameters ($\gamma$ and $\beta$)
Because $\gamma$ and $\beta$ only impact the final linear transformation, their derivatives are straightforward:

* **Bias Gradient ($\beta$):**
  $$\frac{\partial L}{\partial \beta_i} = \frac{\partial L}{\partial y_i}$$

* **Scale Gradient ($\gamma$):**
  $$\frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

*(Note: In practice, these gradients are summed across the batch and sequence dimensions).*

### 2. Intermediate Gradient ($\hat{x}$)
Before computing the final input gradient, we calculate how the loss changes with respect to the normalized feature space:
$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma_i$$

### 3. Gradient for the Input ($x$)
Because an individual input element $x_i$ affects the output directly, but *also* affects it indirectly by altering the calculated $\mu$ and $\sigma^2$, the chain rule must combine all three pathways. Combining and simplifying these pathways yields the final optimized input gradient formula:

$$\frac{\partial L}{\partial x_i} = \frac{1}{D \cdot \sqrt{\sigma^2 + \epsilon}} \left[ D \cdot \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^D \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \cdot \sum_{j=1}^D \left( \frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j \right) \right]$$

#### Component Breakdown:
* **Term 1:** $D \cdot \frac{\partial L}{\partial \hat{x}_i}$ is the direct gradient passing backward through the scale/shift normalization step.
* **Term 2:** $\sum_{j=1}^D \frac{\partial L}{\partial \hat{x}_j}$ is the mathematical correction factor required because we subtracted the mean $\mu$.
* **Term 3:** $\hat{x}_i \cdot \sum_{j=1}^D \left( \frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j \right)$ is the mathematical correction factor required because we divided by the variance $\sigma^2$.

A slightly more implementation friendly equivalent is:

$$\frac{\partial L}{\partial x} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left[ g - mean(g) - \hat{x} \cdot men(g \cdot \hat{x}) \right]$$

where:

$$g = {\partial L}/{\partial \hat{x}}$$

## RMSNorm
RMSNorm rescales a feature vector using its root-mean-square magnitude, rather than first subtracting its mean as LayerNorm does. The common Transformer form includes a learned per-feature scale parameter $\gamma$ and no learned bias parameter $\beta$.

Because RMSNorm does not depend on the input mean, its backward pass does not need LayerNorm's mean-related correction term.

Reference:  
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
[https://github.com/bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm)


Because RMSNorm bypasses mean subtraction and bias tracking, its backward pass requires fewer operations than LayerNorm. We use the chain rule to compute gradients of a scalar loss $L$ with respect to our inputs ($x$) and learnable scale parameters ($\gamma$).

For a single feature vector $x$ of dimension $D$ (where $D = \text{HIDDEN}$):
1. **Mean Square:** $\text{MS} = \frac{1}{D} \sum_{i=1}^D x_i^2$
2. **Root Mean Square:** $\text{RMS} = \sqrt{\text{MS} + \epsilon}$
3. **Normalized Input:** $\hat{x}_i = \frac{x_i}{\text{RMS}}$
4. **Output:** $y_i = \gamma_i \hat{x}_i$

### 1. Gradient for Learnable Parameter ($\gamma$)
In the common bias-free RMSNorm variant, their is no leanred $\beta$ parameter, so only the scale parameter $\gamma$ has a normalization-layer gradient. An implementation could optionally include an additional bias, but that is not part of the usual RMSNorm formulation,

$$\frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

### 2. Intermediate Gradient ($\hat{x}$)
$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma_i$$

### 3. Gradient for the Input ($x$)
Applying the chain rule through the $\text{RMS}$ denominator yields a highly streamlined input gradient formula:

$$\frac{\partial L}{\partial x_i} = \frac{1}{\text{RMS}} \left[ \frac{\partial L}{\partial \hat{x}_i} - \frac{\hat{x}_i}{D} \sum_{j=1}^D \left( \frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j \right) \right]$$

Equivalently, in vector notation:

$$\frac{\partial L}{\partial x} = \frac{1}{\text{RMS}} \left[g - \hat{x} \cdot mean(g\cdot \hat{x}) \right]$$

where:
$$g = {\partial L}/{\partial \hat{x}}$$

#### Component Breakdown:
* **Term 1:** $\frac{\partial L}{\partial \hat{x}_i}$ is the direct gradient passing back through the division step.
* **Term 2:** $\frac{\hat{x}_i}{D} \sum_{j=1}^D \left( \frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j \right)$ is the correction factor for the scale denominator.



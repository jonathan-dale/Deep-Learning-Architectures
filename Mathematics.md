# Mathemaics behind Layer Norm and RMSNorm
Explaination of the math behind normalization in deep learning.

## Layer Normalization Backward Pass Mathematics

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



## RMSNorm Backward Pass Mathematics
>- Notice how much cleaner this is compared to LayerNorm because mean and bias are gone.

Because RMSNorm bypasses mean subtraction and bias tracking, its backward pass requires fewer operations than LayerNorm. We use the chain rule to compute gradients of a scalar loss $L$ with respect to our inputs ($x$) and learnable scale parameters ($\gamma$).

For a single feature vector $x$ of dimension $D$ (where $D = \text{HIDDEN}$):
1. **Mean Square:** $\text{MS} = \frac{1}{D} \sum_{i=1}^D x_i^2$
2. **Root Mean Square:** $\text{RMS} = \sqrt{\text{MS} + \epsilon}$
3. **Normalized Input:** $\hat{x}_i = \frac{x_i}{\text{RMS}}$
4. **Output:** $y_i = \gamma_i \hat{x}_i$

### 1. Gradient for Learnable Parameter ($\gamma$)
Since $\beta$ does not exist in RMSNorm, we only need the derivative for $\gamma$:
$$\frac{\partial L}{\partial \gamma_i} = \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

### 2. Intermediate Gradient ($\hat{x}$)
$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma_i$$

### 3. Gradient for the Input ($x$)
Applying the chain rule through the $\text{RMS}$ denominator yields a highly streamlined input gradient formula:

$$\frac{\partial L}{\partial x_i} = \frac{1}{\text{RMS}} \left[ \frac{\partial L}{\partial \hat{x}_i} - \frac{\hat{x}_i}{D} \sum_{j=1}^D \left( \frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j \right) \right]$$

#### Component Breakdown:
* **Term 1:** $\frac{\partial L}{\partial \hat{x}_i}$ is the direct gradient passing back through the division step.
* **Term 2:** $\frac{\hat{x}_i}{D} \sum_{j=1}^D \left( \frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j \right)$ is the correction factor for the scale denominator.



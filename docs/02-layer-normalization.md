## Lesson 2: Layer Normalization

Gradient descent updates parameters using gradients. Layer normalization changes the values flowing through a netowrk so those gradients see a more predictable scale. It is an architectural operation, not an optimizer. 

The key question is: **which values do we normalize together?**

## 1. The formula

Suppose one token or example has a feature vector

$$
x=(x_1,\ldots,x_d)
$$

Layer normalization computes the mean and variance across its features:

$$
\mu=\frac{1}{d}\sum_{i=1}^d x_i,
\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^d(x_i-\mu)^2.
$$


It standardizes the vector:

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

Finally, it learns a scale and shift:

$$
y_i=\gamma_i\hat{x}_i+\beta_i.
$$

Where:

- $\epsilon$ prevents division by zero;
- $\gamma$ starts at 1 and allows the model to restore useful scale;
- $\beta$ starts at 0 and allows the model to restore a useful offset.

Normalization dose not force the final output to have mean zero and standard deviation one, because learned $\gamma$ and $\beta$ can change them. It normalizes the intermediate signal before the learned affine transformation.

## 2. What "layer" means

For an input shaped `(batch, sequence, features)`, layer normalization usually normalizes the **last dimension**, independently for every batch item and every sequence position.

If the input is 

```text
(batch=2, sequence=3, features=4)
```

then, there are six separate four-dimensional vectors. For each one, we calculate its own mean and variance.

To do this, we use `dim=-1`, we are only caclulating these values over the last dimemsion.

```python
mean = x.mean(dim=-1, keepdim=True)
variance = x.var(dim=-1, keepdim=True, correction=0)
normalized = (x = mean) / torch.sqrt(variance + eps)
return gamma * normalized + beta
```

The `keepdim=True` matters because it preserves a shape that broadcasts cleanly back over the feature dimension. 

## 3. Why normalization helps optimization

Consider a deep chain of transformations. If one layer produces activations that are extremely large, the next layer may receive large gradients or enter an inconvenient nonlinear regime. If activations are extremely small, the signal is hard to detect since deratives becomes too small (when they are at or near zero).

Layer normalization centers and rescales each input across the dimensions being normalized. It uses statistics from each input rather than from the batch, so its behavior does not depend on batch size, and it performs the same computation during training and inference. This can make activations less sensitive to their overall scale and help stabilize training, including in sequence models, but it does not by itself handle padding or prevent gradient problems. Learnable gamma and beta parameters let the model adjust the scale and offset after normalization.


The important connection to gradient descent is geometric. The loss landscape is expressed in terms of parameters, but the landscape also depends on the internal coordinates used by the network. Normalization can make those coordinates better conditioned, so a single learning rate works more reasonably across directions.

It does not guarantee faster training in every setting, and it does not remove the need for a good optimizer or learning rate schedule. 

## 4. Layer norm vs. batch norm

**Batch normalization** estimates statistics across a batch, commonly for convolutional feature maps. Its behavior depends on the other examples in the batch and it needs special behavior when evaluating.

**Layer normalization** estimates statistics within each individual example across its feature dimension. It wokrs naturally with small batches and does not need running batch statistics.

This difference makes layer normalization especially convenient for recurrent networks and Transformer-style sequence models, where sequence lengths and batch size may vary and where mixing statistics across examples is often undesirable.

Neither is universally better. The normalization axis must match the architecture and the dependency structure we want to preserve.

## 5. Reading the repository implementation

This projects `LayerNorm` has three conceptual pieces:

```python
self.beta = nn.Parameter(torch.zeros(dim))
self.gamma = nn.Parameter(torch.ones(dim))

mean = torch.mean(x, dim=-1, keepdim=True)
variance = torch.var(x, dim=-1, keepdim=True, correction=0)
normalized = (x - mean) / torch.sqrt(variance + self.eps)
return self.gamma * normalized + self.beta
```

`nn.Parameter` tells PyTorch that `gamma` and `beta` are trainable. During backpropagation, the loss produces gradients for them, and the optimizer updates them just like any other model parameter. 

The tests check shape preservation, normalization over the last dimemsion, different input ranks, constant imputs, conparison with PyTorch's reference implementation, and gradients for `gamma` and `beta`.

## 6. A small numerical example

Take $x=(1,2,3)$. Its mean is 2 and its population variance is $2/3$. By ignoring $\epsilon$ for a hand calculation:

$$
\hat{x}=
\left(
\frac{-1}{\sqrt{2/3}},
0,
\frac{1}{\sqrt{2/3}}
\right).
$$


The result is centered around zero and with a population variance of one. If $\gamma=(2,2,2)$ and $\beta=(1,1,1)$ then the final result is twice the normalized vector plus one. The model did not lose the ability to represent other scales; it learns them explicitly.

## 7. Where it appears in Transformer blocks

The common pattern is:

$$
\text{Post-LN:}\quad
\mathrm{LN}\left(x+\mathrm{Sublayer}(x)\right)
$$

and 

$$
\text{Pre-LN:}\quad
x+\mathrm{Sublayer}(\mathrm{LN}(x)).
$$


The notation varies across publications so its a good idea to inspect the exact computation order. In modern Transformer design, pre-normalization is popular because it often gives more stable gradient flow through deep residual stacks. However, this is a design choice with tradeoffs, not a claim that one formula wins everywhere.


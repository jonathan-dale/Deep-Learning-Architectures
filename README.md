# Deep Learning Architectures (dla)

## Project details

This project implements raw pytorch to display understanding of what happens under the hood for the following:

- A custom Transformer block: a multi-head attention, layer normalization, and fee-forward blocks from scratch using raw tensors. One encoder block, not a 12-layer GPT clone.

- Custom Back propagation/loss: custom loss function (like focal loss or sppecialized contrastive loss) and its gradient steps to showcase my statistics background.

- The raw training loop: Explicitly write out `loss.baclward()`, `optimizer.step()`, and `optimizer.zero_grad()` functions. Including gradient clipping to show that I know how to handle unstable training. 

## Goals
1. Implementations, not just a high level wrapper like hugging face trainer. 
2. Transformer block diagram + Pre-LN.
3. Attention and LayerNorm formulas.
4. Focal loss + gradient check command.
5. Training loop snippet (the three calls + clip).
6. How to run tests and the example.
7. "Compared to torch.nn / Hugging Face" table. 


### Environment setup

A simple test to ensure that the environment is set up for this project:
```python
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
```


## Three pillars (implementation):

### 1. The custom Transformer block:
>- One encoder block, not a 12-layer GPT clone.  

##### Attention (raw tensors)
- Linear maps: Q, K, V =
- Scaled socres: 
- Causal or padding mask (add -inf before softmax)
- Softmax -> attend to V
- Concat head -> W0
- Explicit shapes in comments: (B, T, C) -> (B, H, T, d_k)

##### LayerNorm
- Mean/variance over the last dimemtion, epsilon, learnable gamma, beta
- Implement Pre-LN (Norm -> sublayer -> residual). more stability in training and provides a reason to talk about exploding / vanishing gradients.  
Reference: Andrej Karpathy [Yes, you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)

##### FFN (Feed forward network)
- Linear -> GELU/ReLU -> Linear, inner dim `4 * d_model`
- Residual + dropout after attention and after FFN

##### Block
```sh
x = x + Dropout(MHA(LN(x)))
x = x + Dropou(FFN(LN(x)))
```

>- Later plan is to include a tiny stack of 2-4 blocks + token/position embeddings so the example actually learns

### 2. Custom loss + gradient
- Focal loss (classification):
FL = -alpha(1-pt)^gamma * log(pt)



TODO: doccument this in the readme:
```note
1. write out the formula (with pt defined).
2. Implement `forward` in log-space (for stability).
3. Derive partial FL wrt. to z (logits) on paper and implement in the code.
4. Gradient check: `torch.autograd.grad` vs my closed form.
```


### 3. Raw training loop
One function, no trainer class theater:
```python
for epoch:
    for batch:
        optimiser.zero_grad()
        logits = model(x)
        loss = critertion(logits, y)
        loss.backward()
        clip_grad_norm_(model.prameters(), max_norm=1.0)
        optimizer.step()
```

Add explicit:
- `model.train()` / `model.eval()` + `torch.no_grad()`
- log `loss`, `grad_norm` before clip to watch the instability
- cosine LR warmup 
- Keep AdamW: Transformer logits/attention can spike and clip bounds the update.







### The mathematics behind it all is covered in the Mathematics.md
[Mathematics.md](./Mathematics.md)

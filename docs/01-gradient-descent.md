## Lesson 1: Gradient Descent

Gradient descent is the basic learning mechanism behind nural networks and deep learning. A network starts with randomly choosen parameters that follow specific mathematical rules however, these parameters do not initially make good predictions. The loss function measures these parameters and then adjusts them to reduce the loss. 

Think of the loss function as a landscape (simple 3D model), the parameters are coordinates on the landscape, and gradient points uphill. This simple model makes visualizing the loss easy and relatable to our lived experinece, however most deep learning models use hundreds of dimensions making it imposible to visulalize with images or drawings that reference this concept. The math works out in this high dimemsional space the same as in our physical 3D world even though we can not physically see it.

## 1. The optimization problem

Let the model have parameters

$$
\theta = [\theta_1, \theta_2 \ldots, \theta_n]
$$

and let its loss be $L(\theta)$. Training the model means finding parameters with the smallest loss: 

$$
\theta^* = \mathrm{argmin}_{\theta} L(\theta).
$$

For only a single parameter, the derivative $dL/d\theta$ tells us the local slope:
 - positive slope: means increasing $\theta$ increases the loss, so in this case we decrease $\theta$;
 - negative slope: means increasing $\theta$ decreases the loss, so in this case we increase $\theta$;
 - zero slope: this means that we are at a flat point, however it may only be a local min and not the true bottom.

For a tensor with many parameters, the derivatives form a gradient:

$$
\nabla_\theta L = 
\begin{bmatrix}
\frac{\partial L}{\partial \theta_1}\\
\vdots\\
\frac{\partial L}{\partial \theta_n}
\end{bmatrix}.
$$

The update is

$$
\boxed{\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)}
$$

where $\eta$ is the **learnign rate**. This is not a mysterious neural network rule; it is a repeated first-order approximation of optimization.

### Why the negative gradient?

For a small displacement $\Delta\theta$, Taylor expansion gives

$$
L(\theta+\Delta\theta)
\approx L(\theta) + \nabla L(\theta)^T\Delta\theta.
$$

If we choose $\Delta\theta=-\eta\nabla L$, then

$$
L(\theta+\Delta\theta)
\approx L(\theta)-\eta\|\nabla L\|^2,
$$

This decreases the loss whenever the step is sufficiently samll and the gradient is nonzero. The gradient is the direction of steepest increase under the usual Euclidean geometry, so its negative is the direction of steepst local decrease.

## 2. A first example: a bowl-shaped loss

Consider

$$
L(w) = (w-3)^2.
$$


Its derivative is $dL/dw=2(w-3)$, and the minimum is at $w=3$. Starting from $w=0$, gradient descent repeatedly moves toward 3:

```python
import torch

w = torch.tensor(0.0, requires_grad=True)

for step in range(20):
    loss = (w - 3).square()
    loss.backward()

    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_()

    print(step, w.item(), loss.item())
```

After just 20 steps, the weight tensor `w` is aproaching the target value of 3.0. By step 49 it reaches the value `2.999957`

```text
0 0.6000000238418579 9.0
1 1.0800000429153442 5.760000228881836
2 1.4639999866485596 3.6863999366760254
3 1.7711999416351318 2.3592960834503174
4 2.0169599056243896 1.5099495649337769
5 2.2135679721832275 0.9663678407669067
6 2.370854377746582 0.6184753179550171
7 2.4966835975646973 0.39582422375679016
8 2.597346782684326 0.2533273994922638
9 2.677877426147461 0.16212961077690125
10 2.7423019409179688 0.10376295447349548
11 2.793841600418091 0.06640829145908356
12 2.835073232650757 0.042501285672187805
13 2.868058681488037 0.027200838550925255
14 2.894446849822998 0.01740851067006588
15 2.915557384490967 0.011141467839479446
16 2.932446002960205 0.007130555342882872
17 2.9459567070007324 0.004563542548567057
18 2.9567654132843018 0.0029206774197518826
19 2.9654123783111572 0.0018692294834181666
```

Their are four important lines in the code above:

1. construct the loss;
2. call `backward()` to calculate the derivative;
3. subtract the product of the learning rate and gradient;
4. clear the gradietn at the end of each iteration, `w.grad.zero_()`.

PyTorch accumulates gradients in `.grad`, so forgetting to clear the gradient (line 4) causes old gradients to be added to the newly calculated iteration.

The `gradient_descent` function makes this loop reusable:

```python
from dla.gradient_descent import gradient_descent

parameters, losses = gradient_descent(
    lambda x: (x-3).square().sum(),
    torch.tensor([0.0]),
    learning_rate = 0.1,
    steps = 100,
)
```

## 3. Learning rate: the most important knob

The learning rate controls how large of a step per update.

- Too small: training is stable but painfully slow.
- Reasonable: the loss decreases efficiently.
- Too large: the algorithm overshoots, oscillates, or diverges.

For the quadratic $L(w)=(w-3)^2$, the update is 

$$
w_{t+1}=w_t-\eta\,2(w_t-3).
$$

Writing the error as $e_t=w_t-3$ gives

$$
e_{t+1}=(1-2\eta)e_t.
$$

So this problem converges when $0<\eta<1$. At $\eta=0.5$, the error becomes zero in one step; near $\eta=1$, it alternates with slowly shrinking magnitide; above 1, it grwos. Real nural-network losses are not single perfect bowls, so this analysis shows why learning rates are so important. 


## 4. Fron full-batch GD to SGD

For a dataset with examples $(x_i,y_i)$, the training objective is often an average:

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^N \ell(f_\theta(x_i),y_i).
$$


**Batch gradient descent** computes the gradient using all $N$ examples. This is accurate but expensive for a large dataset. 

**Stochastic gradient descent (SGD)** estimates the gradient with one example or a mini-batch $B$:

$$
\theta_{t+1}
=\theta_t-\eta\frac{1}{|B|}\sum_{i\in B}
\nabla_\theta\ell_i(\theta_t).
$$

The mini-batch estimate is noisy, and the noise comes at a cost. However this can help the optimizer move through narrow regions and escape some saddle-like behavior. In practice mini-batch training is the standard compromise between computational efficiency and gradient quality.

## 5. Momentum and adaptive methods

Plain SGD treats every update independently. Momentum keeps a moving direction:

$$
v_{t+1}=\beta v_t+\nabla L(\theta_t),\qquad
\theta_{t+1}=\theta_t-\eta v_{t+1}.
$$

It smooths noisy gradients and can accelerate movement along directions where the gradient is consistently aligned.

Adam combines momentum-like first moments with a moving estimate of squared gradients:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_{t+1}
=\theta_t-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

The denominator scales updates by recent gradient magnitude. Adam is often a strong default, however it still requires the correct learning rate and schedul choices, and SGD can sometimes genralize better after careful tuning. 

## 6. A short history and the current direction

- **1847 - Cauchy:** published an early gradient method for solving systems of equations. The main idea was to "decrease a function by moving its gradient". 
- **1951 Robbins and Monro:** established stochastic approximation, a mathematical ancestor of modern SGD for learning from noisy observations.
- **1980s-1990s - back propogation becomes practical:** efficient chain-rule derivatives made multilayer neural-network training feasible.
- **2010s - deep learning  scales:** GPU's, large data sets, ReLU networks, momentum, RMSProp, and Adam made gradient-based training effective at depth. 
- **Today:** the basic update remains, but research focuses on geometry, normalization, memory use, optimizer state, scaling with model size, and better learning-rate schedules. Recent work explores stateless and norm-aware optimizers for large language models; these are refinements of the same central loop, not replacements for differentation.

The lesson is that optimization and architecture are coupled. A network can represent the right function and still be difficult to train. Good parameterization, initialization, normalization, batch size, schedules, and optimizer choice shapes the geometry seen by graident descent.



## 7. What gradient descent can and cannot promise

For a convex bowl, gradient descent with suitable leanring rate can reach the global minimum. Neural-network losses are generally non-convex, so guarantees are weaker:

- a zero gradient may be a local minimum, maximum, or a saddle point;
- different parameter settings can represent similar functions;
- curvature can differ dramatically by direction;
- trainging loss and held-out generalization are different objectives.

High-dimensional neural netowrks often have many useful low-loss solutions, and first-order methods work remarkably well in practice. Understanding the algorighm's assumptions helps us diagnose failures instead of treating training as magic.

### References
- A.-L. Cauchy, *Méthode générale pour la résolution des systèmes d'équations simultanées*, 1847.
    - [A free English translation](https://www.probabilityandfinance.com/pulskamp/Cauchy/Orbits/1847%20CR%20536(383).pdf)
- H. Robbins and S. Monro, *A Stochastic Approximation Method*, 1951.
    - [A Stochastic Approximation Method](https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2Faoms%2F1177729586)
- Bishop, Christopher M., and Hugh Bishop. *Deep Learning: Foundations and Concepts*, 2023.
    - [Deep Learning: Foundations and Concepts.](https://link.springer.com/book/10.1007/978-3-031-45468-4)
- Iam Goodfellwo, Yoshua Bengio, and Aaron Courville, *Deep Learning*, 2016.
    - [deeplearningbook.org](https://www.deeplearningbook.org/)
"""
deep_learning_from_scratch.py
==============================
Every component built with only NumPy.
Each section explains the math, then implements it.
Run the file directly to execute all experiments at the end.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(42)


# ==============================================================================
# 1. WEIGHT INITIALIZATION
# ==============================================================================
# Why it matters: if weights start too large, activations explode through layers.
# Too small and gradients vanish before reaching early layers.
# Both Xavier and He tune the variance based on layer fan-in / fan-out.

def init_random(fan_in, fan_out):
    """Naive random init — baseline to show what goes wrong."""
    return np.random.randn(fan_in, fan_out) * 1.0

def init_xavier(fan_in, fan_out):
    """
    Xavier / Glorot uniform initialization.
    Designed for Sigmoid / Tanh where the derivative near 0 is ~1.
    Keeps variance of activations roughly equal across layers.

    Derivation: Var(W) = 2 / (fan_in + fan_out)
    Uniform draw from [-limit, +limit] has variance = limit^2 / 3
    Setting limit^2/3 = 2/(fan_in+fan_out) → limit = sqrt(6/(fan_in+fan_out))
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def init_he(fan_in, fan_out):
    """
    He / Kaiming normal initialization.
    Designed for ReLU: because ReLU zeros half its inputs, effective variance
    is halved, so we compensate by doubling the variance → Var(W) = 2/fan_in.

    Derivation: std = sqrt(2 / fan_in)
    """
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std


# ==============================================================================
# 2. ACTIVATION FUNCTIONS
# ==============================================================================
# Each activation must implement forward(x) and backward(grad_out, cache).
# The cache stores whatever the backward pass needs (usually the input x,
# or the output for activations whose derivative is expressed in terms of output).

class ReLU:
    """
    ReLU(x) = max(0, x)
    Derivative: 1 if x > 0, else 0.
    Problem: neurons where x <= 0 always output 0 and receive no gradient
    ('dead neurons'). He init helps prevent this at the start.
    """
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad_out):
        return grad_out * (self.cache > 0)


class LeakyReLU:
    """
    LeakyReLU(x) = x if x > 0, else alpha * x
    Fixes dead neurons by allowing a small gradient on the negative side.
    Derivative: 1 if x > 0, else alpha.
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.cache = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_out):
        dx = np.where(self.cache > 0, 1.0, self.alpha)
        return grad_out * dx


class ELU:
    """
    ELU(x) = x           if x > 0
           = alpha*(e^x - 1)  if x <= 0
    Smoother than LeakyReLU. Negative saturation helps push mean activations
    toward zero (like batch norm but built-in).
    Derivative:
        1               if x > 0
        ELU(x) + alpha  if x <= 0   ← note we reuse the forward output
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        self.out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        self.cache = x
        return self.out

    def backward(self, grad_out):
        dx = np.where(self.cache > 0, 1.0, self.out + self.alpha)
        return grad_out * dx


class GELU:
    """
    GELU(x) = x * Phi(x)   where Phi is the standard normal CDF.
    Used in Transformers (BERT, GPT). Smooth, non-monotonic.
    Tanh approximation (fast):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Exact derivative is messy; we use the tanh approximation's derivative.
    """
    def forward(self, x):
        self.cache = x
        c = np.sqrt(2.0 / np.pi)
        self.tanh_arg = c * (x + 0.044715 * x**3)
        self.tanh_val = np.tanh(self.tanh_arg)
        return 0.5 * x * (1 + self.tanh_val)

    def backward(self, grad_out):
        x = self.cache
        c = np.sqrt(2.0 / np.pi)
        sech2 = 1 - self.tanh_val**2          # derivative of tanh
        dtanh_arg = c * (1 + 3 * 0.044715 * x**2)
        dgelu = 0.5 * (1 + self.tanh_val) + 0.5 * x * sech2 * dtanh_arg
        return grad_out * dgelu


class Swish:
    """
    Swish(x) = x * sigmoid(x)
    Self-gated activation. Smooth, non-monotonic like GELU but simpler.
    Derivative: Swish(x) + sigmoid(x) * (1 - Swish(x))
              = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
    """
    def forward(self, x):
        self.sig = 1 / (1 + np.exp(-x))
        self.out = x * self.sig
        self.cache = x
        return self.out

    def backward(self, grad_out):
        dx = self.out + self.sig * (1 - self.out)
        return grad_out * dx


class Sigmoid:
    """
    Sigmoid(x) = 1 / (1 + e^-x)
    Squashes output to (0, 1). Classic binary output activation.
    Derivative: sigma * (1 - sigma)  ← reuse forward output, no re-computation.
    Problem: saturates at extremes → gradients ~0 → vanishing gradient.
    """
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad_out):
        return grad_out * self.out * (1 - self.out)


class Tanh:
    """
    Tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    Zero-centered (unlike Sigmoid), output in (-1, 1).
    Derivative: 1 - tanh^2(x)  ← reuse forward output.
    Also saturates, so vanishing gradient still an issue in deep nets.
    """
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out):
        return grad_out * (1 - self.out**2)


# ==============================================================================
# 3. DENSE (FULLY CONNECTED) LAYER
# ==============================================================================
# A Dense layer computes: out = X @ W + b
# Shapes: X=(batch, in), W=(in, out), b=(out,), out=(batch, out)
#
# Backward pass derives from chain rule:
#   dL/dX = dL/dout @ W.T          (gradient to pass to previous layer)
#   dL/dW = X.T @ dL/dout          (gradient for W update)
#   dL/db = sum over batch of dL/dout

class Dense:
    def __init__(self, in_features, out_features, init='he'):
        if init == 'xavier':
            self.W = init_xavier(in_features, out_features)
        elif init == 'random':
            self.W = init_random(in_features, out_features)
        else:
            self.W = init_he(in_features, out_features)
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        x = self.cache
        self.dW = x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


# ==============================================================================
# 4. DROPOUT LAYER
# ==============================================================================
# Dropout randomly zeros activations during training (probability p of keeping).
# Inverted dropout: scale by 1/p during training so that at inference we can
# simply pass the input through unchanged — no rescaling needed at test time.
#
# Backward: gradient only flows through the neurons that were kept.

class Dropout:
    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self.training = True

    def forward(self, x):
        if not self.training:
            return x
        self.mask = (np.random.rand(*x.shape) < self.keep_prob) / self.keep_prob
        return x * self.mask

    def backward(self, grad_out):
        if not self.training:
            return grad_out
        return grad_out * self.mask


# ==============================================================================
# 5. BATCH NORMALIZATION
# ==============================================================================
# Problem: internal covariate shift — distribution of layer inputs changes
# during training, slowing convergence. BatchNorm fixes each mini-batch to
# have zero mean and unit variance, then applies learnable scale (gamma)
# and shift (beta).
#
# Train mode: normalize using batch statistics.
# Eval mode:  normalize using running statistics accumulated during training.
#
# Backward through BatchNorm is non-trivial because the mean and variance
# are functions of x too. Full derivation shown in comments.

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)   # learnable scale
        self.beta  = np.zeros(num_features)  # learnable shift
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)
        self.training = True

    def forward(self, x):
        if self.training:
            mu  = x.mean(axis=0)
            var = x.var(axis=0)
            # Update running stats (used at eval time)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
            x_hat = (x - mu) / np.sqrt(var + self.eps)
            self.cache = (x, x_hat, mu, var)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta

    def backward(self, grad_out):
        x, x_hat, mu, var = self.cache
        N = x.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        self.dgamma = (grad_out * x_hat).sum(axis=0)
        self.dbeta  = grad_out.sum(axis=0)

        # Gradient w.r.t. x_hat
        dx_hat = grad_out * self.gamma

        # Gradient w.r.t. variance
        dvar = (-0.5 * dx_hat * (x - mu) * std_inv**3).sum(axis=0)

        # Gradient w.r.t. mean
        dmu = (-dx_hat * std_inv).sum(axis=0) + dvar * (-2 * (x - mu)).mean(axis=0)

        # Final gradient w.r.t. input x
        dx = dx_hat * std_inv + dvar * 2 * (x - mu) / N + dmu / N
        return dx

    def params(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


# ==============================================================================
# 6. LOSS FUNCTIONS
# ==============================================================================

class CrossEntropyLoss:
    """
    For multi-class classification. Expects raw logits (before softmax).
    Combines softmax + negative log-likelihood in one numerically stable step.

    Softmax: p_k = exp(z_k) / sum_j(exp(z_j))
    Loss:    L = -sum_k [ y_k * log(p_k) ]  averaged over batch.

    Numerical stability: subtract max(z) before exp → same result, no overflow.

    Gradient w.r.t. logits (combined softmax + CE):
        dL/dz_k = p_k - y_k   (for one-hot y)
    This is the clean result that makes backprop simple.
    """
    def forward(self, logits, y_one_hot):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        self.probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        self.y = y_one_hot
        N = logits.shape[0]
        loss = -np.sum(y_one_hot * np.log(self.probs + 1e-12)) / N
        return loss

    def backward(self):
        N = self.y.shape[0]
        return (self.probs - self.y) / N


class MSELoss:
    """
    Mean Squared Error for regression: L = (1/N) * sum((y_hat - y)^2)
    Gradient: dL/dy_hat = 2*(y_hat - y) / N
    The factor of 2 is often absorbed into the learning rate, but we keep it
    explicit here for mathematical correctness.
    """
    def forward(self, y_hat, y):
        self.diff = y_hat - y
        N = y.shape[0]
        return np.sum(self.diff**2) / N

    def backward(self):
        N = self.diff.shape[0]
        return 2 * self.diff / N


# ==============================================================================
# 7. REGULARIZATION — WEIGHT PENALTY TERMS
# ==============================================================================
# Added to the loss before backward pass. The gradient of the penalty is
# added to dL/dW before the optimizer step.

def l1_penalty(params, lam):
    """
    L1 = lambda * sum(|W|)
    Gradient: lambda * sign(W)
    Effect: drives weights exactly to zero → sparse models.
    """
    loss = sum(lam * np.sum(np.abs(W)) for W, _ in params if W.ndim > 1)
    grads = [lam * np.sign(W) if W.ndim > 1 else np.zeros_like(W) for W, _ in params]
    return loss, grads

def l2_penalty(params, lam):
    """
    L2 = lambda * sum(W^2)
    Gradient: 2 * lambda * W
    Effect: shrinks weights uniformly toward zero, does not zero them out.
    """
    loss = sum(lam * np.sum(W**2) for W, _ in params if W.ndim > 1)
    grads = [2 * lam * W if W.ndim > 1 else np.zeros_like(W) for W, _ in params]
    return loss, grads

def elastic_net_penalty(params, lam, alpha=0.5):
    """
    Elastic Net = alpha * L1 + (1 - alpha) * L2
    Combines sparse selection (L1) with grouping effect (L2).
    alpha=1 → pure L1, alpha=0 → pure L2.
    """
    l1_loss, l1_g = l1_penalty(params, lam * alpha)
    l2_loss, l2_g = l2_penalty(params, lam * (1 - alpha))
    total_loss = l1_loss + l2_loss
    total_grads = [g1 + g2 for g1, g2 in zip(l1_g, l2_g)]
    return total_loss, total_grads


# ==============================================================================
# 8. OPTIMIZERS
# ==============================================================================
# All optimizers update parameters in-place given their gradients.
# SGD is the baseline; each subsequent optimizer adds an idea on top.

class SGD:
    """
    W ← W - lr * dW
    Simple, noisy, requires small lr, slow on ill-conditioned loss surfaces.
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params):
        for W, dW in params:
            if dW is not None:
                W -= self.lr * dW


class SGDMomentum:
    """
    Momentum accumulates a velocity vector in directions of persistent gradient.
    v ← beta * v + dW
    W ← W - lr * v
    Effect: accelerates along consistent gradient directions, dampens oscillation.
    beta=0.9 is standard → effectively looks back ~10 steps.
    """
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def update(self, params):
        for i, (W, dW) in enumerate(params):
            if dW is None:
                continue
            if i not in self.v:
                self.v[i] = np.zeros_like(W)
            self.v[i] = self.beta * self.v[i] + dW
            W -= self.lr * self.v[i]


class Adam:
    """
    Adaptive Moment Estimation. Combines momentum (1st moment) with
    RMSProp (2nd moment — per-parameter adaptive lr).

    m ← beta1 * m + (1 - beta1) * dW        ← exponential moving avg of gradient
    v ← beta2 * v + (1 - beta2) * dW^2      ← exponential moving avg of squared grad
    m_hat = m / (1 - beta1^t)               ← bias correction (important early on)
    v_hat = v / (1 - beta2^t)
    W ← W - lr * m_hat / (sqrt(v_hat) + eps)

    Bias correction: m and v start at 0, so early estimates are biased
    toward 0. Dividing by (1 - beta^t) corrects this at early timesteps.

    Typical hyperparams: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8.
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params):
        self.t += 1
        for i, (W, dW) in enumerate(params):
            if dW is None:
                continue
            if i not in self.m:
                self.m[i] = np.zeros_like(W)
                self.v[i] = np.zeros_like(W)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * dW
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * dW**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW:
    """
    AdamW decouples weight decay from the gradient update.
    In standard L2 regularization added to the loss, the decay gets scaled
    by the adaptive learning rate (v_hat denominator), weakening it.
    AdamW applies weight decay directly to the weight before the Adam step:

    W ← W * (1 - lr * weight_decay)   ← decoupled decay
    ... then Adam update as usual ...

    This is the standard for modern LLM training (GPT, BERT, etc.).
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params):
        self.t += 1
        for i, (W, dW) in enumerate(params):
            if dW is None:
                continue
            if i not in self.m:
                self.m[i] = np.zeros_like(W)
                self.v[i] = np.zeros_like(W)
            # Decoupled weight decay (only on weights, not biases)
            if W.ndim > 1:
                W *= (1 - self.lr * self.wd)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * dW
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * dW**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==============================================================================
# 9. GRADIENT CLIPPING BY NORM
# ==============================================================================
# Exploding gradients make training unstable (common in RNNs, deep nets).
# Solution: compute the global L2 norm of all gradients. If it exceeds
# a threshold, scale all gradients down so the norm equals the threshold.
# This preserves gradient direction, just reduces magnitude.
#
# global_norm = sqrt(sum over all params of sum(grad^2))
# if global_norm > max_norm:
#     scale = max_norm / global_norm
#     grad *= scale  (for every grad)

def clip_gradients(params, max_norm=1.0):
    total_norm = np.sqrt(sum(
        np.sum(dW**2) for _, dW in params if dW is not None
    ))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for _, dW in params:
            if dW is not None:
                dW *= scale
    return total_norm


# ==============================================================================
# 10. COSINE ANNEALING LR SCHEDULER
# ==============================================================================
# Learning rate schedule: start high (explore), end low (converge).
# Cosine shape is smooth and avoids sharp drops.
#
# eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T))
#
# At t=0:  cos(0) = 1  → eta_t = eta_max
# At t=T:  cos(pi) = -1 → eta_t = eta_min
# The cosine curve gives a smooth annealing between these.

class CosineAnnealingScheduler:
    def __init__(self, optimizer, eta_max, eta_min=0.0, T_max=100):
        self.optimizer = optimizer
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max = T_max
        self.t = 0

    def step(self):
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(np.pi * self.t / self.T_max)
        )
        self.optimizer.lr = lr
        self.t += 1
        return lr


# ==============================================================================
# 11. SIMPLE MLP WRAPPER
# ==============================================================================
# Puts layers together for clean experiment code below.

class MLP:
    def __init__(self, layer_sizes, activation_cls=ReLU, init='he',
                 dropout_keep=None, use_batchnorm=False):
        self.layers = []
        self.activations = []
        self.dropouts = []
        self.batchnorms = []
        self.use_batchnorm = use_batchnorm

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], init=init))
            if i < len(layer_sizes) - 2:
                self.activations.append(activation_cls())
                if use_batchnorm:
                    self.batchnorms.append(BatchNorm(layer_sizes[i+1]))
                if dropout_keep is not None:
                    self.dropouts.append(Dropout(dropout_keep))
            else:
                self.activations.append(None)

    def forward(self, x):
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            x = layer.forward(x)
            if act is not None:
                if self.use_batchnorm and i < len(self.batchnorms):
                    x = self.batchnorms[i].forward(x)
                x = act.forward(x)
                if i < len(self.dropouts):
                    x = self.dropouts[i].forward(x)
        return x

    def backward(self, grad):
        for i in reversed(range(len(self.layers))):
            if self.activations[i] is not None:
                if i < len(self.dropouts):
                    grad = self.dropouts[i].backward(grad)
                grad = self.activations[i].backward(grad)
                if self.use_batchnorm and i < len(self.batchnorms):
                    grad = self.batchnorms[i].backward(grad)
            grad = self.layers[i].backward(grad)
        return grad

    def params(self):
        p = []
        for layer in self.layers:
            p.extend(layer.params())
        for bn in self.batchnorms:
            p.extend(bn.params())
        return p

    def set_training(self, mode):
        for d in self.dropouts:
            d.training = mode
        for bn in self.batchnorms:
            bn.training = mode


def train(model, optimizer, X_train, y_train, X_val, y_val,
          epochs=100, batch_size=64, reg=None, lam=1e-4, clip_norm=None):
    loss_fn = CrossEntropyLoss()
    n = X_train.shape[0]
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        model.set_training(True)
        idx = np.random.permutation(n)
        epoch_loss = 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            X_b, y_b = X_train[batch_idx], y_train[batch_idx]

            logits = model.forward(X_b)
            loss = loss_fn.forward(logits, y_b)

            if reg is not None:
                params = [(W, dW) for W, dW in model.params() if W.ndim > 1]
                reg_loss, reg_grads = reg(params, lam)
                loss += reg_loss

            grad = loss_fn.backward()
            model.backward(grad)

            if reg is not None:
                for (W, dW), rg in zip(
                    [(W, dW) for W, dW in model.params() if W.ndim > 1], reg_grads
                ):
                    if dW is not None:
                        dW += rg

            if clip_norm is not None:
                clip_gradients(model.params(), max_norm=clip_norm)

            optimizer.update(model.params())
            epoch_loss += loss

        train_losses.append(epoch_loss / (n // batch_size))

        # Validation
        model.set_training(False)
        val_logits = model.forward(X_val)
        val_loss = loss_fn.forward(val_logits, y_val)
        val_losses.append(val_loss)
        preds = val_logits.argmax(axis=1)
        labels = y_val.argmax(axis=1)
        val_accs.append((preds == labels).mean())

    return train_losses, val_losses, val_accs


# ==============================================================================
# 12. BAYES-OPTIMAL CLASSIFIER
# ==============================================================================
# On a toy dataset with known Gaussian class distributions, the Bayes-optimal
# classifier achieves the minimum possible error rate.
# Decision rule: assign x to the class with the highest posterior P(C|x).
# By Bayes' theorem: P(C|x) ∝ P(x|C) * P(C).
# With equal priors: just pick the class with the highest likelihood P(x|C).
# This gives us the theoretical ceiling accuracy any model can achieve.

class BayesOptimalClassifier:
    def __init__(self, means, covs, priors=None):
        self.means = means
        self.covs = covs
        self.priors = priors if priors else [1/len(means)] * len(means)

    def predict(self, X):
        posteriors = np.array([
            p * multivariate_normal.pdf(X, mean=mu, cov=cov)
            for mu, cov, p in zip(self.means, self.covs, self.priors)
        ]).T
        return posteriors.argmax(axis=1)


# ==============================================================================
# EXPERIMENTS
# ==============================================================================

def make_dataset(n=2000, n_classes=3):
    """Toy Gaussian mixture dataset."""
    means = [np.array([0, 0]), np.array([3, 1]), np.array([1.5, 3])]
    covs  = [np.eye(2)*0.8, np.eye(2)*0.8, np.eye(2)*0.8]
    X_list, y_list = [], []
    per_class = n // n_classes
    for c, (mu, cov) in enumerate(zip(means, covs)):
        X_list.append(np.random.multivariate_normal(mu, cov, per_class))
        y_list.append(np.full(per_class, c))
    X = np.vstack(X_list)
    y_int = np.hstack(y_list)
    y = np.eye(n_classes)[y_int]
    n_actual = X.shape[0]
    idx = np.random.permutation(n_actual)
    X, y, y_int = X[idx], y[idx], y_int[idx]
    split = int(0.8 * n_actual)
    return (X[:split], y[:split], X[split:], y[split:],
            y_int[:split], y_int[split:], means, covs)


def run_experiments():
    print("Preparing dataset...")
    X_tr, y_tr, X_val, y_val, y_tr_int, y_val_int, means, covs = make_dataset()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Deep Learning From Scratch — Experiments", fontsize=13)

    # ------------------------------------------------------------------
    # Experiment 1: ReLU vs Sigmoid — vanishing gradient
    # ------------------------------------------------------------------
    print("Exp 1: ReLU vs Sigmoid...")
    results = {}
    for name, act_cls in [('ReLU', ReLU), ('Sigmoid', Sigmoid)]:
        model = MLP([2, 64, 64, 64, 3], activation_cls=act_cls)
        opt = Adam(lr=1e-3)
        tl, vl, va = train(model, opt, X_tr, y_tr, X_val, y_val, epochs=80)
        results[name] = vl

    ax = axes[0, 0]
    for name, vl in results.items():
        ax.plot(vl, label=name)
    ax.set_title("Exp 1: ReLU vs Sigmoid (val loss)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()

    # ------------------------------------------------------------------
    # Experiment 2: Adam vs SGD — convergence speed
    # ------------------------------------------------------------------
    print("Exp 2: Adam vs SGD...")
    results2 = {}
    for name, opt in [('Adam', Adam(lr=1e-3)), ('SGD', SGD(lr=0.05)),
                      ('SGD+Momentum', SGDMomentum(lr=0.05))]:
        model = MLP([2, 64, 64, 3], activation_cls=ReLU)
        tl, vl, va = train(model, opt, X_tr, y_tr, X_val, y_val, epochs=80)
        results2[name] = vl

    ax = axes[0, 1]
    for name, vl in results2.items():
        ax.plot(vl, label=name)
    ax.set_title("Exp 2: Optimizer comparison (val loss)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()

    # ------------------------------------------------------------------
    # Experiment 3: Dropout vs no Dropout — overfitting
    # ------------------------------------------------------------------
    print("Exp 3: Dropout vs no dropout...")
    results3 = {}
    for name, keep in [('With dropout (0.7)', 0.7), ('No dropout', None)]:
        model = MLP([2, 128, 128, 3], activation_cls=ReLU, dropout_keep=keep)
        opt = Adam(lr=1e-3)
        tl, vl, va = train(model, opt, X_tr, y_tr, X_val, y_val, epochs=80)
        results3[name] = (tl, vl)

    ax = axes[0, 2]
    for name, (tl, vl) in results3.items():
        style = '-' if 'With' in name else '--'
        ax.plot(tl, linestyle=style, label=f'{name} (train)')
        ax.plot(vl, linestyle=':', label=f'{name} (val)')
    ax.set_title("Exp 3: Dropout — train vs val gap")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Experiment 4: Xavier vs Random init — gradient norms
    # ------------------------------------------------------------------
    print("Exp 4: Xavier vs random init...")
    grad_norms = {}
    for name, init in [('Xavier', 'xavier'), ('He', 'he'), ('Random', 'random')]:
        model = MLP([2, 64, 64, 64, 3], activation_cls=Sigmoid, init=init)
        loss_fn = CrossEntropyLoss()
        logits = model.forward(X_tr[:64])
        loss_fn.forward(logits, y_tr[:64])
        model.backward(loss_fn.backward())
        norms = [np.linalg.norm(dW) for _, dW in model.params() if dW is not None]
        grad_norms[name] = norms

    ax = axes[1, 0]
    for name, norms in grad_norms.items():
        ax.plot(norms, marker='o', label=name)
    ax.set_title("Exp 4: Gradient norms at step 0 (Sigmoid)")
    ax.set_xlabel("Parameter group"); ax.set_ylabel("||grad||"); ax.legend()

    # ------------------------------------------------------------------
    # Experiment 5: All activations — validation accuracy
    # ------------------------------------------------------------------
    print("Exp 5: All activations...")
    act_classes = [('ReLU', ReLU), ('LeakyReLU', LeakyReLU), ('ELU', ELU),
                   ('GELU', GELU), ('Swish', Swish), ('Sigmoid', Sigmoid), ('Tanh', Tanh)]
    ax = axes[1, 1]
    for name, act_cls in act_classes:
        model = MLP([2, 64, 64, 3], activation_cls=act_cls)
        opt = Adam(lr=1e-3)
        _, _, va = train(model, opt, X_tr, y_tr, X_val, y_val, epochs=80)
        ax.plot(va, label=name)
    ax.set_title("Exp 5: All activations — val accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Bayes-optimal classifier comparison
    # ------------------------------------------------------------------
    print("Bayes-optimal classifier...")
    bayes = BayesOptimalClassifier(means, covs)
    bayes_acc = (bayes.predict(X_val) == y_val_int).mean()

    model_best = MLP([2, 128, 128, 3], activation_cls=ReLU)
    opt = Adam(lr=1e-3)
    _, _, va = train(model_best, opt, X_tr, y_tr, X_val, y_val, epochs=150)
    nn_acc = va[-1]

    ax = axes[1, 2]
    ax.bar(['Bayes Optimal', 'NN (ReLU+Adam)'], [bayes_acc, nn_acc],
           color=['#1D9E75', '#185FA5'])
    ax.set_ylim(0, 1)
    ax.set_title("Bayes-optimal vs NN accuracy")
    ax.set_ylabel("Val accuracy")
    for i, v in enumerate([bayes_acc, nn_acc]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig("dl_scratch_experiments.png", dpi=130, bbox_inches='tight')
    print("\nAll experiments done. Plot saved to dl_scratch_experiments.png")
    print(f"Bayes-optimal accuracy: {bayes_acc:.4f}")
    print(f"NN accuracy:            {nn_acc:.4f}")


if __name__ == "__main__":
    run_experiments()
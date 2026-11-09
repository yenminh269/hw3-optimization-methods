# ============================================================================
# CONFIGURATION
# ============================================================================
 
VIEW_TYPE = 1  # 1 = B/W, 2 = Color
 
PARAM_A = 0.5
PARAM_B = 10.0
 
THETA_INIT = np.array([9.0, 1.5])
 
NUM_STEPS = 250
 
# Standard GD
ETA_GD = 0.085
 
# Decay GD
ETA_DECAY_INIT = 0.085
GAMMA_DECAY = 0.96
 
# Momentum
ETA_MOM = 0.045
BETA_MOM = 0.85
 
# AdaGrad
ETA_ADAGRAD = 1.20
EPS_ADAGRAD = 1e-8
 
# RMSProp
ETA_RMSPROP = 0.25
GAMMA_RMSPROP = 0.90
EPS_RMSPROP = 1e-8
 
# Adam
ETA_ADAM = 0.35
BETA1_ADAM = 0.85
BETA2_ADAM = 0.99
EPS_ADAM = 1e-8

 
import numpy as np
 
 
class Optimizer:
    def reset(self):
        pass
 
    def step(self, theta, grad_fn, t):
        raise NotImplementedError
 
 
class GradientDescent(Optimizer):
    def __init__(self, lr=0.001):
        self.name = "Gradient Descent"
        self.lr = lr
 
    def reset(self):
        pass
 
    def step(self, theta, grad_fn, t):
        g = grad_fn(theta)
        return -self.lr * g
 
 
class Momentum(Optimizer):
    def __init__(self, lr=0.001, beta=0.9):
        self.name = "Momentum"
        self.lr = lr
        self.beta = beta
        self.reset()
 
    def reset(self):
        self.velocity = None
 
    def step(self, theta, grad_fn, t):
        g = grad_fn(theta)
 
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)
 
        self.velocity = (
            self.beta * self.velocity
            - self.lr * g
        )
 
        return self.velocity
 
 
class Nesterov(Optimizer):
    def __init__(self, lr=0.001, beta=0.9):
        self.name = "Nesterov"
        self.lr = lr
        self.beta = beta
        self.reset()
 
    def reset(self):
        self.velocity = None
 
    def step(self, theta, grad_fn, t):
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)
 
        # Look-ahead position
        theta_lookahead = theta + self.beta * self.velocity
 
        # Gradient evaluated at look-ahead position
        g = grad_fn(theta_lookahead)
 
        self.velocity = (
            self.beta * self.velocity
            - self.lr * g
        )
 
        return self.velocity
 
 
class Adagrad(Optimizer):
    def __init__(self, lr=0.1, eps=1e-8):
        self.name = "Adagrad"
        self.lr = lr
        self.eps = eps
        self.reset()
 
    def reset(self):
        self.G = None
 
    def step(self, theta, grad_fn, t):
        g = grad_fn(theta)
 
        if self.G is None:
            self.G = np.zeros_like(theta)
 
        self.G += g ** 2
 
        return (
            -self.lr * g
            / (np.sqrt(self.G) + self.eps)
        )
 
 
class RMSprop(Optimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.name = "RMSProp"
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.reset()
 
    def reset(self):
        self.v = None
 
    def step(self, theta, grad_fn, t):
        g = grad_fn(theta)
 
        if self.v is None:
            self.v = np.zeros_like(theta)
 
        self.v = (
            self.beta * self.v
            + (1 - self.beta) * (g ** 2)
        )
 
        return (
            -self.lr * g
            / (np.sqrt(self.v) + self.eps)
        )
 
 
class Adam(Optimizer):
    def __init__(
        self,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8
    ):
        self.name = "Adam"
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.reset()
 
    def reset(self):
        self.m = None
        self.v = None
 
    def step(self, theta, grad_fn, t):
        g = grad_fn(theta)
 
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
 
        self.m = (
            self.beta1 * self.m
            + (1 - self.beta1) * g
        )
 
        self.v = (
            self.beta2 * self.v
            + (1 - self.beta2) * (g ** 2)
        )
 
        m_hat = self.m / (
            1 - self.beta1 ** t
        )
 
        v_hat = self.v / (
            1 - self.beta2 ** t
        )
 
        return (
            -self.lr
            * m_hat
            / (np.sqrt(v_hat) + self.eps)
        )
 
 
class AdamW(Optimizer):
    def __init__(
        self,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01
    ):
        self.name = "AdamW"
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.reset()
 
    def reset(self):
        self.m = None
        self.v = None
 
    def step(self, theta, grad_fn, t):
        g = grad_fn(theta)
 
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
 
        self.m = (
            self.beta1 * self.m
            + (1 - self.beta1) * g
        )
 
        self.v = (
            self.beta2 * self.v
            + (1 - self.beta2) * (g ** 2)
        )
 
        m_hat = self.m / (
            1 - self.beta1 ** t
        )
 
        v_hat = self.v / (
            1 - self.beta2 ** t
        )
 
        adam_update = (
            -self.lr
            * m_hat
            / (np.sqrt(v_hat) + self.eps)
        )
 
        weight_decay_update = (
            -self.lr
            * self.weight_decay
            * theta
        )
 
        return adam_update + weight_decay_update
 

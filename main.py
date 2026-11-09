
from optimizers import GradientDescent, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW
from experiment import run_optimizer 
from functions import rosenbrock, rosenbrock_gradient

optimizers = [
    GradientDescent(lr=0.001),
    Momentum(lr=0.001, beta=0.9),
    Nesterov(lr=0.001, beta=0.9),
    Adagrad(lr=0.1),
    RMSprop(lr=0.001),
    Adam(lr=0.01),
    AdamW(lr=0.01, weight_decay=0.01),
]
 
for opt in optimizers:
  result = run_optimizer(
    opt,
    theta_init=np.array([-1.2, 1.0]),
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient)

 
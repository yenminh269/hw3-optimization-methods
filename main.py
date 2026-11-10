from experiment import run_optimizer 
from functions import rosenbrock, rosenbrock_gradient, sphere, sphere_gradient
import numpy as np
from optimizers import GradientDescent, Momentum, Nesterov, Adagrad, RMSprop, Adam, AdamW
from plots import (plot_trajectory, plot_convergence)

optimizers = [
    GradientDescent(lr=0.002),
    Momentum(lr=0.002, beta=0.9),
    Nesterov(lr=0.001, beta=0.9),
    Adagrad(lr=1.0),
    RMSprop(lr=0.01, beta=0.9),
    Adam(lr=0.1),
    AdamW(lr=0.1, weight_decay=0.0001),
]

results = []
 
for opt in optimizers:
  result = run_optimizer(
    opt,
    theta_init=np.array([-1.5, 2.0]),
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient)

  results.append(result)

  print(
    f"{opt.name}: "
    f"loss={result['loss'][-1]:.6e}, "
    f"iterations={result['iterations']}, "
    f"theta={result['final_theta']}"
    f"distance_to_minimum={result['distance']:.6e}"
  )

plot_convergence(results)
plot_trajectory(results)
from experiment import run_optimizer
from functions import rosenbrock, rosenbrock_gradient
from optimizers import (
    GradientDescent,
    Momentum,
    Nesterov,
    Adagrad,
    RMSprop,
    Adam,
    AdamW
)
import numpy as np
 
 
THETA_INIT = np.array([-1.5, 2.0])
 
 
def tune_optimizer(
    optimizer_class,
    learning_rates,
    objective_fn,
    grad_fn,
    theta_init,
    **kwargs
):
    best_result = None
 
    print(f"\n===== {optimizer_class.__name__} =====")
 
    for lr in learning_rates:
 
        opt = optimizer_class(
            lr=lr,
            **kwargs
        )
 
        result = run_optimizer(
            opt,
            theta_init=theta_init,
            objective_fn=objective_fn,
            grad_fn=grad_fn
        )
 
        loss = result["loss"][-1]
        distance = result["distance"]
 
        print(
            f"lr={lr:<8} "
            f"loss={loss:.6e} "
            f"distance={distance:.6e}"
        )
 
        if (
            best_result is None
            or distance < best_result["distance"]
        ):
            best_result = {
                "lr": lr,
                "loss": loss,
                "distance": distance,
                "iterations": result["iterations"],
                "theta": result["final_theta"]
            }
 
    print("\nBest Result")
    print(best_result)
 
    return best_result
 
 
# --------------------------------------------------
# Tune Gradient Descent
# --------------------------------------------------
 
best_gd = tune_optimizer(
    GradientDescent,
    learning_rates=[
        0.0001,
        0.0005,
        0.001,
        0.002,
        0.005
    ],
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient,
    theta_init=THETA_INIT
)
 
# --------------------------------------------------
# Tune Momentum
# --------------------------------------------------
 
best_momentum = tune_optimizer(
    Momentum,
    learning_rates=[
        0.0001,
        0.0005,
        0.001,
        0.002,
        0.005
    ],
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient,
    theta_init=THETA_INIT,
    beta=0.9
)
 
# --------------------------------------------------
# Tune Nesterov
# --------------------------------------------------
 
best_nesterov = tune_optimizer(
    Nesterov,
    learning_rates=[
        0.0001,
        0.0005,
        0.001,
        0.002,
        0.005
    ],
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient,
    theta_init=THETA_INIT,
    beta=0.9
)
 
# --------------------------------------------------
# Tune Adagrad
# --------------------------------------------------
 
best_adagrad = tune_optimizer(
    Adagrad,
    learning_rates=[
        0.1,
        0.2,
        0.5,
        1.0
    ],
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient,
    theta_init=THETA_INIT
)
 
# --------------------------------------------------
# Tune RMSprop
# --------------------------------------------------
 
best_rmsprop = tune_optimizer(
    RMSprop,
    learning_rates=[
        0.001,
        0.005,
        0.01,
        0.02,
        0.05
    ],
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient,
    theta_init=THETA_INIT,
    beta=0.9
)
 
# --------------------------------------------------
# Tune Adam
# --------------------------------------------------
 
best_adam = tune_optimizer(
    Adam,
    learning_rates=[
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1
    ],
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient,
    theta_init=THETA_INIT
)
 
# --------------------------------------------------
# Tune AdamW
# --------------------------------------------------
 
best_adamw = tune_optimizer(
    AdamW,
    learning_rates=[
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1
    ],
    objective_fn=rosenbrock,
    grad_fn=rosenbrock_gradient,
    theta_init=THETA_INIT,
    weight_decay=0.001
)
 
# --------------------------------------------------
# Final Summary
# --------------------------------------------------
 
print("\n\n===== FINAL BEST SETTINGS =====")
 
print("GD      :", best_gd)
print("Momentum:", best_momentum)
print("Nesterov:", best_nesterov)
print("Adagrad :", best_adagrad)
print("RMSprop :", best_rmsprop)
print("Adam    :", best_adam)
print("AdamW   :", best_adamw)
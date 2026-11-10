import numpy as np

def run_optimizer(opt, theta_init, objective_fn, grad_fn, MAX_ITER=1000, TOL=1e-8):
    opt.reset()
 
    theta = theta_init.copy()
 
    traj = [theta.copy()]
    losses = [objective_fn(theta)]
 
    for t in range(1, MAX_ITER + 1):
 
        d_theta = opt.step(theta, grad_fn, t)
 
        theta += d_theta

       
        traj.append(theta.copy())
        losses.append(objective_fn(theta))
 
        if np.linalg.norm(d_theta) < TOL:
            break

    distance = np.sum(
        (theta - np.array([1.0, 1.0]))**2
        )
 
    return {
        "name": opt.name,
        "traj": np.array(traj),
        "loss": np.array(losses),
        "iterations": t,
        "final_theta": theta.copy(),
        "final_loss": losses[-1],
        "distance": distance
    }
 
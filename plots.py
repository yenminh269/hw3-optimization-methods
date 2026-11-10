import numpy as np
import matplotlib.pyplot as plt
 
from functions import rosenbrock
 
 
def plot_convergence(results):
    """
    Plot loss vs iteration on a logarithmic scale.
    """
 
    plt.figure(figsize=(10, 6))
 
    for result in results:
 
        plt.semilogy(
            result["loss"],
            linewidth=2,
            label=result["name"]
        )
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Convergence Comparison on Rosenbrock Function")
 
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
 
    plt.legend()
 
    plt.tight_layout()
 
    plt.savefig(
        "convergence_plot.png",
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.show()
 
 
def plot_trajectory(results):
    """
    Plot optimizer trajectories on Rosenbrock contour map.
    """
 
    beta_min = -2.0
    beta_max = 2.0
 
    gamma_min = -1.0
    gamma_max = 3.0
 
    beta = np.linspace(beta_min, beta_max, 500)
    gamma = np.linspace(gamma_min, gamma_max, 500)
 
    B, G = np.meshgrid(beta, gamma)
 
    Z = 100 * (G - B**2) ** 2 + (1 - B) ** 2
 
    plt.figure(figsize=(10, 8))
 
    contour_levels = np.logspace(-1, 3, 25)
 
    plt.contour(
        B,
        G,
        Z,
        levels=contour_levels,
        cmap="viridis"
    )
 
    for result in results:
 
        traj = result["traj"]

        print(result["name"], result["traj"].shape)

        plt.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=2,
            label=result["name"]
        )
        #mark final position
        plt.scatter(
            traj[-1, 0],
            traj[-1, 1],
            s=80, 
            marker="X",
            color="black"
        )
 
    # Starting point
    plt.scatter(
        -1.5,
        2.0,
        s=120,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Start Point"
    )
 
    # True optimum
    plt.scatter(
        1.0,
        1.0,
        s=250,
        marker="*",
        color="red",
        label="Optimum (1,1)"
    )
 
    plt.xlabel("Beta")
    plt.ylabel("Gamma")
 
    plt.title(
        "Optimizer Trajectories on Rosenbrock Contours"
    )
 
    plt.legend()
 
    plt.tight_layout()
 
    plt.savefig(
        "trajectory_map.png",
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.show()

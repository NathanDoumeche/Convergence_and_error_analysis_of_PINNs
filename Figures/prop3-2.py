import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import os

if __name__ == "__main__":
    # Samples
    np.random.seed(1)
    x = np.linspace(-1, 1, 20)
    t = np.linspace(0, 1, 20)
    X, T = np.meshgrid(x, t)

    # Samples for initial condition
    x0 = np.linspace(-1, 1, 50)
    t0 = np.zeros(x0.shape)
    # X0,T0 = np.meshgrid(np.linspace(-1,1,50),0)

    p = 100
    Z = np.tanh(X + .5 + p * T) - np.tanh(X - .5 + p * T) + np.tanh(.5 + p * T) - np.tanh(1.5 + p * T)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.tick_params(axis='both', which='major', labelsize=11)
    cmap = colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "lightsalmon"])

    # Plot the surface
    surf = ax.plot_surface(X, T, Z, cmap=cmap,
                           linewidth=0, alpha=0.6)  # antialiased=False #cm.RdBu
    fig.colorbar(surf, ax=ax, location = 'left')

    # Plot the initial condition
    z0 = np.tanh(x0 + .5) - np.tanh(x0 - .5) + np.tanh(.5) - np.tanh(1.5)
    plt.plot(x0, t0, z0,
             c='black', linestyle='--',
             label='Initial condition')
    plt.legend(fontsize=12)
    plt.xlabel("Space", fontsize=12)
    plt.ylabel("Time", fontsize=12)
    ax.view_init(20, -20)
    plt.savefig(os.path.join("Outputs", "prop3-2.pdf"), bbox_inches="tight")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os


def tanh_H(x, H=2):
    output = x
    for i in range(H):
        output = np.tanh(output)
    return output


def initial_condition(x):
    return tanh_H(x+ .5, H=2) - tanh_H(x- .5, H=2) + tanh_H(.5, H=2) - tanh_H(1.5, H=2)


def boundary_condition(t):
    return np.zeros(np.shape(t))


def propagation_heat(u, dx, dt):
    #Laplacian computation
    u_x_plus = np.concatenate((u[1:], [0]))
    u_x_minus = np.concatenate(([0], u[:len(u)-1]))
    delta_u = (u_x_plus + u_x_minus - 2*u)/(dx**2)

    #Heat equation propagation at t+dt
    u_next = u + dt*delta_u
    u_next[0] = 0
    u_next[len(u)-1] = 0
    return u_next


if __name__ == "__main__":
    #Context variables
    x_min = -1
    x_max = 1
    t_min = 0
    t_max = 1
    n_x = 100
    n_t = 2*n_x**2

    dx = (x_max-x_min)/n_x
    dt = (t_max-t_min)/n_t

    #Mesh creation and initial condition sampling
    x = np.linspace(x_min, x_max, n_x)
    t = np.linspace(t_min, t_max, n_x)
    X, T = np.meshgrid(x, t)

    x_ini = np.linspace(-1, 1, 50)
    t_ini = np.zeros(x_ini.shape)
    x_boundary = np.ones(50)
    t_boundary = np.linspace(0, 1, 50)
    x0 = np.concatenate((-x_boundary, x_ini, x_boundary))
    t0 = np.concatenate((t_boundary, t_ini, t_boundary))


    #Overfitting network
    p = 1000
    overfitting_nn = tanh_H(X + .5 + p * T) - tanh_H(X - .5 + p * T) + tanh_H(.5 + p * T) - tanh_H(1.5 + p * T)

    #U star
    u_0 = initial_condition(x)
    u = [u_0]
    for i in range(n_t-1):
        u.append(propagation_heat(u[i], dx, dt))

    u_star = np.array([u[int(n_t/n_x*i)] for i in range(n_x)])


    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Plotting the overfitting network
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=11)
    cmap = colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "lightsalmon"])
    surf = ax.plot_surface(X, T, overfitting_nn, cmap=cmap, linewidth=0, alpha=0.6)
    z0 = initial_condition(x0)
    plt.plot(x0, t0, z0, c='black', linestyle='--', label='Initial and boundary conditions')
    plt.legend(fontsize=12)
    plt.xlabel("Space", fontsize=12)
    plt.ylabel("Time", fontsize=12)
    ax.view_init(20, -20)


    # Plotting u_star
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=11)
    cmap = colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "lightsalmon"])
    surf = ax.plot_surface(X, T, u_star, cmap=cmap, linewidth=0, alpha=0.6)
    z0 = initial_condition(x0)
    plt.plot(x0, t0, z0, c='black', linestyle='--', label='Initial and boundary conditions')
    plt.legend(fontsize=12)
    plt.xlabel("Space", fontsize=12)
    plt.ylabel("Time", fontsize=12)
    ax.view_init(20, -20)
    plt.savefig(os.path.join("Outputs", "prop3-2.pdf"), bbox_inches="tight")

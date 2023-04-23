import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from src.regularized_neural_network import NeuralNetwork

matplotlib.rcParams.update({'font.size': 14})

def base_10(n):
    return "{"+str(int(np.log(n)/np.log(10)))+"}"


def figure_monitoring(train_loss, test_loss, overfitting_gap_list, physics_inconsistency, n, n_e, n_r, D):
    plt.figure()
    plt.plot(np.log(np.array(train_loss)), label="$\ln(R_{10^"+base_10(n)+", 10^" + base_10(n_e) +
                                                 ", 10^"+base_10(n_r)+"}^{(\mathrm{reg})})$")
    plt.plot(np.log(np.array(np.abs(overfitting_gap_list))), label="ln(|overfitting gap|)")
    plt.plot(np.log(np.array(physics_inconsistency)), label="ln(PI)")
    plt.plot(np.log(np.array(test_loss)), linestyle='dashed',
             label="$\ln(\int_\Omega \|u^\\star - u_{\\hat \\theta(p, 10^" + base_10(n_e) + ", 10^" + base_10(n_r) +
                   ", 10^" + base_10(D) + ")}\|_2^2d\\mu_X)$")
    plt.legend()
    plt.xlabel("Epoch p")
    plt.savefig(os.path.join("Outputs", "training", "perf_" + str(n) + ".pdf"), bbox_inches="tight")


def linear_reg():
    results = pd.read_csv(os.path.join("Outputs", "validation_loss.csv"))
    log_test_loss = results[["log(Validation_loss)"]].to_numpy().reshape(-1)
    log_n = results[["log(n)"]].to_numpy().reshape(-1)

    coefficients = np.polyfit(log_n, log_test_loss, 1)
    m = coefficients[0]
    b = coefficients[1]
    print("Slope = " + str(m))

    # Generate regression line points
    regression_n = np.linspace(min(log_n), max(log_n), 100)
    regression_loss = m * regression_n + b

    # Plot the data points and regression line
    plt.figure()
    plt.plot(log_n, log_test_loss, color="dodgerblue", linestyle='dashed',  label="ln(err)(n)", marker="x",
             markersize = 15, markeredgewidth=3)
    plt.plot(regression_n, regression_loss, color='lightsalmon',
             label='Linear regression: \n ln(err)(n) = ' + str(round(m, 2)) + '$\;\ln(n)\;$' + str(round(b, 2)))
    plt.xlabel('ln(n)')
    plt.ylabel("ln(err)(n)")
    plt.legend()
    plt.savefig(os.path.join("Outputs", "linear_regression.pdf"), bbox_inches="tight")
    return


def PI():
    results = pd.read_csv(os.path.join("Outputs", "validation_loss.csv"))
    log_PI = results[["log(PI)"]].to_numpy().reshape(-1)
    log_n = results[["log(n)"]].to_numpy().reshape(-1)

    plt.figure()
    plt.plot(log_n, log_PI, color="dodgerblue", linestyle='dashed',  label="ln(PI)", marker="x",
             markersize = 15, markeredgewidth=3)
    plt.axhline(y=-1.6, color='lightsalmon', linestyle='-', label='ln(PI($u^\star$))')
    plt.xlabel('ln(n)')
    plt.ylabel("ln(PI)")
    plt.legend()
    plt.savefig(os.path.join("Outputs", "log_PI.pdf"), bbox_inches="tight")
    return


def figure_nn(u_star, D, n):
    model = NeuralNetwork(D)
    model.load_state_dict(torch.load(os.path.join("Outputs", "NN_weights", "model_"+str(n)+".pt")))
    model.eval()

    steps = 51
    t = torch.linspace(0, 1, steps)
    x = torch.linspace(0, 1, steps)
    t_mesh, x_mesh = torch.meshgrid(x, t)
    t_mesh_reshape = t_mesh.reshape(-1, 1)
    x_mesh_reshape = x_mesh.reshape(-1, 1)
    tx = torch.concat((t_mesh_reshape, x_mesh_reshape), dim=1)
    pred = model(tx).reshape(-1, steps).detach()


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    #Plot the neural network
    ax.tick_params(axis='both', which='major', labelsize=12)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue", "lightsalmon"])
    ax.plot_surface(x_mesh.numpy(), t_mesh.numpy(), pred.numpy(), cmap=cmap, linewidth=0, alpha=0.6)
    plt.plot(x_mesh[0].numpy(), t_mesh[0].numpy(), pred[0].numpy(), c='black', linestyle='--', label='t = 0')
    plt.plot(x_mesh[25].numpy(), t_mesh[25].numpy(), pred[25].numpy(), c='black', label='t = 0.5')
    plt.legend()
    plt.xlabel("Space")
    plt.ylabel("$\;\;$Time")
    ax.view_init(15, 320)
    ax.axes.set_zlim3d(bottom=0.5, top=2.5)
    plt.savefig(os.path.join("Outputs", "hybrid_modeling_nn_"+str(n)+".pdf"), bbox_inches="tight")

    #Plot u_star
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.plot_surface(x_mesh.numpy(), t_mesh.numpy(), u_star(tx).reshape(-1, steps).detach().numpy(), cmap=cmap, linewidth=0, alpha=0.6)
    plt.plot(x_mesh[0].numpy(), t_mesh[0].numpy(), u_star(tx).reshape(-1, steps).detach()[0].numpy(), c='black', linestyle='--', label='t = 0')
    plt.plot(x_mesh[25].numpy(), t_mesh[25].numpy(), u_star(tx).reshape(-1, steps).detach()[25].numpy(), c='black', label='t = 0.5')
    plt.legend()
    plt.xlabel("Space")
    plt.ylabel("$\;\;$Time")
    ax.view_init(15, 320)
    ax.axes.set_zlim3d(bottom=0.5, top=2.5)
    plt.savefig(os.path.join("Outputs", "u_star.pdf"), bbox_inches="tight")

    #Plot the model
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.tick_params(axis='both', which='major', labelsize=12)
    initial_model = torch.exp(tx[:, 0] - tx[:, 1]).reshape(-1, steps).detach()
    ax.plot_surface(x_mesh.numpy(), t_mesh.numpy(), initial_model.numpy(), cmap=cmap, linewidth=0, alpha=0.6)
    plt.plot(x_mesh[0].numpy(), t_mesh[0].numpy(), initial_model[0].numpy(), c='black', linestyle='--', label='t = 0')
    plt.plot(x_mesh[25].numpy(), t_mesh[25].numpy(), initial_model[25].numpy(), c='black', label='t = 0.5')
    plt.legend()
    plt.xlabel("Space")
    plt.ylabel("$\;\;$Time")
    ax.view_init(15, 320)
    ax.axes.set_zlim3d(bottom=0.5, top=2.5)
    plt.savefig(os.path.join("Outputs", "model.pdf"), bbox_inches="tight")
    return


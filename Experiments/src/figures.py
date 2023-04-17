import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

matplotlib.rcParams.update({'font.size': 16})

def figure_monitoring(train_loss, val_loss, overfitting_gap_list, n, n_e, n_r, D):
    plt.figure()
    plt.plot(np.log(np.array(train_loss)), label="$\ln(R_{"+str(n)+","+str(n_e)+","+str(n_r)+"}^{(\mathrm{reg})})$")
    plt.plot(np.log(np.array(val_loss)), label="$\ln(\int_\Omega \|u^\\star - u_{\\hat \\theta(p, "+str(n_e)+"," +str(n_r)+"," + str(D)+")}\|_2^2)$")
    plt.plot(np.log(np.array(np.abs(overfitting_gap_list))), label="ln(|overfitting gap|)")
    plt.legend()
    plt.xlabel("Epoch p")
    plt.savefig(os.path.join("Outputs_PINNs", "perf_" + str(n) + ".pdf"), bbox_inches="tight")


def linear_reg():
    results = pd.read_csv(os.path.join("Outputs_PINNs", "validation_loss.csv"))
    log_val_loss = results[["log(Validation_loss)"]].to_numpy().reshape(-1)
    log_n = results[["log(n)"]].to_numpy().reshape(-1)

    coefficients = np.polyfit(log_n, log_val_loss, 1)
    m = coefficients[0]
    b = coefficients[1]
    print("Slope = " + str(m))

    # Generate regression line points
    regression_n = np.linspace(min(log_n), max(log_n), 100)
    regression_loss = m * regression_n + b

    # Plot the data points and regression line
    plt.figure()
    plt.scatter(log_n, log_val_loss, color="blue", label="Data points")
    plt.plot(regression_n, regression_loss, color='red',
             label='Linear regression: \n y = ' + str(round(m, 2)) + 'x ' + str(round(b, 2)))
    plt.xlabel('ln(n)')
    plt.ylabel("$L^2$ error")
    plt.legend()
    plt.savefig(os.path.join("Outputs_PINNs", "linear_regression.pdf"), bbox_inches="tight")
    return

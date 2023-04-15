import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

if __name__ == "__main__":
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
    plt.scatter(log_n, log_val_loss, color="blue", label = "Data points")
    plt.plot(regression_n, regression_loss, color='red', label='Linear regression: \n y = '+ str(round(m, 2))+'x '+ str(round(b, 2)))
    plt.xlabel('ln(n)', fontsize=18)
    plt.ylabel('ln(val L2 loss)', fontsize=18)
    plt.legend(fontsize=18)
    plt.savefig(os.path.join("Outputs_PINNs", "linear_regression.pdf"), bbox_inches="tight")

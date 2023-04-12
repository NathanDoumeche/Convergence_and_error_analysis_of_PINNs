import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(5)

if  __name__ == "__main__":
    # Random samples on [0 T] with T=1
    n, nr = 10, 30
    X_i = np.random.rand(n)
    Y_i = np.exp(-X_i) + 0.1*np.random.normal(size=n)
    Xr_i = np.random.rand(nr)

    #  Computing delta
    X_all = np.sort(np.concatenate((X_i, Xr_i),axis=0))
    delta = np.min(X_all[1:]-X_all[:-1])

    # Sorting input/output
    ind_sorted = np.argsort(X_i)
    X_i_sorted, Y_i_sorted = X_i[ind_sorted], Y_i[ind_sorted]
    X_i_sorted_lag1 = X_i_sorted[:-1] # w/out the last point for the sum in myfun and myfun_inf
    Y_diff = Y_i_sorted[1:]-Y_i_sorted[:-1]

    def overfitting_function(x):
        return Y_i_sorted[0] + .5 * (np.sign(x[:, np.newaxis] - X_i_sorted_lag1 - delta / 2) + 1) @ Y_diff

    #plots
    fig, ax = plt.subplots(figsize=(10,5),linewidth=8)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.scatter(X_i_sorted,
               Y_i_sorted,
               color='dodgerblue',
               marker='o',
               s=100,
               label="Observations $(X_i,Y_i)$")
    ax.scatter(Xr_i,
               overfitting_function(Xr_i),
               color='lightsalmon',
               marker='x',
               s=100,
               linewidth=2,
               label="Random samples $X^{(r)}_j$" )
    t_for_plot = np.linspace(0,1,1000)
    ax.plot(t_for_plot,
            overfitting_function(t_for_plot),
            color='red',
            linewidth=2,
            alpha=0.5,
            label="$u_{\hat{\\theta}(\infty,"+ str(nr)+ ", "+ str(n-1)+ ")}$")
    ax.plot(t_for_plot,
            np.exp(-t_for_plot),
            color='dodgerblue',
            linewidth=2,
            alpha=0.5,
            label="$u^{\\star}$",
            linestyle='dashed')
    ax.legend(fontsize=18)
    plt.savefig(os.path.join("Outputs", "prop3-1.pdf"), bbox_inches="tight")
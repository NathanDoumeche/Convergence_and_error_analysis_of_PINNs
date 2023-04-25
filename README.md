# Convergence and error analysis of Physics-informed neural networks
This project aims at illustrating the results of the paper _Convergence and error analysis of PINNs_ 
(Nathan Doumèche, Gérard Biau, and Claire Boyer). In particular, the Sobolev regularization is implemented in
Pytorch in the file _Section_5/src/regularized_neural_network.py_.

## Install

To install the packages used to generate the figures in the paper, create a virtual environment using Python 3.9.16. 
Then, run the following commands in the terminal.

    pip3 install -r requirements.txt

## Figures of Section 3
To reproduce the figures of Section 3 _PINNs can overfit_, run the following commands in the terminal 
of your Python IDE.

    cd Section_3/
    python3 prop3-1.py
    python3 prop3-2.py

Figures 1 and 2 can then be found in the folder _Section_3/Outputs/_ at _fig_1.pdf_ and _fig_2.pdf_.

## Figures of Section 5
To reproduce the figures of Section 5 _Strong convergence of PINNs for linear PDE systems_, run the following commands in the terminal 
of your Python IDE.

    cd Section_5/
    python3 Hybrid_modeling.py

Figure 3 can then be found at _Section_5/Outputs/training/perf_100.pdf_. \
Figure 4 is composed of the files _linear_regression.pdf_ and _log_PI.pdf_ in the folder _Section_5/Outputs/_.\
Figre 5 is composed of the files _hybrid_modeling_nn_1000.pdf_, _model.pdf_ and _u_star.pdf_ in the folder
_Section_5/Outputs/_.
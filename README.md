# Discriminative reconstruction via simultaneous dense and sparse coding
Official code repository for the simulation experiments of "[Discriminative reconstruction via simultaneous dense and sparse coding](https://openreview.net/forum?id=FkgM06HEbk)", a journal paper published at [Transactions on Machine Learning Research](https://jmlr.org/tmlr/) in 2024.

The repository is organized as follows:
- `phase_transition.py` and `phase_transition_figs.py`: the first file generates the data for the phase transition curves and the second one creates the plots (Figure 3).
- `snr_exp.py` and `snr_exp_figs.py`: the first file generates the data for the recovery simulations and the second one creates the plots (Figure 4).
- `MNIST_decomp.py`: generates the decomposition of an MNIST digit under our framework (Figure 5).
- `MNIST_distr.py`: generates the distribution of total variation and Euclidean norms for the sparse and dense components (Figures 6 and 13).

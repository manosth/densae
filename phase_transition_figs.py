# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib import rc
import pickle

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

np.random.seed(42)

# Core params
p = 0.5

# Secondary params
lin_size = 100
num_ticks = 6
start_point = 0.1
end_point = 0.9

# Load data
with open("errors_p=05.pickle", "rb") as f:
    errors_x, error_u, errors_both = pickle.load(f)

# ###### X FIGURES ########
plt.figure(figsize=(4.2, 2.4))
sns.heatmap(
    errors_x, cmap=cm.magma,
)
plt.xticks(
    np.linspace(0, lin_size, num_ticks),
    np.round(np.linspace(start_point, end_point, num_ticks), 2),
    fontsize=8,
)
plt.yticks(
    np.linspace(0, lin_size, num_ticks),
    np.round(np.linspace(end_point, start_point, num_ticks), 2),
    fontsize=8,
)
plt.xlabel("$\sigma = \\frac{m}{n + p}$")
plt.ylabel("$\\rho = \\frac{s}{m}$")
plt.title("$p = " + str(p) + "m$")
plt.tight_layout()
plt.savefig("figs/heatmap_x_p=" + str(p) + "_magma.pdf")
plt.show()
plt.close()

# ###### U FIGURES ########
plt.figure(figsize=(3.2, 2.4))
sns.heatmap(
    errors_u, cmap=cm.magma,
)
plt.xticks(
    np.linspace(0, lin_size, num_ticks),
    np.round(np.linspace(start_point, end_point, num_ticks), 2),
    fontsize=8,
)
plt.yticks(
    np.linspace(0, lin_size, num_ticks),
    np.round(np.linspace(end_point, start_point, num_ticks), 2),
    fontsize=8,
)
plt.xlabel("$\sigma = \\frac{m}{n + p}$")
plt.ylabel("$\\rho = \\frac{s}{m}$")
plt.title("$p = " + str(p) + "m$")
plt.tight_layout()
plt.savefig("figs/heatmap_u_p=" + str(p) + "_magma.pdf")
plt.show()
plt.close()

# ###### BOTH FIGURES ########
plt.figure(figsize=(3.2, 2.4))
sns.heatmap(
    errors_both, cmap=cm.magma,
)
plt.xticks(
    np.linspace(5, lin_size - 5, num_ticks),
    np.round(np.linspace(start_point, end_point, num_ticks), 2),
    fontsize=8,
)
plt.yticks(
    np.linspace(0, lin_size, num_ticks),
    np.round(np.linspace(end_point + 0.05, start_point - 0.05, num_ticks), 2),
    fontsize=8,
)
plt.xlabel("$\sigma = \\frac{m}{n + p}$")
plt.ylabel("$\\rho = \\frac{s}{m}$")
plt.title("$p = " + str(p) + "m$")
plt.tight_layout()
plt.savefig("figs/heatmap_both_p=" + str(p) + "_colormap.pdf")
plt.show()
plt.close()

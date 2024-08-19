# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")
np.random.seed(42)

# Core params
p = 0.1
sigma = 0.5
rho = 0.05

# Secondary params
lin_size = 200
num_ticks = 10
start_point = -40
end_point = 40

# Load data
with open("results/rho=005/errors_snr_s=05_p=01.pickle", "rb") as f:
    errors_u_ours, errors_u_cs = pickle.load(f)

# ###### X FIGURES ########
snrs = np.linspace(start_point, end_point, lin_size)

plt.figure(figsize=(3.2, 2.4))
plt.plot(snrs, errors_u_ours, "k", label="Ours")
plt.plot(snrs, errors_u_cs, "r", label="CS")
plt.xlim([snrs[0], snrs[-1]])
plt.xlabel("SNR (dB)")
plt.ylabel(
    r"Normalized error $\frac{\lVert\hat{\mathbf{u}} - \mathbf{u}^*\rVert_2}{\lVert\mathbf{u}^*\rVert_2}$"
)
plt.title(
    r"$\sigma = " + str(sigma) + ", \\rho = " + str(rho) + ", p = " + str(p) + "m$"
)
plt.tight_layout()
plt.legend()
plt.savefig("figs/noisy/snr.pdf")
plt.show()
plt.close()

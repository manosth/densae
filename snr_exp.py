# Import packages.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm
import pickle

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

np.random.seed(42)

# Core params
N = 100
p = 0.5

sigma = 0.95
M = max(int(sigma / (1 - sigma * p) * N), 1)
L = max(int(p * M), 1)

rho = 0.5
S = max(min(int(rho * M), N), 1)
epsilon = 0.001

# Secondary params
iter = 100
lin_size = 200
num_ticks = 10
start_point = -40
end_point = 40

# initiliaze matrices
snrs = np.linspace(start_point, end_point, lin_size)

errors_u_ours = np.zeros((lin_size,))
errors_u_cs = np.zeros((lin_size,))

for jdx in tqdm(range(snrs.shape[0])):
    snr = snrs[jdx]

    # based on the snr formula
    relative_power = 10 ** (snr / 20)

    error_sum_u_ours = 0
    error_sum_u_cs = 0
    for idx in range(iter):
        # generate matrices and vectors
        A = 1 / np.sqrt(M) * np.random.randn(M, L)
        B = 1 / np.sqrt(M) * np.random.randn(M, N)

        gamma = np.random.randn(M)
        x = np.dot(A.T, gamma)
        x /= np.linalg.norm(x)

        u = np.zeros((N,))
        indexes = np.random.choice(N, S, replace=False)
        u[indexes] = np.random.randn(S)
        u /= np.linalg.norm(u)
        u *= relative_power

        # generate measurement vector
        y = np.dot(A, x) + np.dot(B, u)

        # solve optimization - ours
        x_s = cp.Variable(L)
        u_s = cp.Variable(N)

        const = [A @ x_s + B @ u_s == y]
        obj = cp.Minimize(cp.norm(A @ x_s, 2) + cp.norm(u_s, 1))

        prob = cp.Problem(obj, const)
        prob.solve(solver=cp.SCS)

        # solve optimization - cs
        x_v = cp.Variable(L)
        u_v = cp.Variable(N)

        # since norm of x (noise) is normalized
        const = [cp.norm(y - B @ u_v, 2) <= np.linalg.norm(np.dot(A, x), 2)]
        obj = cp.Minimize(cp.norm(u_v, 1))

        prob = cp.Problem(obj, const)
        prob.solve(solver=cp.SCS)

        # check if the vectors match
        error_sum_u_ours += np.linalg.norm(u - u_s.value, 2) / np.linalg.norm(u, 2)
        error_sum_u_cs += np.linalg.norm(u - u_v.value, 2) / np.linalg.norm(u, 2)
    errors_u_ours[jdx] = error_sum_u_ours / iter
    errors_u_cs[jdx] = error_sum_u_cs / iter

# Save data
with open("errors_snr.pickle", "wb") as f:
    pickle.dump([errors_u_ours, errors_u_cs], f)

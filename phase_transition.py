# Import packages.
import cvxpy as cp
import numpy as np
import pickle

np.random.seed(42)

# Core params
N = 100
p = 0.5
epsilon = 0.001

# Secondary params
iter = 100
lin_size = 100
start_point = 0.05
end_point = 0.95

# initiliaze matrices
sigmas = np.linspace(start_point, end_point, lin_size)

errors_x = np.zeros((lin_size, lin_size))
errors_u = np.zeros((lin_size, lin_size))
errors_both = np.zeros((lin_size, lin_size))

for jdx in range(sigmas.shape[0]):
    sigma = sigmas[jdx]

    M = max(int(sigma / (1 - sigma * p) * N), 1)
    L = max(int(p * M), 1)

    rhos = np.linspace(start_point, end_point, lin_size)
    for kdx in range(rhos.shape[0]):
        rho = rhos[kdx]

        S = max(min(int(rho * M), N), 1)

        error_sum_x = 0
        error_sum_u = 0
        error_sum_both = 0
        for idx in range(iter):
            # print progress
            prog = ((jdx + 1) * len(rhos) * iter + (kdx + 1) * iter + (idx + 1)) / (
                (1 + (1 + len(sigmas)) * len(rhos)) * iter
            )
            print(
                "\rprogress: {prog:>5}%".format(prog=round(100 * prog, 3)), end="",
            )

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

            # generate measurement vector
            y = np.dot(A, x) + np.dot(B, u)

            # solve optimization
            x_s = cp.Variable(L)
            u_s = cp.Variable(N)

            const = [A @ x_s + B @ u_s == y]
            obj = cp.Minimize(cp.norm(A @ x_s, 2) + cp.norm(u_s, 1))

            prob = cp.Problem(obj, const)
            prob.solve(solver=cp.SCS)

            # check if the vectors match
            error_x = np.linalg.norm(x - x_s.value, 2) / np.linalg.norm(x, 2)
            if error_x < epsilon:
                error_sum_x += 1

            error_u = np.linalg.norm(u - u_s.value, 2) / np.linalg.norm(u, 2)
            if error_u < epsilon:
                error_sum_u += 1
            if error_u < epsilon and error_x < epsilon:
                error_sum_both += 1
        errors_x[lin_size - kdx - 1][jdx] = error_sum_x / iter
        errors_u[lin_size - kdx - 1][jdx] = error_sum_u / iter
        errors_both[lin_size - kdx - 1][jdx] = error_sum_both / iter

# Save data
with open("results/errors.pickle", "wb") as f:
    pickle.dump([errors_x, error_u, errors_both], f)

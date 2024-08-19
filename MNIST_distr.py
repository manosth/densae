# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as ds

# Import packages.
import cvxpy as cp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{bm}"

from scipy.linalg import qr

import time

def report_statistics(start, idx, total_len, val=0.0):
    current = time.time()
    total = current - start
    seconds = int(total % 60)
    minutes = int((total // 60) % 60)
    hours = int((total // 60) // 60)

    if idx == -1:
        print("")
        print(f"Total time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
    else:
        remain = (total_len - idx - 1) / (idx + 1) * total
        seconds_r = int(remain % 60)
        minutes_r = int((remain // 60) % 60)
        hours_r = int((remain // 60) // 60)
        print(f"progress: {(idx + 1) / total_len * 100:5.2f}%\telapsed: {hours:02d}:{minutes:02d}:{seconds:02d}\tremaining: {hours_r:02d}:{minutes_r:02d}:{seconds_r:02d}\tval: {val}", end="\r")

def load_mnist(datadir="~/data"):
    train_ds = ds.MNIST(root=datadir, train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    test_ds = ds.MNIST(root=datadir, train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    def to_xy(dataset):
        Y = dataset.targets.long()
        # this size is necessary to work with the matmul broadcasting when using channels
        X = dataset.data.view(dataset.data.shape[0], 1, -1) / 255.0
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    mean_tr = X_tr.mean(dim=0)
    mean_te = X_te.mean(dim=0)
    X_tr -= mean_tr
    X_te -= mean_te
    return X_tr, Y_tr, X_te, Y_te

def make_loader(dataset, shuffle=True, batch_size=128, num_workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)

# np.random.seed(42)
if __name__ == '__main__':
    # Core params
    N = 1024
    M = 784
    L = 1024

    epsilon = 0.001

    D, _ = qr(1 / np.sqrt(M) * np.random.randn(M, M))

    A = np.zeros((M, L))
    B = np.zeros((M, N))

    A[:, :int(M / 2)] = D[:, :int(M / 2)]
    B[:, :int(M / 2)] = D[:, int(M / 2):]
    for idx in range(L - int(M / 2)):
        lin_comb = 1 / np.sqrt(M) * np.random.randn(int(M / 2))
        A[:, int(M / 2) + idx] = np.dot(D[:, :int(M / 2)], lin_comb)

    for idx in range(N - int(M / 2)):
        lin_comb = 1 / np.sqrt(M) * np.random.randn(int(M / 2))
        B[:, int(M / 2) + idx] = np.dot(D[:, int(M / 2):], lin_comb)


    A = np.random.permutation(A.T).T
    B = np.random.permutation(B.T).T
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    workers = max(4 * torch.cuda.device_count(), 4)

    X_tr, Y_tr, _, _ = load_mnist()
    train_dl = make_loader(TensorDataset(X_tr, Y_tr), batch_size=1, num_workers=workers)

    norm_us = []
    norm_xs = []
    tv_us = []
    tv_xs = []
    end = 100
    start = time.time()
    for idx, (y, _) in enumerate(train_dl):
        y = y.squeeze().numpy()

        # solve optimization
        x_s = cp.Variable(L)
        u_s = cp.Variable(N)

        mu = 3
        lam = 0.01
        const = []
        obj = cp.Minimize(cp.norm(A @ x_s + B @ u_s - y, 2) ** 2 + mu * cp.norm(A @ x_s, 2) + lam * cp.norm(u_s, 1))

        prob = cp.Problem(obj, const)
        prob.solve(solver=cp.SCS)

        norm_u = np.linalg.norm(np.dot(B, u_s.value))
        norm_x = np.linalg.norm(np.dot(A, x_s.value))
        norm_us.append(norm_u)
        norm_xs.append(norm_x)
        tv_u = np.sum(np.abs(np.diff(np.dot(B, u_s.value))))
        tv_x = np.sum(np.abs(np.diff(np.dot(A, x_s.value))))
        tv_us.append(tv_u)
        tv_xs.append(tv_x)
        report_statistics(start, idx, end, val="")
        if idx == end:
            break

    plt.figure(figsize=[9.6, 4.8])
    sns.histplot(norm_us, label=r"$\bm{Bu}$")
    sns.histplot(norm_xs, label=r"$\bm{Ax}$")
    plt.xlabel("Norm")
    plt.legend()
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig("figs/distribution_norm.pdf")
    plt.show()

    plt.figure(figsize=[9.6, 4.8])
    sns.histplot(tv_us, label=r"$\bm{Bu}$")
    sns.histplot(tv_xs, label=r"$\bm{Ax}$")
    plt.xlabel("Total Variation")
    plt.legend()
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
    plt.savefig("figs/distribution_tv.pdf")
    plt.show()

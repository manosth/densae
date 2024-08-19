# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as ds

# Import packages
import cvxpy as cp
import numpy as np
import pickle
import matplotlib.pyplot as plt

from scipy.linalg import qr

def load_mnist(datadir="~/data"):
    train_ds = ds.MNIST(root=datadir, train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    test_ds = ds.MNIST(root=datadir, train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    def to_xy(dataset):
        Y = dataset.targets.long()
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

        plt.figure()
        plt.plot(np.arange(len(u_s.value)), u_s.value, "lightgreen", label="Ours")
        plt.title("Recovered u")
        plt.tight_layout()
        plt.legend()
        plt.savefig("figs/u_recov.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(x_s.value)), x_s.value, "lightgreen", label="Ours")
        plt.title("Recovered x")
        plt.tight_layout()
        plt.legend()
        plt.savefig("figs/x_recov.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(y)), np.dot(A, x_s.value), "lightgreen", label="Ours")
        plt.title("Recovered Ax")
        plt.tight_layout()
        plt.legend()
        plt.savefig("figs/Ax_recov.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(y)), np.dot(A, x_s.value) + np.dot(B, u_s.value), "lightgreen", label="Ours")
        plt.plot(np.arange(len(y)), y, "--k", label="Orig")
        plt.title("Measured signal y")
        plt.tight_layout()
        plt.legend()
        plt.savefig("figs/y_sig.pdf")
        plt.show()
        plt.close()


        plt.figure()
        plt.imshow(y.reshape(28, 28))
        plt.axis("off")
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        plt.savefig("figs/y_img.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow((np.dot(B, u_s.value) + np.dot(A, x_s.value)).reshape(28, 28))
        plt.axis("off")
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        plt.savefig("figs/ax_bu_img.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow(u_s.value.reshape(32, 32), cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        plt.savefig("figs/u_img.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow(np.dot(B, u_s.value).reshape(28, 28))
        plt.axis("off")
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        plt.savefig("figs/Bu_img.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow(x_s.value.reshape(32, 32), cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        plt.savefig("figs/x_img.pdf")
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow(np.dot(A, x_s.value).reshape(28, 28))
        plt.axis("off")
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0)
        plt.savefig("figs/Ax_img.pdf")
        plt.show()
        plt.close()

        tv_u = np.sum(np.abs(np.diff(np.dot(B, u_s.value))))
        tv_x = np.sum(np.abs(np.diff(np.dot(A, x_s.value))))
        print(f"total variation for u: {tv_u}")
        print(f"total variation for x: {tv_x}")

        norm_u = np.linalg.norm(np.dot(B, u_s.value))
        norm_x = np.linalg.norm(np.dot(A, x_s.value))
        print(f"norm for Bu: {norm_u}")
        print(f"norm for Ax: {norm_x}")
        break

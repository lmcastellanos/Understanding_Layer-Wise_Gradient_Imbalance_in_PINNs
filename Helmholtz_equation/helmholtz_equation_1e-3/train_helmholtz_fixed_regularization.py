#!/usr/bin/env python3

"""
1D Helmholtz training with a fixed PDE-regularization constant.

Equation:
    u_xx + k^2 u = f(x)

Domain:
    x in [0,1]

Exact solution used to generate the dataset:
    u(x) = sin(pi x) + 0.5 sin(4 pi x)

Boundary conditions:
    u(0) = u(1) = 0

Loss:
    L = L_data + beta * L_PDE
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SEED = 42
EPOCHS = 4000
LEARNING_RATE = 1e-3

BETA = 1e-3
K = 20.0

TRAIN_FRACTION = 0.80
N_COLLOCATION = 1000

DATASET_FILE = "helmholtz_dataset.csv"
OUTPUT_FOLDER = "helmholtz_fixed_regularization_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class HelmholtzNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


def exact_source(x):
    """
    f(x) corresponding to:
        u = sin(pi*x) + 0.5*sin(4*pi*x)
    """

    pi = torch.pi

    u_exact = (
        torch.sin(pi * x)
        + 0.5 * torch.sin(4.0 * pi * x)
    )

    u_xx_exact = (
        -(pi ** 2) * torch.sin(pi * x)
        - 0.5 * (4.0 * pi) ** 2 * torch.sin(4.0 * pi * x)
    )

    return u_xx_exact + (K ** 2) * u_exact


def helmholtz_residual(model, x):
    x = x.clone().detach().requires_grad_(True)

    u = model(x)

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    f = exact_source(x)

    return u_xx + (K ** 2) * u - f


def load_data(script_dir):
    df = pd.read_csv(script_dir / DATASET_FILE)

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    split = int(TRAIN_FRACTION * len(indices))

    train_df = df.iloc[indices[:split]].reset_index(drop=True)
    test_df = df.iloc[indices[split:]].reset_index(drop=True)

    def tensors(frame):
        x = torch.tensor(
            frame[["x"]].to_numpy(),
            dtype=torch.float32,
            device=DEVICE,
        )

        u = torch.tensor(
            frame[["u"]].to_numpy(),
            dtype=torch.float32,
            device=DEVICE,
        )

        return x, u

    return (
        tensors(train_df),
        tensors(test_df),
        train_df,
        test_df,
    )


def sample_collocation(n):
    return torch.rand(
        n,
        1,
        device=DEVICE,
    )


def boundary_loss(model):
    x_bc = torch.tensor(
        [[0.0], [1.0]],
        dtype=torch.float32,
        device=DEVICE,
    )

    u_bc = model(x_bc)

    return torch.mean(u_bc ** 2)


def main():
    script_dir = Path(__file__).resolve().parent

    output_dir = (
        script_dir
        / OUTPUT_FOLDER
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (x_train, u_train), (
        x_test,
        u_test,
    ), _, test_df = load_data(script_dir)

    model = HelmholtzNet().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(x_train)

        data_loss = torch.mean(
            (prediction - u_train) ** 2
        )

        x_f = sample_collocation(
            N_COLLOCATION
        )

        residual = helmholtz_residual(
            model,
            x_f,
        )

        pde_loss = torch.mean(
            residual ** 2
        )

        bc_loss = boundary_loss(
            model
        )

        regularization_loss = (
            pde_loss
            + bc_loss
        )

        total_loss = (
            data_loss
            + BETA * regularization_loss
        )

        total_loss.backward()

        optimizer.step()

        history.append({
            "epoch": epoch,
            "data_loss": data_loss.item(),
            "pde_loss": pde_loss.item(),
            "bc_loss": bc_loss.item(),
            "regularization_loss": regularization_loss.item(),
            "total_loss": total_loss.item(),
        })

        if epoch == 1 or epoch % 250 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Data={data_loss.item():.4e} | "
                f"PDE={pde_loss.item():.4e} | "
                f"BC={bc_loss.item():.4e} | "
                f"Total={total_loss.item():.4e}"
            )

    model.eval()

    with torch.no_grad():

        test_pred = model(
            x_test
        )

    test_mse = torch.mean(
        (test_pred - u_test) ** 2
    ).item()

    test_rmse = float(
        np.sqrt(test_mse)
    )

    output = test_df.copy()

    output["u_pred"] = (
        test_pred.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    output["absolute_error"] = np.abs(
        output["u_pred"]
        - output["u"]
    )

    pd.DataFrame(
        history
    ).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    output.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame([{
        "beta": BETA,
        "test_MSE": test_mse,
        "test_RMSE": test_rmse,
    }]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    torch.save(
        model.state_dict(),
        output_dir
        / "helmholtz_fixed_regularization_model.pth",
    )

    print(
        f"\nTest RMSE: {test_rmse:.6e}"
    )

    print(
        f"Results saved in: {output_dir}"
    )


if __name__ == "__main__":
    main()

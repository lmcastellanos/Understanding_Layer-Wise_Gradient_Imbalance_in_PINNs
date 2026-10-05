#!/usr/bin/env python3
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
EPOCHS = 3000
LEARNING_RATE = 1e-3
NU = 0.01 / np.pi
TRAIN_FRACTION = 0.80
N_COLLOCATION = 1000
EPS = 1e-12
DATASET_FILE = "burgers_small_dataset.csv"
OUTPUT_FOLDER = "burgers_gradient_normalization_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class BurgersNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xt):
        return self.net(xt)


def burgers_residual(model, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(torch.cat([x, t], dim=1))

    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]

    return u_t + u * u_x - NU * u_xx


def load_data(script_dir):
    df = pd.read_csv(script_dir / DATASET_FILE)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    split = int(TRAIN_FRACTION * len(idx))
    train_df = df.iloc[idx[:split]].reset_index(drop=True)
    test_df = df.iloc[idx[split:]].reset_index(drop=True)

    def tensors(frame):
        x = torch.tensor(frame[["x"]].to_numpy(), dtype=torch.float32, device=DEVICE)
        t = torch.tensor(frame[["t"]].to_numpy(), dtype=torch.float32, device=DEVICE)
        u = torch.tensor(frame[["u"]].to_numpy(), dtype=torch.float32, device=DEVICE)
        return x, t, u

    return tensors(train_df), tensors(test_df), train_df, test_df


def sample_collocation(n):
    x = -1.0 + 2.0 * torch.rand(n, 1, device=DEVICE)
    t = torch.rand(n, 1, device=DEVICE)
    return x, t


def linear_layer_groups(model):
    groups = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            groups.append([module.weight, module.bias])
    return groups


def zero_if_none(grad, parameter):
    return torch.zeros_like(parameter) if grad is None else grad


def group_norm(grads):
    return torch.sqrt(sum(torch.sum(g ** 2) for g in grads))


def main():
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    (x_train, t_train, u_train), (x_test, t_test, u_test), _, test_df = load_data(script_dir)

    model = BurgersNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    parameters = list(model.parameters())
    layer_groups = linear_layer_groups(model)

    history = []
    scale_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        pred = model(torch.cat([x_train, t_train], dim=1))
        mse_loss = torch.mean((pred - u_train) ** 2)

        x_f, t_f = sample_collocation(N_COLLOCATION)
        residual = burgers_residual(model, x_f, t_f)
        pde_loss = torch.mean(residual ** 2)

        mse_grads = torch.autograd.grad(
            mse_loss, parameters, retain_graph=True, allow_unused=True
        )

        pde_grads = torch.autograd.grad(
            pde_loss, parameters, retain_graph=False, allow_unused=True
        )

        mse_map = {
            id(p): zero_if_none(g, p)
            for p, g in zip(parameters, mse_grads)
        }

        pde_map = {
            id(p): zero_if_none(g, p)
            for p, g in zip(parameters, pde_grads)
        }

        optimizer.zero_grad(set_to_none=True)
        epoch_scales = []

        for layer_index, layer_params in enumerate(layer_groups, start=1):
            g_mse = [mse_map[id(p)] for p in layer_params]
            g_pde = [pde_map[id(p)] for p in layer_params]

            mse_norm = group_norm(g_mse)
            pde_norm = group_norm(g_pde)

            scale = torch.minimum(
                torch.tensor(1.0, device=DEVICE),
                mse_norm / (pde_norm + EPS),
            )

            for p, gm, gp in zip(layer_params, g_mse, g_pde):
                p.grad = gm + scale * gp

            epoch_scales.append(scale.item())

            scale_history.append({
                "epoch": epoch,
                "layer": layer_index,
                "mse_grad_norm": mse_norm.item(),
                "pde_grad_norm": pde_norm.item(),
                "scale": scale.item(),
            })

        optimizer.step()

        history.append({
            "epoch": epoch,
            "mse_loss": mse_loss.item(),
            "pde_loss": pde_loss.item(),
            "mean_layer_scale": float(np.mean(epoch_scales)),
        })

        if epoch == 1 or epoch % 250 == 0:
            print(
                f"Epoch {epoch:5d} | MSE={mse_loss.item():.4e} | "
                f"PDE={pde_loss.item():.4e} | mean scale={np.mean(epoch_scales):.4e}"
            )

    model.eval()
    with torch.no_grad():
        test_pred = model(torch.cat([x_test, t_test], dim=1))

    test_mse = torch.mean((test_pred - u_test) ** 2).item()
    test_rmse = float(np.sqrt(test_mse))

    pred_df = test_df.copy()
    pred_df["u_pred"] = test_pred.detach().cpu().numpy().reshape(-1)
    pred_df["absolute_error"] = np.abs(pred_df["u_pred"] - pred_df["u"])

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(scale_history).to_csv(output_dir / "layer_gradient_scales.csv", index=False)
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame([{
        "test_MSE": test_mse,
        "test_RMSE": test_rmse,
    }]).to_csv(output_dir / "metrics.csv", index=False)

    torch.save(model.state_dict(), output_dir / "burgers_gradient_normalization_model.pth")

    print(f"\nTest RMSE: {test_rmse:.6e}")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()

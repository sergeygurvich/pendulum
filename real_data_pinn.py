# mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts

#!/usr/bin/env python
# coding: utf-8
from PIL import Image
from scipy.integrate import odeint
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # disabled while testing is off

# Set random seed for reproducibility
torch.manual_seed(123)

# mlflow setup
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("pinn_experiments_global")

# Device configuration
device='cpu'
# device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
patience = 200  # Early stopping patience
end_time = 5.0  # Simulation end time
n_epochs = 100000 # Maximum number of epochs

# Physical constants
g = 9.81  # Gravity acceleration (m/s^2)

# Physics loss weights (base values)
BASE_PHYS_W = 1e-4
BASE_IC_W = 1e-3
# Annealing parameters: ramp physics weights from base to max over warmup epochs
PHYS_W_MAX = 1e-2
IC_W_MAX = 1e-2
WARMUP_EPOCHS = 5000


# read real data from CSV
data_df = pd.read_csv('data/clean_data/real_pendulum_data_5sec.csv')  # Ensure this CSV file

# Convert training data to tensors
x_tensor = torch.tensor(data_df[['t','length','start_angle']].values, dtype=torch.float32)
y_tensor = torch.tensor(data_df['theta'].values, dtype=torch.float32).view(-1,1)

# Small angle analytical solution for pendulum
# w: angular frequency, x: time
# Returns theta(t) for small angle approximation
# def pendulum_solution(w, x, initial_state):
#     theta0 = float(initial_state)
#     if isinstance(x, torch.Tensor):
#         return theta0 * torch.cos(torch.as_tensor(w, dtype=x.dtype, device=x.device) * x)
#     return theta0 * np.cos(w * x)

# ODE for simple pendulum
# state: [theta, theta_dot], t: time, L: length, g: gravity
# Returns derivatives [theta_dot, theta_ddot]
def simple_pendulum_eqn(state, t, L, g):
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return [theta_dot, theta_ddot]

# Solve pendulum ODE for a given length L
# Returns time array and angle array
def solve_length(L, initial_state):
    t = np.linspace(0, end_time, 2500)
    states = odeint(simple_pendulum_eqn, initial_state, t, args=(L, g))
    y = states[:,0]
    return t, y


# Fully Connected Neural Network (FCN) definition
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        # Input layer
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())
        # Hidden layers
        self.fch = nn.Sequential(*[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS - 1)])
        # Output layer
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

# Plot results for training and validation
# step: current training step
# x_time, y_true: true values for plotting
# x_train_time, y_train: training data
# y_pred: model predictions
def plot_result(step,
                # x_time,
                # y_true,
                x_train_time,
                y_train,
                y_pred
                ):
    def to_np(a): return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    plt.figure(figsize=(8,4))
    # plt.plot(to_np(x_time), to_np(y_true), color="tab:green", label="Exact (L=%.3f)"%L_plot)
    # plt.plot(to_np(x_time), to_np(pendulum_solution(w_plot, x_time)), '--', color='tab:grey', label='Small angle approx')
    plt.plot(to_np(x_train_time), to_np(y_pred), color="tab:blue", linewidth=2, label="PINN")
    plt.scatter(to_np(x_train_time), to_np(y_train), color='tab:orange', s=50, alpha=0.6, label='Training/Val data')
    plt.xlim(-0.05, end_time+0.05)
    plt.ylim(-1.1,1.1)
    plt.title(f"Training step {step}")
    plt.legend(frameon=False)
    plt.tight_layout()

# Save a sequence of PNGs as a GIF using PIL
def save_gif_PIL(outfile, files, fps=10, loop=0):
    imgs = [Image.open(f) for f in files]
    if not imgs: return
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

# Train the PINN model
# model_name: name for saving artifacts
# x_train, y_train: training data
# num_epoch: number of epochs
# pinn: whether to use physics-informed loss
def train_pinn(model_name, x_train, y_train, num_epoch, pinn=True):
    # Log parameters to MLflow
    mlflow.log_param("end_time", end_time)
    mlflow.log_param("patience", patience)
    mlflow.log_param("num_epochs_init", num_epoch)
    mlflow.log_param("device", device)
    model = FCN(3,1,32,3).to(device)
    lr = 1e-4 if pinn else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mlflow.log_param("learning_rate", lr)
    # Log physics weighting params
    mlflow.log_param("BASE_PHYS_W", BASE_PHYS_W)
    mlflow.log_param("BASE_IC_W", BASE_IC_W)
    mlflow.log_param("PHYS_W_MAX", PHYS_W_MAX)
    mlflow.log_param("IC_W_MAX", IC_W_MAX)
    mlflow.log_param("WARMUP_EPOCHS", WARMUP_EPOCHS)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    files = []
    best_loss = float('inf')
    counter = 0
    final_epoch = 0

    # Cache unique (length, start_angle) pairs for physics sampling
    pairs = torch.tensor(
        data_df[['length', 'start_angle']].values, dtype=torch.float32
    )
    pairs = pairs[~torch.isnan(pairs).any(dim=1)]
    pairs = torch.unique(pairs, dim=0).to(device)

    for i in range(num_epoch):
        model.train()
        optimizer.zero_grad()
        # Annealed weights
        if pinn:
            ramp = min(1.0, (i+1) / max(1, WARMUP_EPOCHS))
            phys_w = BASE_PHYS_W + ramp * (PHYS_W_MAX - BASE_PHYS_W)
            ic_w = BASE_IC_W + ramp * (IC_W_MAX - BASE_IC_W)
        else:
            phys_w = 0.0
            ic_w = 0.0
        y_pred = model(x_train)
        data_loss = torch.mean((y_pred - y_train)**2)

        physics_losses = []
        ic_losses = []
        for L_theta0 in pairs:
            L_phys = L_theta0[0:1]
            theta0 = L_theta0[1:2]
            t_phys = torch.linspace(0.0, end_time, steps=64, device=device).view(-1, 1)
            t_phys.requires_grad_(True)
            l_phys = L_phys.expand_as(t_phys)
            th0_phys = theta0.expand_as(t_phys)
            x_phys = torch.cat([t_phys, l_phys, th0_phys], dim=1)
            y_phys = model(x_phys)
            dy_dt = torch.autograd.grad(y_phys, t_phys, torch.ones_like(y_phys), create_graph=True)[0]
            d2y_dt2 = torch.autograd.grad(dy_dt, t_phys, torch.ones_like(dy_dt), create_graph=True)[0]
            k_phys = g / l_phys
            residual = d2y_dt2 + k_phys * torch.sin(y_phys)
            physics_losses.append(torch.mean(residual ** 2))
            t0 = torch.zeros(1, 1, device=device, requires_grad=True)
            x0 = torch.cat([t0, L_phys.view(1, 1), theta0.view(1, 1)], dim=1)
            y0 = model(x0)
            dy_dt0 = torch.autograd.grad(y0, t0, torch.ones_like(y0), create_graph=True)[0]
            ic_theta = torch.mean((y0 - theta0.view(1, 1)) ** 2)
            ic_dtheta = torch.mean(dy_dt0 ** 2)
            ic_losses.append(ic_theta + ic_dtheta)
        physics_loss = torch.mean(torch.stack(physics_losses)) if physics_losses else torch.tensor(0.0, device=device)
        ic_loss = torch.mean(torch.stack(ic_losses)) if ic_losses else torch.tensor(0.0, device=device)

        loss = data_loss + phys_w * physics_loss + ic_w * ic_loss

        # Log weights and components periodically
        if (i+1) % 500 == 0:
            mlflow.log_metric("physics_weight", float(phys_w), step=i+1)
            mlflow.log_metric("ic_weight", float(ic_w), step=i+1)
            mlflow.log_metric("data_loss", float(data_loss.item()), step=i+1)
            mlflow.log_metric("physics_loss", float(physics_loss.item()), step=i+1)
            mlflow.log_metric("ic_loss", float(ic_loss.item()), step=i+1)

        loss.backward()
        optimizer.step()

        # Optionally save plots every 500 epochs based on training data only
        if (i+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                y_train_pred = model(x_train)
            plot_result(i+1, x_train[:, 0], y_train, y_train_pred)
            os.makedirs('plots', exist_ok=True)
            file = f"plots/{model_name}_{i+1:08d}.png"
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close()
            files.append(file)
            mlflow.log_metric("train_loss", float(loss.item()), step=i+1)

        # Early stopping based on training loss
        current_loss = float(loss.item())
        # if current_loss < best_loss:
        #     best_loss = current_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping at epoch {i+1}")
        #         final_epoch = i+1
        #         break
        if (i+1) % 1000 == 0:
            print(f"Epoch {i+1} Loss {current_loss:.6f}")
        final_epoch = i+1
    mlflow.log_param("total_training_epochs", final_epoch)
    return model, files

# Wrapper to start MLflow run using only training data (no validation/test)
def start_training(model_name, x_all, y_all, num_epochs=1000, pinn=True):
    # Use entire dataset as training data
    x_train = x_all
    y_train = y_all

    mlflow.end_run()
    with mlflow.start_run(run_name=model_name):
        model, files = train_pinn(model_name, x_train, y_train, num_epochs, pinn)
        if files:
            save_gif_PIL(f"{model_name}.gif", files, fps=10, loop=0)
            mlflow.log_artifact(f"{model_name}.gif")
        torch.save(model.state_dict(), f"{model_name}.pth")
        mlflow.pytorch.log_model(model, "models")
    mlflow.end_run()

# Main entry point
if __name__ == "__main__":
    model_name = '7 data variations'
    start_training(model_name, x_tensor, y_tensor, num_epochs=n_epochs, pinn=True)

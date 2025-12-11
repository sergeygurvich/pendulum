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

# Set random seed for reproducibility
torch.manual_seed(123)

# mlflow setup
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("pinn_experiments_global")

# Device configuration
device='cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

patience = 200  # Early stopping patience
end_time = 5.0  # Simulation end time
n_epochs = 20000 # Maximum number of epochs
# Physical constants
g = 9.81  # Gravity
initial_state = [1.596, 0]  # Initial angle and angular velocity
initial_state_str = str(initial_state)
lengths = [1]  # List of pendulum lengths to simulate

# ODE for simple pendulum
# state: [theta, theta_dot], t: time, L: length, g: gravity
# Returns derivatives [theta_dot, theta_ddot]
def simple_pendulum_eqn(state, t, L, g):
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return [theta_dot, theta_ddot]

# Solve pendulum ODE for a given length L
# Returns time array and angle array
def solve_length(L):
    t = np.linspace(0, end_time, 2500)
    states = odeint(simple_pendulum_eqn, initial_state, t, args=(L, g))
    y = states[:,0]
    return t, y

# Prepare training data
frames = []
num_samples = 10  # Number of training samples per length
for L in lengths:
    t, y = solve_length(L)
    indices = np.linspace(0, len(t)-1, num_samples, dtype=int)
    x_sub = torch.tensor(t[indices], dtype=torch.float32).view(-1,1)
    y_sub = torch.tensor(y[indices], dtype=torch.float32).view(-1,1)
    frames.append(pd.DataFrame({'t': x_sub.view(-1).tolist(), 'theta': y_sub.view(-1).tolist(), 'length': L}))

# Merge all training samples into a single DataFrame
merged_df = pd.concat(frames).reset_index(drop=True)

# read real data from CSV
# merged_df = pd.read_csv('data/curated_data/l09_pi2_5s.csv')  # Ensure this CSV file
# merged_df = merged_df[:32]

# Convert training data to tensors
x_tensor = torch.tensor(merged_df[['t','length']].values, dtype=torch.float32)
y_tensor = torch.tensor(merged_df['theta'].values, dtype=torch.float32).view(-1,1)

# Prepare data for plotting
L_plot = lengths[-1]
plot_t, plot_y = solve_length(L_plot)
plot_t_tensor = torch.tensor(plot_t, dtype=torch.float32).view(-1,1)
plot_y_tensor = torch.tensor(plot_y, dtype=torch.float32).view(-1,1)
w_plot = np.sqrt(g / L_plot)

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
def plot_result(step, x_time, y_true, x_train_time, y_train, y_pred):
    def to_np(a): return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    plt.figure(figsize=(8,4))
    plt.plot(to_np(x_time), to_np(y_true), color="tab:green", label="Exact (L=%.3f)"%L_plot)
    # plt.plot(to_np(x_time), to_np(pendulum_solution(w_plot, x_time)), '--', color='tab:grey', label='Small angle approx')
    plt.plot(to_np(x_time), to_np(y_pred), color="tab:blue", linewidth=2, label="PINN")
    plt.scatter(to_np(x_train_time), to_np(y_train), color='tab:orange', s=50, alpha=0.6, label='Training data')
    plt.xlim(-0.05, end_time+0.05)
    plt.ylim(-1.6,1.6)
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
def train_pinn(model_name, x_train, y_train, num_epoch=1000, pinn=True):
    # Log parameters to MLflow
    mlflow.log_param("initial_state", initial_state_str)
    mlflow.log_param("lengths", lengths)
    mlflow.log_param("end_time", end_time)
    mlflow.log_param("patience", patience)
    mlflow.log_param("num_epochs_init", num_epoch)
    mlflow.log_param("device", device)
    model = FCN(2,1,32,3).to(device)
    lr = 1e-4 if pinn else 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mlflow.log_param("learning_rate", lr)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    files = []  # For saving plot images
    final_epoch = 0

    for i in range(num_epoch):
        optimizer.zero_grad()
        y_pred = model(x_train)
        data_loss = torch.mean((y_pred - y_train)**2)
        loss = data_loss
        if pinn:
            # Physics-informed loss calculation
            physics_losses = []
            for L_phys in lengths:
                t_phys = torch.linspace(0, end_time, 30, device=device).view(-1,1)
                l_phys = torch.full_like(t_phys, L_phys)
                x_phys = torch.cat([t_phys, l_phys], dim=1).requires_grad_(True)
                y_phys = model(x_phys)
                grad1 = torch.autograd.grad(y_phys, x_phys, torch.ones_like(y_phys), create_graph=True)[0]
                dy_dt = grad1[:,0:1]
                grad2 = torch.autograd.grad(dy_dt, x_phys, torch.ones_like(dy_dt), create_graph=True)[0]
                d2y_dt2 = grad2[:,0:1]
                k_phys = g / x_phys[:,1:2]
                residual = d2y_dt2 + k_phys * torch.sin(y_phys)
                physics_losses.append(torch.mean(residual**2))
            physics_loss = torch.mean(torch.stack(physics_losses))
            loss = loss + 1e-4 * physics_loss
        loss.backward()
        optimizer.step()
        # Save plots and log metrics every 500 epochs
        if (i+1) % 500 == 0:
            x_full_plot = torch.cat([plot_t_tensor.to(device), torch.full((plot_t_tensor.shape[0],1), L_plot, dtype=plot_t_tensor.dtype, device=device)], dim=1)
            y_full_pred = model(x_full_plot).detach().cpu()
            # Plot all training points, not just the first 10
            plot_result(
                i+1,
                plot_t_tensor,
                plot_y_tensor,
                torch.tensor(merged_df['t'].values).view(-1,1),
                torch.tensor(merged_df['theta'].values).view(-1,1),
                y_full_pred
            )
            os.makedirs('../plots', exist_ok=True)
            file = f"plots/{model_name}_{i+1:08d}.png"
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close()
            files.append(file)
            mlflow.log_metric("validation_loss", float(loss.item()), step=i+1)

        current_loss = float(loss.item())
        if (i+1) % 1000 == 0:
            print(f"Epoch {i+1} Loss {current_loss:.6f}")
        final_epoch = i+1
    mlflow.log_param("total_training_epochs", final_epoch)
    return model, files

# Wrapper to start MLflow run and train model
def start_training(model_name, x_train, y_train, num_epochs=1000, pinn=True):
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
    model_name = 'pinn_l_1met_no_data'
    start_training(model_name, x_tensor, y_tensor, num_epochs=n_epochs, pinn=True)

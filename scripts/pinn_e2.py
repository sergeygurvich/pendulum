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

MODEL_NAME = "PINN_Hybrid_2___0.1xData_0.1xPhy_0.1xIC__100k__lr0.001"
N_EPOCHS = 100000
# PHYSICS_LOSS_WEIGHT = 1e-4
DATA_LOSS_WEIGHT = 0.1
PHYSICS_LOSS_WEIGHT = 0.1
IC_LOSS_WEIGHT = 0.1
mlflow.set_experiment("experiment2_sparse_data")

# mlflow setup
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

# Device configuration
device='cpu'
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Training parameters
end_time = 5.0  # Simulation end time

# Physical constants
g = 9.81  # Gravity
L=0.19  # Pendulum length for simulation
t = np.linspace(0, end_time, 2500)
x = torch.tensor(t, dtype=torch.float32).view(-1, 1)

# ODE for simple pendulum
# state: [theta, theta_dot], t: time, L: length, g: gravity
# Returns derivatives [theta_dot, theta_ddot]
def simple_pendulum_eqn(state, t, L, g):
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return [theta_dot, theta_ddot]

# Solve pendulum ODE for a given length L
# Returns time array and angle array
def solve_length(initial_state):
    states = odeint(simple_pendulum_eqn, initial_state, t, args=(L, g))
    y = states[:,0]
    return t, y

# Prepare training data
frames = []
num_samples = 10  # Number of training samples per length

# read real data from CSV
data_df = pd.read_csv('../data/clean_data/real_pendulum_data_5sec.csv')
# leave only one start angle (~0.3)
data_df = data_df[data_df['length']==0.19]


# Split into train / test
start_angle_test =[0.786]
train_df = data_df[~data_df.start_angle.isin(start_angle_test)].sort_values('t')
test_df = data_df.drop(train_df.index).sort_values('t')
start_angles = data_df['start_angle'].unique().tolist()

# start_angles = data_df['start_angle'].unique().tolist()

# If test is empty (very small dataset), fall back to using full data as train
if test_df.shape[0] == 0:
    train_df = data_df.sort_values('t')
    test_df = data_df.sort_values('t')

# Convert training and test data to tensors
x_train_tensor = torch.tensor(train_df[['t','start_angle']].values, dtype=torch.float32)
y_train_tensor = torch.tensor(train_df['theta'].values, dtype=torch.float32).view(-1,1)

x_test_tensor = torch.tensor(test_df[['t','start_angle']].values, dtype=torch.float32)
y_test_tensor = torch.tensor(test_df['theta'].values, dtype=torch.float32).view(-1,1)

# train_df = train_df[:0]

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
                x_train_time,
                y_train,
                y_pred,
                _start_angles
                ):
    def to_np(a): return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    plt.figure(figsize=(8,4))
    plt.scatter(to_np(x_train_time), to_np(y_pred), color="tab:blue", linewidth=2)
    plt.scatter(to_np(x_train_time), to_np(y_train), color='tab:orange', s=50, alpha=0.6, label='Training data')
    for start_angle in _start_angles:
        states = odeint(simple_pendulum_eqn, [start_angle,0], t, args=(L, g))
        y = torch.tensor(states[:, 0], dtype=torch.float32).view(-1, 1)
        plt.plot(x,y, label='Exact solution', color='red')
    plt.xlim(-0.05, end_time+0.05)
    plt.ylim(-1.6,1.6)
    plt.title(f"Training step: {step}")
    plt.legend(frameon=False, loc='upper right')
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
def train(model_name, x_train, y_train, num_epoch):
    # Log parameters to MLflow
    mlflow.log_param("data_loss_weight", DATA_LOSS_WEIGHT)
    mlflow.log_param("physics_loss_weight", PHYSICS_LOSS_WEIGHT)
    mlflow.log_param("ic_loss_weight", IC_LOSS_WEIGHT)

    mlflow.log_param("end_time", end_time)
    mlflow.log_param("num_epochs_init", num_epoch)
    mlflow.log_param("device", device)

    model = FCN(2,1,64,4).to(device)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-6)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("scheduler", "CosineAnnealingLR")
    mlflow.log_param("eta_min", 1e-6)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    files = []  # For saving plot images
    final_epoch = 0

    for i in range(num_epoch):
        optimizer.zero_grad()
        y_pred = model(x_train)
        data_loss = torch.mean((y_pred - y_train)**2)
        # data_loss = torch.tensor(0.0, device=device)  # Ignore data loss for PINN
        # loss = data_loss
        # Physics-informed loss calculation
        physics_losses = []
        # Enforce initial condition: theta(0) = start_angle for each start_angle
        ic_losses = []
        for L_phys in start_angles:
            t0 = torch.zeros(1, 1, device=device, requires_grad=True)
            l0 = torch.full_like(t0, L_phys)
            x0 = torch.cat([t0, l0], dim=1)
            y0 = model(x0)
            # Position IC
            ic_pos = (y0 - L_phys) ** 2
            # Velocity IC
            dy_dt0 = torch.autograd.grad(y0, t0, torch.ones_like(y0), create_graph=True)[0]
            ic_vel = (dy_dt0 - 0.0) ** 2
            ic_losses.append(ic_pos + ic_vel)

            t_phys = torch.linspace(0, end_time, 200, device=device).view(-1,1)
            l_phys = torch.full_like(t_phys, L_phys)
            x_phys = torch.cat([t_phys, l_phys], dim=1).requires_grad_(True)
            y_phys = model(x_phys)
            grad1 = torch.autograd.grad(y_phys, x_phys, torch.ones_like(y_phys), create_graph=True)[0]
            dy_dt = grad1[:,0:1]
            grad2 = torch.autograd.grad(dy_dt, x_phys, torch.ones_like(dy_dt), create_graph=True)[0]
            d2y_dt2 = grad2[:,0:1]
            # Use fixed L, not start_angle!
            residual = d2y_dt2 + (g / L) * torch.sin(y_phys)
            physics_losses.append(torch.mean(residual**2))

        ic_loss = torch.mean(torch.stack(ic_losses))
        physics_loss = torch.mean(torch.stack(physics_losses))
        loss = DATA_LOSS_WEIGHT * data_loss + PHYSICS_LOSS_WEIGHT * physics_loss + IC_LOSS_WEIGHT * ic_loss
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # log metrics every 1000 epochs
        if (i+1) %1000 == 0:
            # # Plot training predictions (use time column from x_train)
            # # Ensure we pass 1D arrays for plotting
            # plot_result(
            #     i+1,
            #     x_train[:,0].detach().cpu(),
            #     y_train.detach().cpu().squeeze(),
            #     y_pred.detach().cpu().squeeze(),
            #     start_angles
            # )
            # os.makedirs('../plots', exist_ok=True)
            # file = f"../plots/{model_name}_{i+1:08d}.png"
            # plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100)
            # plt.close()
            # files.append(file)
            # Log training loss components
            mlflow.log_metric("train_loss_total", round(float(loss.item()), 6), step=i+1)
            mlflow.log_metric("train_loss_data", round(float(data_loss.item()), 6), step=i+1)
            mlflow.log_metric("train_loss_physics", round(float(physics_loss.item()), 6), step=i+1)
            mlflow.log_metric("train_loss_ic", round(float(ic_loss.item()), 6), step=i+1)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=i+1)

        if (i+1) %N_EPOCHS == 0:
            # # Plot training predictions (use time column from x_train)
            # # Ensure we pass 1D arrays for plotting
            plot_result(
                i+1,
                x_train[:,0].detach().cpu(),
                y_train.detach().cpu().squeeze(),
                y_pred.detach().cpu().squeeze(),
                start_angles
            )
            os.makedirs('../plots', exist_ok=True)
            file = f"../plots/{model_name}_{i+1:08d}.png"
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close()
            files.append(file)

        current_loss = float(loss.item())
        if (i+1) % 1000 == 0:
            print(f"Epoch {i+1} Total: {current_loss:.6f}| Data: {float(data_loss.item()):.6f} | Physics: {float(physics_loss.item()):.6f} | IC: {float(ic_loss.item()):.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        final_epoch = i+1
    mlflow.log_param("total_training_epochs", final_epoch)
    return model, files

# Wrapper to start MLflow run and train model
def start_training(model_name, x_train, y_train, x_test, y_test, num_epochs):
    mlflow.end_run()
    with mlflow.start_run(run_name=model_name):
        model, files = train(model_name, x_train, y_train, num_epoch=num_epochs)

        # Final evaluation on the test set (after all training is done)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        with torch.no_grad():
            y_test_pred = model(x_test)
            test_mae = torch.mean(torch.abs(y_test_pred - y_test)).item()
            test_mse = torch.mean((y_test_pred - y_test)**2).item()
            test_rmse = torch.sqrt(torch.mean((y_test_pred - y_test)**2)).item()

        # Log and print final test metrics
        mlflow.log_metric("test_MSE", round(test_mse, 2))
        mlflow.log_metric("test_MAE", round(test_mae, 3))
        mlflow.log_metric("test_RMSE", round(test_rmse,3))
        print(f"Final test MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")

        # Save a final plot comparing predictions to true test values
        plot_result(
            "test_final",
            x_test[:,0].detach().cpu(),
            y_test.detach().cpu().squeeze(),
            y_test_pred.detach().cpu().squeeze(),
            start_angle_test
        )
        os.makedirs('../plots', exist_ok=True)
        test_file = f"../plots/{model_name}_test.png"
        plt.savefig(test_file, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close()
        mlflow.log_artifact(test_file)

        # If there were intermediate files (for GIF), save GIF and clean up
        if files:
            try:
                # save_gif_PIL(f"../plots/{model_name}.gif", files, fps=10, loop=0)
                mlflow.log_artifact(files[-1])
            except Exception:
                pass
            for f in files:
                if os.path.exists(f):
                    os.remove(f)

        # Log the trained model
        mlflow.pytorch.log_model(model, "models")
    mlflow.end_run()

# Main entry point
if __name__ == "__main__":
    model_name = MODEL_NAME
    # Start training with explicit train / test tensors
    start_training(model_name, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, num_epochs=N_EPOCHS)

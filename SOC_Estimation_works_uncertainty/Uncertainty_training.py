import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from calce_data_Process import BatteryDataProcessor
from calce_plotting import BatteryDataPlotter
from gru_uncertainty_model import GRUEvidentialModel
from gru_uncertainty_trainer import GRUEvidentialTrainer

# Load battery data
folder_path = "./training_Set"
processor = BatteryDataProcessor(folder_path)
df = processor.load_data()
print(df.head())

# Optional: Plot
plotter = BatteryDataPlotter(df)
plotter.plot_separate_by_source_sorter()

# Parameters
seq_length = 100
batch_size = 256
num_epochs = 150
save_dir = "./saved_models"

# Select features and target
df_filtered = df[df["Step Index"] == 7]
train_features = df_filtered[["Current", "Voltage", "Temperature"]].values


# ######for Noisy Training###############################
# def add_input_noise(X, std_dict, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     X_noisy = X.copy()
#     for i, col in enumerate(["Current", "Voltage", "Temperature"]):
#         noise = np.random.normal(0, std_dict[col], size=X.shape[0])
#         X_noisy[:, i] += noise
#     return X_noisy

# # Inject Gaussian noise before training
# std_dict = {"Current": 0.5, "Voltage": 0.05, "Temperature": 5}
# train_features = df_filtered[["Current", "Voltage", "Temperature"]].values
# train_features = add_input_noise(train_features, std_dict, seed=42)

# ######for Noisy Training###############################

train_soc = df_filtered["gt_soc"].values.reshape(-1, 1)

# Normalize
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))
train_features_scaled = scaler_features.fit_transform(train_features)
train_soc_scaled = scaler_target.fit_transform(train_soc)

# Define model
input_size = 3
hidden_size = 256
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GRUEvidentialModel(input_size, hidden_size, num_layers).to(device)
torch.cuda.empty_cache()

lambda_reg = 10

# Train
trainer = GRUEvidentialTrainer(model, train_features_scaled, train_soc_scaled,
                               seq_length=seq_length,
                               batch_size=batch_size,
                               num_epochs=num_epochs,
                               save_dir=save_dir,lambda_reg=lambda_reg)
trainer.train()

# Save utilities
trainer.save_scalers(scaler_features, scaler_target)
trainer.save_training_config()
trainer.plot_loss()

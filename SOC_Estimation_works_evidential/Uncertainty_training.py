import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from calce_data_Process import BatteryDataProcessor
from calce_plotting import BatteryDataPlotter
from gru_uncertainty_model import GRUEvidentialModel
from torch.utils.data import DataLoader, TensorDataset
from gru_uncertainty_trainer import GRUEvidentialTrainer

folder_path = "./training_Set"  # Update with the actual folder path


processor = BatteryDataProcessor(folder_path)
df = processor.load_data()

# Display the first few rows
print(df.head())

# plotter= BatteryDataPlotter(df)

# plotter.plot_step_index_7_by_drain_soc_source()

# plotter.plot_separate_by_drain_soc_source()

# plotter.plot_separate_by_source_sorter()

# Define sliding window size





seq_length = 100
batch_size = 256
num_epochs = 150
save_dir = "./saved_models"

# # Use Current (A) and Voltage (V) as input features and SOC as the target variable
# train_features = df[["Current", "Voltage"]].values
# train_soc = df["gt_soc"].values.reshape(-1, 1)

df_filtered = df[df["Step Index"] == 7]  # Filter data for Step Index 7
train_features=df_filtered[["Current","Voltage","Temperature"]].values
train_soc=df_filtered["gt_soc"].values.reshape(-1,1)


val_path = "./validation_set"
val_processor = BatteryDataProcessor(val_path)
val_df = val_processor.load_data()
val_df_filtered = val_df[val_df["Step Index"] == 7]  # Filter data for Step Index 7
val_features = val_df_filtered[["Current", "Voltage","Temperature"]].values
val_soc = val_df_filtered["gt_soc"].values.reshape(-1, 1)

# Normalize data (MinMax Scaling to range [0,1])
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))
train_features_scaled = scaler_features.fit_transform(train_features)
train_soc_scaled = scaler_target.fit_transform(train_soc)

val_features_scaled = scaler_features.transform(val_features)
val_soc_scaled = scaler_target.transform(val_soc)





# Define model parameters
input_size = 3  # Current & Voltage as input features
hidden_size = 256  # Number of hidden neurons
num_layers = 2  # Number of GRU layers
output_size = 1  # Predicting SOC
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GRUEvidentialModel(input_size, hidden_size, num_layers).to(device)
torch.cuda.empty_cache()

lambda_reg = .1

# Train
trainer = GRUEvidentialTrainer(model, train_features_scaled, train_soc_scaled,
                               val_features_scaled, val_soc_scaled,
                               seq_length=seq_length,
                               batch_size=batch_size,
                               num_epochs=num_epochs,
                               save_dir=save_dir,lambda_reg=lambda_reg)
trainer.train()

# Save utilities
trainer.save_scalers(scaler_features, scaler_target)
trainer.save_training_config()
trainer.plot_loss()

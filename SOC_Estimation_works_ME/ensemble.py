import os
import shutil
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
from gru_model import GRUModel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from gru_trainer import GRUTrainer

folder_path = "./training_Set"  # Update with the actual folder path
processor = BatteryDataProcessor(folder_path)
df = processor.load_data()

# Display the first few rows
print(df.head())

plotter= BatteryDataPlotter(df)

# plotter.plot_step_index_7_by_drain_soc_source()

# plotter.plot_separate_by_drain_soc_source()

# plotter.plot_separate_by_source_sorter()


def get_next_experiment_folder(self, save_dir):
    """Find the next available `expX` folder inside `save_dir`."""
    existing_exps = [d for d in os.listdir(save_dir) if d.startswith("exp") and d[3:].isdigit()]
    exp_nums = [int(d[3:]) for d in existing_exps]
    next_exp_num = max(exp_nums, default=0) + 1
    exp_path = os.path.join(save_dir, f"exp{next_exp_num}")
    os.makedirs(exp_path, exist_ok=True)
    return exp_path




val_path = "./validation_set"
val_processor = BatteryDataProcessor(val_path)
val_df = val_processor.load_data()
val_df_filtered = val_df[val_df["Step Index"] == 7]  # Filter data for Step Index 7
val_features = val_df_filtered[["Current", "Voltage","Temperature"]].values
val_soc = val_df_filtered["gt_soc"].values.reshape(-1, 1)


# This normalization is common for all models
scaler_features_val = MinMaxScaler()
scaler_target_val = MinMaxScaler()

# Note: These will be overwritten later by each model's own training scalers
val_features_scaled_template = scaler_features_val.fit_transform(val_features)
val_soc_scaled_template = scaler_target_val.fit_transform(val_soc)






# Define sliding window size



seq_length = 100
batch_size = 256
num_epochs = 150
save_dir = "./saved_models"

# # Use Current (A) and Voltage (V) as input features and SOC as the target variable
# train_features = df[["Current", "Voltage"]].values
# train_soc = df["gt_soc"].values.reshape(-1, 1)

df_filtered = df[df["Step Index"] == 7]  # Filter data for Step Index 7
unique_sources = df_filtered['source_sorter'].unique()

temp_trainer = GRUTrainer(
    model=GRUModel(3, 256, 2, 1),
    features=np.zeros((150, 3)),  # dummy init
    target=np.zeros((150, 1)),
    val_features=np.zeros((150, 3)),
    val_target=np.zeros((150, 1)),
    num_epochs=1,
    save_dir=save_dir
)


exp_dir = temp_trainer.exp_dir

del temp_trainer


best_models_dir = os.path.join(exp_dir, "best_models")
last_models_dir = os.path.join(exp_dir, "last_models")
os.makedirs(best_models_dir, exist_ok=True)
os.makedirs(last_models_dir, exist_ok=True)


for i, left_out_source in enumerate(unique_sources):
    print(f"\n=== Training model {i+1} with '{left_out_source}' excluded from training ===")

    # Leave one source out
    train_df = df_filtered[df_filtered['source_sorter'] != left_out_source]

    train_features = train_df[["Current", "Voltage", "Temperature"]].values
    train_soc = train_df["gt_soc"].values.reshape(-1, 1)

    # Normalize based on training data only
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    train_features_scaled = scaler_features.fit_transform(train_features)
    train_soc_scaled = scaler_target.fit_transform(train_soc)

    # Apply training scalers to fixed validation set
    val_features_scaled = scaler_features.transform(val_features)
    val_soc_scaled = scaler_target.transform(val_soc)

    model = GRUModel(3, 256, 2, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.cuda.empty_cache()

    model_dir = os.path.join(exp_dir, f"model{i+1}")
    trainer = GRUTrainer(
        model=model,
        features=train_features_scaled,
        target=train_soc_scaled,
        val_features=val_features_scaled,
        val_target=val_soc_scaled,
        seq_length=100,
        batch_size=256,
        num_epochs=150,
        save_dir=model_dir
    )

    trainer.train()
    trainer.save_scalers(scaler_features, scaler_target)
    trainer.save_training_config()
    trainer.plot_loss()

    # Save best/last models to central folder
    shutil.copy(os.path.join(model_dir,"exp1", "best_gru_model.pth"), os.path.join(best_models_dir, f"model{i+1}_best.pth"))
    shutil.copy(os.path.join(model_dir,"exp1","last_gru_model.pth"), os.path.join(last_models_dir, f"model{i+1}_last.pth"))





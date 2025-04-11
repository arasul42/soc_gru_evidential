# import pandas as pd
# import torch
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from calce_data_Process import BatteryDataProcessor

# from gru_model import GRUModel


# folder_path = "./test_set"  # Update with the actual folder path

# test_data=BatteryDataProcessor(folder_path)

# df_new=test_data.load_data()

# print(df_new)


# # Load trained GRU model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GRUModel(input_size=3, hidden_size=256, num_layers=2, output_size=1).to(device)
# model.load_state_dict(torch.load("saved_models/exp25/last_gru_model.pth", map_location=device),strict=False)
# model.eval()

# # Clear CUDA cache
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# # Load scalers
# scaler_target = joblib.load("saved_models/exp25/scaler_target.pkl")
# scaler_features = joblib.load("saved_models/exp25/scaler_features.pkl")

# df_new = df_new[df_new["Step Index"] == 7]

# # Extract features (Current, Voltage)
# new_features = df_new[["Current", "Voltage", "Temperature"]].values

# # Normalize using the same scaler as training
# new_features_scaled = scaler_features.transform(new_features)

# # Define the same sequence length
# seq_length = 100

# # Create sequences (Ensure it's the same as training)
# def create_sequences(features, seq_length):
#     sequences = []
#     for i in range(len(features) - seq_length):
#         sequences.append(features[i:i + seq_length])
#     return np.array(sequences)

# X_new = create_sequences(new_features_scaled, seq_length)

# # Check if there are enough sequences for prediction
# if len(X_new) == 0:
#     raise ValueError("Not enough data to form sequences for prediction.")

# # Convert to PyTorch tensor
# X_new_torch = torch.tensor(X_new, dtype=torch.float32).to(device)


# # Predict SOC for new dataset
# model.eval()
# with torch.no_grad():
#     y_pred_new_torch = model(X_new_torch)

# # Convert predictions back to original SOC scale
# y_pred_new = scaler_target.inverse_transform(y_pred_new_torch.cpu().numpy())


# # Extract actual SOC values (if available)
# if "gt_soc" in df_new.columns and len(df_new["gt_soc"]) > seq_length:
#     y_actual_new = df_new["gt_soc"].values[seq_length:].reshape(-1, 1)
# else:
#     y_actual_new = np.full((len(y_pred_new), 1), np.nan)

# # Compute RMSE & MAE
# rmse_new = np.sqrt(mean_squared_error(y_actual_new, y_pred_new))
# mae_new = mean_absolute_error(y_actual_new, y_pred_new)

# # Print RMSE & MAE
# print(f"RMSE on New Dataset: {rmse_new:.6f}")
# print(f"MAE on New Dataset: {mae_new:.6f}")

# # Save predictions to DataFrame
# df_new["Predicted_SOC"] = np.nan  # Ensure column exists
# df_new.iloc[seq_length:seq_length + len(y_pred_new), df_new.columns.get_loc("Predicted_SOC")] = y_pred_new.flatten()

# # Save results to an Excel file
# df_new.to_excel("SOC_Predictions_New_Dataset.xlsx", index=False)
# print("Predictions saved to 'SOC_Predictions_New_Dataset.xlsx'")

# source = df_new["Drain_SOC_Source"].unique()
# source = source.item()

# # Plot Predictions vs. True SOC against Total Time
# plt.figure(figsize=(10, 10))
# plt.plot(df_new["Total Time"].iloc[seq_length:], y_actual_new, label="True SOC", color="b")
# plt.plot(df_new["Total Time"].iloc[seq_length:], y_pred_new, label="Predicted SOC", color="r", linestyle="dashed")
# plt.xlabel("Total Time (s)")
# plt.ylabel("SOC")
# plt.ylim(0, 1)
# plt.title(f"GRU Prediction on {source} Dataset")
# plt.legend()
# plt.grid(True)
# plt.savefig(f"./plots/prediction on {source}.png")
# plt.show()


import os
import re
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from gru_model import GRUModel
from calce_data_Process import BatteryDataProcessor


def get_latest_exp_folder(base_path="./saved_models"):
    exp_dirs = [d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d)) and re.match(r'exp\d+', d)]
    if not exp_dirs:
        raise FileNotFoundError("No experiment directories found in ./saved_models")
    latest_exp = max(exp_dirs, key=lambda x: int(re.search(r'\d+', x).group()))
    return os.path.join(base_path, latest_exp)


def create_sequences(features, seq_length):
    return np.array([features[i:i + seq_length] for i in range(len(features) - seq_length)])


def evaluate_gru_by_source_sorter(df, model_output_dir=None, seq_length=100):
    # Use latest exp directory by default
    if model_output_dir is None:
        model_output_dir = get_latest_exp_folder()

    # Prepare subfolders inside model/output directory
    plots_dir = os.path.join(model_output_dir, "Prediction Plots")
    results_dir = os.path.join(model_output_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load GRU model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(input_size=3, hidden_size=256, num_layers=2, output_size=1).to(device)
    model.load_state_dict(torch.load(os.path.join(model_output_dir, "last_gru_model.pth"), map_location=device), strict=False)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load scalers
    scaler_target = joblib.load(os.path.join(model_output_dir, "scaler_target.pkl"))
    scaler_features = joblib.load(os.path.join(model_output_dir, "scaler_features.pkl"))

    # Filter Step Index
    df = df[df["Step Index"] == 7]

    results = []

    for source in df["source_sorter"].unique():
        df_source = df[df["source_sorter"] == source].copy()

        if len(df_source) < seq_length:
            print(f"Skipping {source}: Not enough data.")
            continue

        features = df_source[["Current", "Voltage", "Temperature"]].values
        features_scaled = scaler_features.transform(features)
        X = create_sequences(features_scaled, seq_length)

        if len(X) == 0:
            print(f"Skipping {source}: No sequences created.")
            continue

        X_torch = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_torch = model(X_torch)

        y_pred = scaler_target.inverse_transform(y_pred_torch.cpu().numpy())

        if "gt_soc" in df_source.columns and len(df_source["gt_soc"]) > seq_length:
            y_actual = df_source["gt_soc"].values[seq_length:].reshape(-1, 1)
        else:
            y_actual = np.full((len(y_pred), 1), np.nan)

        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)

        results.append({
            "source_sorter": source,
            "RMSE": rmse,
            "MAE": mae
        })

        df_source["Predicted_SOC"] = np.nan
        df_source.iloc[seq_length:seq_length + len(y_pred),
                       df_source.columns.get_loc("Predicted_SOC")] = y_pred.flatten()

        # Plot
        plt.figure(figsize=(4.72, 3))
        plt.plot(df_source["Total Time"].iloc[seq_length:], y_actual, label="True SOC", color="b")
        plt.plot(df_source["Total Time"].iloc[seq_length:], y_pred, label="Predicted SOC", color="r", linestyle="dashed")
        plt.xlabel("Time (s)")
        plt.ylabel("SOC")
        plt.ylim(0, 1)
        plt.title(f"GRU Prediction on {source}")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(plots_dir, f"GRU_Prediction_{source}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        df_source.to_excel(os.path.join(results_dir, f"PREDICTED_SOC_{source}.xlsx"), index=False)

        print(f"✅ Processed: {source} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Save combined metrics
    df_metrics = pd.DataFrame(results)
    df_metrics.to_excel(os.path.join(model_output_dir, "GRU_Evaluation_Metrics.xlsx"), index=False)
    print(f"\n📁 Evaluation complete. Results saved in: {model_output_dir}")

folder_path = "./test_set"  # Update with the actual folder path

test_data=BatteryDataProcessor(folder_path)

df_new=test_data.load_data()

print(df_new)

evaluate_gru_by_source_sorter(df_new)
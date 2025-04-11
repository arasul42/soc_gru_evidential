import os
import re
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from gru_model import GRUEvidentialModel
from calce_data_Process import BatteryDataProcessor
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def get_latest_exp_folder(base_path="./saved_models"):
    exp_dirs = [d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d)) and re.match(r'exp\d+', d)]
    if not exp_dirs:
        raise FileNotFoundError("No experiment directories found in ./saved_models")
    latest_exp = max(exp_dirs, key=lambda x: int(re.search(r'\d+', x).group()))
    return os.path.join(base_path, latest_exp)

def create_sequences(features, seq_length):
    return np.array([features[i:i + seq_length] for i in range(len(features) - seq_length)])

def evaluate_evidential_gru_by_source_sorter(df, model_output_dir=None, seq_length=100):
    if model_output_dir is None:
        model_output_dir = get_latest_exp_folder()

    plots_dir = os.path.join(model_output_dir, "Prediction Plots")
    results_dir = os.path.join(model_output_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUEvidentialModel(input_size=3, hidden_size=256, num_layers=2).to(device)
    model.load_state_dict(torch.load(os.path.join(model_output_dir, "last_gru_model.pth"), map_location=device), strict=False)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scaler_target = joblib.load(os.path.join(model_output_dir, "scaler_target.pkl"))
    scaler_features = joblib.load(os.path.join(model_output_dir, "scaler_features.pkl"))

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
            gamma, nu, alpha, beta = model(X_torch)

        pred_scaled = gamma.cpu().numpy()
        pred = scaler_target.inverse_transform(pred_scaled)

        aleatoric = (beta / (alpha - 1)).cpu().numpy()
        epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
        total_uncertainty = aleatoric + epistemic

        if "gt_soc" in df_source.columns and len(df_source["gt_soc"]) > seq_length:
            y_actual = df_source["gt_soc"].values[seq_length:].reshape(-1, 1)
        else:
            y_actual = np.full((len(pred), 1), np.nan)

        rmse = np.sqrt(mean_squared_error(y_actual, pred))
        mae = mean_absolute_error(y_actual, pred)

        results.append({
            "source_sorter": source,
            "RMSE": rmse,
            "MAE": mae
        })

        df_source["Predicted_SOC"] = np.nan
        df_source.iloc[seq_length:seq_length + len(pred), df_source.columns.get_loc("Predicted_SOC")] = pred.flatten()

        # Kernel-style uncertainty plot using LineCollection
        time = df_source["Total Time"].iloc[seq_length:].values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, y_actual, label="True SOC", color="blue", linewidth=1.5)

        norm = Normalize(vmin=total_uncertainty.min(), vmax=total_uncertainty.max())
        points = np.array([time, pred.flatten()]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(total_uncertainty.flatten())
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.set_xlim(time.min(), time.max())
        ax.set_ylim(0, 1)

        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Total Uncertainty')

        ax.set_title(f"Kernel-colored Prediction: {source}")
        ax.set_xlabel("Total Time (s)")
        ax.set_ylabel("SOC")
        ax.legend()
        ax.grid(True)

        plot_path = os.path.join(plots_dir, f"Evidential_GRU_KernelPlot_{source}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        df_source.to_excel(os.path.join(results_dir, f"PREDICTED_SOC_{source}.xlsx"), index=False)
        print(f"✅ Processed: {source} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    df_metrics = pd.DataFrame(results)
    df_metrics.to_excel(os.path.join(model_output_dir, "Evidential_GRU_Evaluation_Metrics.xlsx"), index=False)
    print(f"\n📁 Evaluation complete. Results saved in: {model_output_dir}")

folder_path = "./test_set"
test_data = BatteryDataProcessor(folder_path)
df_new = test_data.load_data()
evaluate_evidential_gru_by_source_sorter(df_new)

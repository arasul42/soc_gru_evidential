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

def evaluate_evidential_gru_by_source_sorter(df, model_output_dir=None, seq_length=100):
    if model_output_dir is None:
        model_output_dir = get_latest_exp_folder()

    plots_dir = os.path.join(model_output_dir, "Prediction Plots")
    results_dir = os.path.join(model_output_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(input_size=3, hidden_size=256, num_layers=2).to(device)
    model.load_state_dict(torch.load(os.path.join(model_output_dir, "last_gru_model.pth"), map_location=device), strict=False)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scaler_target = joblib.load(os.path.join(model_output_dir, "scaler_target.pkl"))
    scaler_features = joblib.load(os.path.join(model_output_dir, "scaler_features.pkl"))

    # df = df[df["Step Index"] == 7]
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
            output = model(X_torch)

        # var_al = torch.sqrt(beta / ((alpha - 1))).cpu()
        # var_ep = torch.sqrt(beta / (nu * (alpha - 1))).cpu()
        # var_total = torch.sqrt((beta / (alpha - 1)) * (1 + (1 / nu))).cpu()
        # var_al_np = var_al.numpy().flatten()
        # var_ep_np = var_ep.numpy().flatten()
        # var_total_np = var_total.numpy().flatten()
        
        # gamma_cpu = gamma.cpu().numpy()
        # gamma_np = scaler_target.inverse_transform(gamma.cpu().numpy()).flatten()
        pred = scaler_target.inverse_transform(output.cpu().numpy()).flatten()

        # aleatoric = (beta / (alpha - 1)).cpu().numpy()
        # epistemic = (beta / (nu * (alpha - 1))).cpu().numpy()
        # total_uncertainty = aleatoric + epistemic

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

        time = df_source["Total Time"].iloc[seq_length:]
        plt.figure(figsize=(4.72, 3))
        plt.plot(time, y_actual, label="True SOC", color="blue", linewidth=1)
        plt.plot(time, pred, label="Predicted SOC", color="red", linestyle="--", linewidth=1)


        plt.xlabel("Total Time (s)")
        plt.ylabel("SOC")
        plt.ylim(0, 1)
        plt.title(f"Prediction on {source}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"Prediction_{source}.png")
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

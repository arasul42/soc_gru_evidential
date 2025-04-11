import os
import re
import gc
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from gru_uncertainty_model import GRUEvidentialModel
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

def add_input_noise(df, std_dict, seed=None):
    df_noisy = df.copy()
    if seed is not None:
        np.random.seed(seed)
    for col, std in std_dict.items():
        if col in df_noisy.columns:
            noise = np.random.normal(loc=0.0, scale=std, size=len(df_noisy))
            df_noisy[col] += noise
    return df_noisy


def add_salt_pepper_noise(df, prob=0.05, magnitude=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility

    df_noisy = df.copy()
    for col in df.columns:
        mask = np.random.rand(len(df)) < prob
        spikes = np.random.choice([-magnitude, magnitude], size=len(df))
        df_noisy[col] += mask * spikes
    return df_noisy


def model_inference_batched(model, X_torch, batch_size=512):
    mus, log_vars = [], []
    for i in range(0, len(X_torch), batch_size):
        X_batch = X_torch[i:i+batch_size]
        with torch.no_grad():
            mu, log_var = model(X_batch)
            mus.append(mu.squeeze())
            log_vars.append(log_var.squeeze())
    mu_all = torch.cat(mus)
    log_var_all = torch.cat(log_vars)
    return mu_all, log_var_all



def evaluate_evidential_gru_by_source_sorter(df, model_output_dir=None, seq_length=100):
    if model_output_dir is None:
        model_output_dir = get_latest_exp_folder()

    plots_dir = os.path.join(model_output_dir, "Prediction Plots")
    noiseless_dir = os.path.join(model_output_dir, "Noiseless Plots")
    results_dir = os.path.join(model_output_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(noiseless_dir, exist_ok=True)

    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUEvidentialModel(input_size=3, hidden_size=256, num_layers=2).to(device)
    model.load_state_dict(torch.load(os.path.join(model_output_dir, "best_gru_model.pth"), map_location=device), strict=False)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scaler_target = joblib.load(os.path.join(model_output_dir, "scaler_target.pkl"))
    scaler_features = joblib.load(os.path.join(model_output_dir, "scaler_features.pkl"))

    results = []

    for source in df["source_sorter"].unique():
        df_source = df[df["source_sorter"] == source].copy()

        if len(df_source) < seq_length:
            print(f"Skipping {source}: Not enough data.")
            continue

        # 🔊 Inject noise before inference
        # noise_std = {"Current": .5, "Voltage": .05, "Temperature": 5}
        # df_source = add_input_noise(df_source, noise_std)
        # df_source[["Current", "Voltage", "Temperature"]] = add_salt_pepper_noise(df_source[["Current", "Voltage", "Temperature"]], prob=0.05, magnitude=0.2, seed=42)
        noise_std = {"Current": .5, "Voltage": 0.05, "Temperature": 5}
        df_source = add_input_noise(df_source, noise_std, seed=42)





        features = df_source[["Current", "Voltage", "Temperature"]].values
        features_scaled = scaler_features.transform(features)
        X = create_sequences(features_scaled, seq_length)


        gc.collect()
        torch.cuda.empty_cache()


        if len(X) == 0:
            print(f"Skipping {source}: No sequences created.")
            continue

        X_torch = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, log_var = model_inference_batched(model, X_torch)
        
        var = torch.exp(log_var).cpu().numpy().reshape(-1, 1)
        std = np.sqrt(var)
        mu_np = mu.cpu().numpy().reshape(-1, 1)
        pred = scaler_target.inverse_transform(mu_np)
        std =std* scaler_target.scale_
        pred = pred.flatten()
        std = std.flatten()



        if "gt_soc" in df_source.columns and len(df_source["gt_soc"]) > seq_length:
            y_actual = df_source["gt_soc"].values[seq_length:].reshape(-1, 1)
        else:
            y_actual = np.full((len(pred), 1), np.nan)

        rmse = np.sqrt(mean_squared_error(y_actual, pred))
        mae = mean_absolute_error(y_actual, pred)

        results.append({
            "source_sorter": source,
            "RMSE": rmse,
            "MAE": mae,
            "std": np.mean(std),
            # "Aleatoric_Uncertainty": np.mean(var_al_np)
        })

        df_source["Predicted_SOC"] = np.nan
        df_source.iloc[seq_length:seq_length + len(pred), df_source.columns.get_loc("Predicted_SOC")] = pred.flatten()
        df_source["Std_dev"] = np.nan
        df_source.iloc[seq_length:seq_length + len(std), df_source.columns.get_loc("Std_dev")] = std.flatten()
    
        time = df_source["Total Time"].iloc[seq_length:]

        # 🌿 Epistemic Uncertainty Plot
        plt.figure(figsize=(4.72, 3))
        plt.plot(time, y_actual, label="True SOC", color="blue", linewidth=1)
        plt.plot(time, pred, label="Predicted SOC", color="red", linestyle="--", linewidth=1)
        for std_scale in range(3):
            plt.fill_between(
                time,
                pred - std_scale * std,
                pred + std_scale * std,
                alpha=0.3,
                facecolor="tab:orange",
                label="Uncertainty"
            )

        plt.xlabel("Time (s)")
        # plt.ylabel("SOC")
        plt.title(f"Prediction on {source}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Epistemic_GRU_Prediction_{source}.png"), dpi=300)
        plt.close()



        # Round the middle point down to nearest 50 to get a clean zoom_start
        zoom_start = int(np.floor(time.iloc[len(time) // 2] / 50.0)) * 50
        zoom_end = zoom_start + 250

        zoom_mask = (time >= zoom_start) & (time <= zoom_end)

        time_zoom = time[zoom_mask]
        pred_zoom = pred.flatten()[zoom_mask]
        gt_zoom = y_actual.flatten()[zoom_mask]
        std_zoom = std[zoom_mask]


        plt.figure(figsize=(4.72, 3.5))
        plt.plot(time_zoom, gt_zoom, label="True SOC", color="blue")
        plt.plot(time_zoom, pred_zoom, label="Predicted SOC", color="red", linestyle="--")

        plt.fill_between(time_zoom, pred_zoom - 3 * std_zoom, pred_zoom + 3 * std_zoom,
            alpha=0.3, color="tab:green", label="Epistemic Uncertainty")
        plt.fill_between(time_zoom, pred_zoom - 3 * std_zoom, pred_zoom + 3 * std_zoom,
            alpha=0.3, color="tab:blue", label="Aleatoric Uncertainty")

        plt.title(f"Prediction on {source}")
        plt.xlabel("Time (s)")
        # plt.ylabel("SOC")
        
        plt.legend(loc="best", framealpha=0.5, fontsize='small')


        plt.grid(True)

        # Use the same zoom_start as xtick_start
        xtick_start = zoom_start
        xtick_end = int(np.ceil(time_zoom.iloc[-1] / 50.0)) * 50
        plt.xticks(np.arange(xtick_start, xtick_end + 1, 50))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Zoomed_Total_{source}.png"), dpi=300)
        plt.close()







        df_source.to_excel(os.path.join(results_dir, f"PREDICTED_SOC_{source}.xlsx"), index=False)
        print(f"✅ Processed: {source} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    df_metrics = pd.DataFrame(results)
    df_metrics.to_excel(os.path.join(model_output_dir, "Evidential_GRU_Evaluation_Metrics.xlsx"), index=False)
    print(f"\n📁 Evaluation complete. Results saved in: {model_output_dir}")


    # Generate all plots without added noise
    for source in df["source_sorter"].unique():
        df_source = df[df["source_sorter"] == source].copy()

        if len(df_source) < seq_length:
            print(f"Skipping {source}: Not enough data.")
            continue

        features = df_source[["Current", "Voltage", "Temperature"]].values
        features_scaled = scaler_features.transform(features)
        X = create_sequences(features_scaled, seq_length)


        gc.collect()
        torch.cuda.empty_cache()


        if len(X) == 0:
            print(f"Skipping {source}: No sequences created.")
            continue

        X_torch = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, log_var = model_inference_batched(model, X_torch)
        
        var = torch.exp(log_var).cpu().numpy()
        std = np.sqrt(var)
        mu_np = mu.cpu().numpy().reshape(-1, 1)
        pred = scaler_target.inverse_transform(mu_np)
        std =std* scaler_target.scale_
        pred = pred.flatten()
        std = std.flatten()



        if "gt_soc" in df_source.columns and len(df_source["gt_soc"]) > seq_length:
            y_actual = df_source["gt_soc"].values[seq_length:].reshape(-1, 1)
        else:
            y_actual = np.full((len(pred), 1), np.nan)

        rmse = np.sqrt(mean_squared_error(y_actual, pred))
        mae = mean_absolute_error(y_actual, pred)

        print(f"✅ Processed without Noise: {source} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        matching_result = next((res for res in results if res["source_sorter"] == source), None)
        if matching_result:
            matching_result.update({
            "RMSE_Clean": rmse,
            "MAE_Clean": mae,
            "Std_Clean": np.mean(std),
            })
        else:
            results.append({
            "source_sorter": source,
            "RMSE_Clean": rmse,
            "MAE_Clean": mae,
            "Std_Clean": np.mean(std),
            })

        time = df_source["Total Time"].iloc[seq_length:]

        # Generate noiseless plots
        plt.figure(figsize=(4.72, 3))
        plt.plot(time, y_actual, label="True SOC", color="blue", linewidth=1)
        plt.plot(time, pred, label="Predicted SOC", color="red", linestyle="--", linewidth=1)
        for std_scale in range(3):
            plt.fill_between(
                time,
                pred - std * std_scale,
                pred + std * std_scale,
                alpha=0.9,
                facecolor="tab:green",
                label="Unc")
        plt.xlabel("Time (s)")
        plt.title(f"Prediction on {source}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(noiseless_dir, f"Epistemic_GRU_Prediction_{source}_Noiseless.png"), dpi=300)
        plt.close()

 

        # Generate zoomed-in noiseless plots
# ---- Zoomed-in Plot for Gaussian NLL ---- #
        zoom_start = int(np.floor(time.iloc[len(time) // 2] / 50.0)) * 50
        zoom_end = zoom_start + 250

        zoom_mask = (time >= zoom_start) & (time <= zoom_end)

        time_zoom = time[zoom_mask]
        pred_zoom = pred.flatten()[zoom_mask]
        gt_zoom = y_actual.flatten()[zoom_mask]
        std_zoom = std[zoom_mask]  # standard deviation from predicted log variance

        plt.figure(figsize=(4.72, 3.5))
        plt.plot(time_zoom, gt_zoom, label="True SOC", color="blue")
        plt.plot(time_zoom, pred_zoom, label="Predicted SOC", color="red", linestyle="--")

        plt.fill_between(
            time_zoom,
            pred_zoom - 1.96 * std_zoom,
            pred_zoom + 1.96 * std_zoom,
            alpha=0.3,
            color="tab:orange",
            label="Predictive Uncertainty"
        )

        plt.title(f"Prediction on {source}")
        plt.xlabel("Time (s)")
        plt.legend(loc="best", framealpha=0.5, fontsize='small')
        plt.grid(True)

        xtick_start = zoom_start
        xtick_end = int(np.ceil(time_zoom.iloc[-1] / 50.0)) * 50
        plt.xticks(np.arange(xtick_start, xtick_end + 1, 50))

        plt.tight_layout()
        plt.savefig(os.path.join(noiseless_dir, f"Zoomed_Total_{source}_Noiseless.png"), dpi=300)
        plt.close()


        # Append results to a consolidated file
        consolidated_results_path = os.path.join(model_output_dir, "Evidential_GRU_Evaluation_Metrics.xlsx")
        if os.path.exists(consolidated_results_path):
            existing_results = pd.read_excel(consolidated_results_path)
            # Ensure no duplicate entries by checking 'source_sorter'
            new_results_df = pd.DataFrame(results)
            updated_results = pd.concat(
            [existing_results[~existing_results["source_sorter"].isin(new_results_df["source_sorter"])], 
             new_results_df], 
            ignore_index=True
            )
        else:
            updated_results = pd.DataFrame(results)

        updated_results.to_excel(consolidated_results_path, index=False)
        print(f"📄 Consolidated results updated at: {consolidated_results_path}")

        df_source["Predicted_SOC"] = np.nan
        df_source.iloc[seq_length:seq_length + len(pred), df_source.columns.get_loc("Predicted_SOC")] = pred.flatten()
        df_source["Std_dev"] = np.nan
        df_source.iloc[seq_length:seq_length + len(std), df_source.columns.get_loc("Std_dev")] = std.flatten()
    
        time = df_source["Total Time"].iloc[seq_length:]


        df_source.to_excel(os.path.join(results_dir, f"PREDICTED_SOC_without_noise{source}.xlsx"), index=False)
        print(f"✅ Processed: {source} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")




    df_metrics = pd.DataFrame(results)
    df_metrics.to_excel(os.path.join(model_output_dir, "Evidential_GRU_Evaluation_Metrics.xlsx"), index=False)
    print(f"\n📁 Evaluation complete. Results saved in: {model_output_dir}")


# 🔍 Load test set
folder_path = "./test_set"
test_data = BatteryDataProcessor(folder_path)
df_new = test_data.load_data()
df_new = df_new[df_new["Step Index"] == 7]
evaluate_evidential_gru_by_source_sorter(df_new)


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
    outputs = []
    for i in range(0, len(X_torch), batch_size):
        X_batch = X_torch[i:i+batch_size]
        gamma, nu, alpha, beta = model(X_batch)
        outputs.append((gamma, nu, alpha, beta))
    
    # Concatenate outputs from all batches
    gamma_all = torch.cat([o[0] for o in outputs], dim=0)
    nu_all = torch.cat([o[1] for o in outputs], dim=0)
    alpha_all = torch.cat([o[2] for o in outputs], dim=0)
    beta_all = torch.cat([o[3] for o in outputs], dim=0)
    
    return gamma_all, nu_all, alpha_all, beta_all


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
        # noise_std = {"Current": .5, "Voltage": 0.05, "Temperature": 5}
        # df_source = add_input_noise(df_source, noise_std, seed=42)





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
            # gamma, nu, alpha, beta = model_inference_batched(model, X_torch, batch_size=512)
            raw_pred = model(X_torch)
            gamma, nu, alpha, beta = (d.squeeze().cpu().numpy() for d in raw_pred)

        
        gamma = scaler_target.inverse_transform(gamma.reshape(-1,1)).flatten()

        var = np.sqrt(beta / (nu * (alpha - 1)))
        var_al= np.sqrt(beta / (alpha - 1))
        # var = np.minimum(var, 1e3)  # Optional: cap for stability

        # Prepare time and ground truth
        time = df_source["Total Time"].values[seq_length:]
        y_true = df_source["gt_soc"].values[seq_length:]


        plt.figure(figsize=(4.72, 3.5), dpi=300)

        # Ground truth (black)
        plt.plot(time, y_true, color="blue", label="Ground Truth", linewidth=1.5)

        # Predicted mean (red dashed)
        plt.plot(time, gamma, color="tab:red", linestyle="--", label="Predicted Mean", linewidth=1.5)

        # Uncertainty bands (green fill)
      
        plt.fill_between(
            time,
            gamma - 1.96 * var,
            gamma + 1.96 * var,
            alpha=0.15,
            facecolor="green",
            label="Uncertainty Band" ,  # label only once
        )


        plt.ylim(0, 1)
        plt.xlabel("Total Time (s)")
        plt.ylabel("SOC")
        plt.title(f"Evidential GRU Uncertainty - {source}")
        plt.legend(loc="best", fontsize="small", framealpha=0.6)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Evidential_unc_{source}.png"), dpi=300)
        # plt.show()


        print(f"✅ Processed {source} — plot saved at Prediction Plots folder.")



        plt.figure(figsize=(4.72, 3.5), dpi=300)

        # Ground truth (black)
        plt.plot(time, y_true, color="blue", label="Ground Truth", linewidth=1.5)

        # Predicted mean (red dashed)
        plt.plot(time, gamma, color="tab:red", linestyle="--", label="Predicted Mean", linewidth=1.5)

        # Uncertainty bands (green fill)
      

        plt.ylim(0, 1)
        plt.xlabel("Total Time (s)")
        plt.ylabel("SOC")
        plt.title(f"Evidential Prediction - {source}")
        plt.legend(loc="best", fontsize="small", framealpha=0.6)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Evidential_pred_{source}.png"), dpi=300)
        # plt.show()


        print(f"✅ Processed {source} — plot saved at Prediction Plots folder.")














    #     var_al = (beta / (alpha - 1)).cpu()
    #     var_ep = (beta / (nu * (alpha - 1))).cpu()
    #     var_total = torch.sqrt(var_al + var_ep)
    #     var_al_np = torch.sqrt(var_al).numpy().flatten()
    #     var_ep_np = torch.sqrt(var_ep).numpy().flatten()
    #     var_total_np = var_total.numpy().flatten()

    #     gamma_cpu = gamma.cpu().numpy()
    #     gamma_np = scaler_target.inverse_transform(gamma_cpu).flatten()
    #     pred = scaler_target.inverse_transform(gamma_cpu)

    #     aleatoric = var_al.numpy()
    #     epistemic = var_ep.numpy()
    #     total_uncertainty = aleatoric + epistemic

    #     if "gt_soc" in df_source.columns and len(df_source["gt_soc"]) > seq_length:
    #         y_actual = df_source["gt_soc"].values[seq_length:].reshape(-1, 1)
    #     else:
    #         y_actual = np.full((len(pred), 1), np.nan)

        rmse = np.sqrt(mean_squared_error(y_true, gamma))
        mae = mean_absolute_error(y_true, gamma)

        results.append({
            "source_sorter": source,
            "RMSE": rmse,
            "MAE": mae,
            "Epistemic_Uncertainty": np.mean(var)
        })

    #     df_source["Predicted_SOC"] = np.nan
    #     df_source.iloc[seq_length:seq_length + len(pred), df_source.columns.get_loc("Predicted_SOC")] = pred.flatten()
    #     df_source["Aleatoric_Uncertainty"] = np.nan
    #     df_source["Epistemic_Uncertainty"] = np.nan
    #     df_source["Total_Uncertainty"] = np.nan
    #     df_source.iloc[seq_length:seq_length + len(pred), df_source.columns.get_loc("Aleatoric_Uncertainty")] = aleatoric.flatten()
    #     df_source.iloc[seq_length:seq_length + len(pred), df_source.columns.get_loc("Epistemic_Uncertainty")] = epistemic.flatten()
    #     df_source.iloc[seq_length:seq_length + len(pred), df_source.columns.get_loc("Total_Uncertainty")] = total_uncertainty.flatten()

    #     time = df_source["Total Time"].iloc[seq_length:]

    #     # 🌿 Epistemic Uncertainty Plot
    #     plt.figure(figsize=(4.72, 3))
    #     plt.plot(time, y_actual, label="True SOC", color="blue", linewidth=1.5)
    #     plt.plot(time, pred, label="Predicted SOC", color="red", linestyle="--", linewidth=1.5)
    #     for std in range(3):
    #         plt.fill_between(
    #             time,
    #             gamma_np - std * var_ep_np,
    #             gamma_np + std * var_ep_np,
    #             alpha=0.9,
    #             facecolor="tab:green",
    #             label="Epistemic Unc." if std == 0 else None,
    #         )
    #     plt.xlabel("Time (s)")
    #     # plt.ylabel("SOC")
    #     plt.title(f"Prediction on {source}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(plots_dir, f"Epistemic_GRU_Prediction_{source}.png"), dpi=300)
    #     plt.close()

    #     # ❄️ Aleatoric Uncertainty Plot
    #     plt.figure(figsize=(4.72, 3))
    #     plt.plot(time, y_actual, label="True SOC", color="blue", linewidth=1.5)
    #     plt.plot(time, pred, label="Predicted SOC", color="red", linestyle="--", linewidth=1.5)
    #     for std in range(3):
    #         plt.fill_between(
    #             time,
    #             gamma_np - std * var_al_np,
    #             gamma_np + std * var_al_np,
    #             alpha=0.9,
    #             facecolor="tab:blue",
    #             label="Aleatoric Unc" if std == 0 else None,
    #         )
    #     plt.xlabel("Time (s)")
    #     # plt.ylabel("Uncertainty")
    #     plt.title(f"Prediction on {source}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(plots_dir, f"Aleatoric_Uncertainty_{source}.png"), dpi=300)
    #     plt.close()

    #     # 🔶 Combined Uncertainty Plot

    #     plt.figure(figsize=(4.72, 3))
    #     plt.plot(time, y_actual, label="True SOC", color="blue", linewidth=1.5)
    #     plt.plot(time, pred, label="Predicted SOC", color="red", linestyle="--", linewidth=1.5)
    #     for std in range(3):
    #         plt.fill_between(
    #             time,
    #             gamma_np - std * var_al_np,
    #             gamma_np + std * var_al_np,
    #             alpha=0.3,
    #             facecolor="tab:blue",
    #             label="Aleatoric Unc" if std == 0 else None,
    #         )

    #         plt.fill_between(
    #             time,
    #             gamma_np - std * var_ep_np,
    #             gamma_np + std * var_ep_np,
    #             alpha=0.3,
    #             facecolor="tab:green",
    #             label="Epistemic Unc." if std == 0 else None,
    #         )



        
    #     plt.xlabel("Time (s)")
    #     # plt.ylabel("Uncertainty")
    #     plt.title(f"Prediction on {source}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(plots_dir, f"Combined_Uncertainty_{source}.png"), dpi=300)
    #     plt.close()





    #     plt.figure(figsize=(4.72, 3))
    #     plt.plot(time, y_actual, label="True SOC", color="blue", linewidth=1.5)
    #     plt.plot(time, pred, label="Predicted SOC", color="red", linestyle="--", linewidth=1.5)
    #     pred_flat = pred.flatten()
    #     for std in range(3):
    #         plt.fill_between(
    #             time,
    #             pred_flat - std * var_total_np,
    #             pred_flat + std * var_total_np,
    #             alpha=0.9,
    #             facecolor="tab:orange",
    #             label="Total Unc." if std == 0 else None,
    #         )
    #     plt.xlabel("Time (s)")
    #     # plt.ylabel("Uncertainty")
    #     plt.title(f"Prediction on {source}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(plots_dir, f"Total_Uncertainty_{source}.png"), dpi=300)
    #     plt.close()



        
    #     # Round the middle point down to nearest 50 to get a clean zoom_start
        zoom_start = int(np.floor(time[len(time) // 2] / 50.0)) * 50
        zoom_end = zoom_start + 1000

        zoom_mask = (time >= zoom_start) & (time <= zoom_end)

        time_zoom = time[zoom_mask]
        pred_zoom = gamma.flatten()[zoom_mask]
        gt_zoom = y_true.flatten()[zoom_mask]
        var_zoom = var[zoom_mask]
        var_al_zoom = var_al[zoom_mask]

        plt.figure(figsize=(4.72, 3.5))
        plt.plot(time_zoom, gt_zoom, label="True SOC", color="blue")
        plt.plot(time_zoom, pred_zoom, label="Predicted SOC", color="red", linestyle="--")


        plt.fill_between(
            time_zoom,
            pred_zoom - 1.96 * var_zoom,
            pred_zoom + 1.96 * var_zoom,
            alpha=0.15,
            facecolor="green",
            label="Uncertainty Band",  # label only once
        )

        plt.ylim(.16,.3)
        plt.xlabel("Total Time (s)")
        plt.ylabel("SOC")
        plt.title(f"Evidential GRU Uncertainty - {source}")
        plt.legend(loc="best", fontsize="small", framealpha=0.6)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Evidential_zoom_{source}.png"), dpi=300)
        # plt.show()


        plt.figure(figsize=(4.72, 3.5))
        plt.plot(time_zoom, gt_zoom, label="True SOC", color="blue")
        plt.plot(time_zoom, pred_zoom, label="Predicted SOC", color="red", linestyle="--")


        plt.fill_between(
            time_zoom,
            pred_zoom - 1.96 * var_zoom,
            pred_zoom + 1.96 * var_zoom,
            alpha=0.15,
            facecolor="green",
            label="Epistemic Unc",  # label only once
        )

        plt.fill_between(
            time_zoom,
            pred_zoom - 1.96 * var_al_zoom,
            pred_zoom + 1.96 * var_al_zoom,
            alpha=0.15,
            facecolor="blue",
            label="Aleatoric Unc",  # label only once
        )

        plt.ylim(.16,.3)
        plt.xlabel("Total Time (s)")
        plt.ylabel("SOC")
        plt.title(f"Evidential GRU Uncertainty - {source}")
        plt.legend(loc="best", fontsize="small", framealpha=0.6)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Evidential_zoom_ep_al{source}.png"), dpi=300)
        # plt.show()






        df_source = df_source.iloc[seq_length:].copy()
        df_source["Predicted_SOC"] = gamma
        df_source["Uncertainty"] = var




        df_source.to_excel(os.path.join(results_dir, f"PREDICTED_SOC_{source}.xlsx"), index=False)
        print(f"✅ Processed: {source} | RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    df_metrics = pd.DataFrame(results)
    df_metrics.to_excel(os.path.join(model_output_dir, "Evidential_GRU_Evaluation_Metrics.xlsx"), index=False)
    print(f"\n📁 Evaluation complete. Results saved in: {model_output_dir}")




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

# 🔍 Load test set
if __name__ == "__main__":
    folder_path = "./test_set"
    test_data = BatteryDataProcessor(folder_path)
    df_test = test_data.load_data()
    evaluate_evidential_gru_by_source_sorter(df_test,"./saved_models/exp23")

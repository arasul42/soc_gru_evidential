import os
import re
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

def evaluate_ensemble_gru_by_source_sorter(df, model_output_dir=None, seq_length=100):
    if model_output_dir is None:
        model_output_dir = get_latest_exp_folder()

    plots_dir = os.path.join(model_output_dir, "Prediction Plots")
    results_dir = os.path.join(model_output_dir, "results")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all 5 models and their scalers
    models = []
    scalers_features = []
    scalers_target = []

    for i in range(1, 6):
        model_dir = os.path.join(model_output_dir, f"model{i}")
        model_path = os.path.join(model_output_dir, "best_models", f"model{i}_best.pth")

        model = GRUModel(input_size=3, hidden_size=256, num_layers=2, output_size=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        scaler_features = joblib.load(os.path.join(model_dir,"exp1", "scaler_features.pkl"))
        scaler_target = joblib.load(os.path.join(model_dir,"exp1", "scaler_target.pkl"))

        models.append(model)
        scalers_features.append(scaler_features)
        scalers_target.append(scaler_target)

    results = []
    all_model_metrics = []  # To hold per-model RMSE and MAE

    for source in df["source_sorter"].unique():
        df_source = df[df["source_sorter"] == source].copy()

        if len(df_source) < seq_length:
            print(f"Skipping {source}: Not enough data.")
            continue

        features = df_source[["Current", "Voltage", "Temperature"]].values
        time = df_source["Total Time"].iloc[seq_length:]
        true_soc = df_source["gt_soc"].values[seq_length:].reshape(-1, 1)

        preds_all = []
        per_model_rmse = []
        per_model_mae = []

        for idx, (model, scaler_f, scaler_t) in enumerate(zip(models, scalers_features, scalers_target)):
            features_scaled = scaler_f.transform(features)
            X = create_sequences(features_scaled, seq_length)

            if len(X) == 0:
                print(f"Skipping {source}: No sequences created.")
                continue

            X_torch = torch.tensor(X, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_scaled = model(X_torch).cpu().numpy()
            pred = scaler_t.inverse_transform(pred_scaled).flatten()

            rmse_model = np.sqrt(mean_squared_error(true_soc, pred))
            mae_model = mean_absolute_error(true_soc, pred)
            print(f" Processed: {source} with {idx} | RMSE: {rmse_model:.4f}, MAE: {mae_model:.4f}")
            per_model_rmse.append(rmse_model)
            per_model_mae.append(mae_model)
            preds_all.append(pred)

            all_model_metrics.append({
                "source_sorter": source,
                "Model": f"model{idx+1}",
                "RMSE": rmse_model,
                "MAE": mae_model
            })

        # Ensemble prediction
        ensemble_pred = np.mean(np.stack(preds_all), axis=0)
        ensemble_rmse = np.sqrt(mean_squared_error(true_soc, ensemble_pred))
        ensemble_mae = mean_absolute_error(true_soc, ensemble_pred)

        results.append({
            "source_sorter": source,
            "Ensemble_RMSE": ensemble_rmse,
            "Ensemble_MAE": ensemble_mae,
        })

        # Plot
        plt.figure(figsize=(4.72, 3))
        plt.plot(time, true_soc, label="True SOC", color="blue", linewidth=1)
        plt.plot(time, ensemble_pred, label="Ensemble Prediction", color="red", linestyle="--", linewidth=1)
        plt.xlabel("Total Time (s)")
        plt.ylabel("SOC")
        plt.ylim(0, 1)
        plt.title(f"Ensemble Prediction on {source}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Prediction_{source}.png"), dpi=300)
        plt.close()
        
        plt.figure(figsize=(4.72, 3))

        for idx, pred in enumerate(preds_all):
            plt.plot(time, pred, linewidth=0.8, linestyle='-', alpha=0.6)

        plt.plot(time, true_soc, label="True SOC", color="blue", linewidth=1.5)
        plt.plot(time, ensemble_pred, label="Ensemble Prediction", color="red", linestyle="--", linewidth=1.5)

        plt.xlabel("Total Time (s)")
        plt.ylabel("SOC")
        plt.ylim(0, 1)
        plt.title(f"Ensemble Prediction on {source}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Prediction_all_models{source}.png"), dpi=300)
        plt.close()


        
        # pred_all_min = np.min(preds_all, axis=0)
        # pred_all_max = np.max(preds_all, axis=0)

        pred_all_min = np.mean(preds_all, axis=0)- np.std(preds_all, axis=0)
        pred_all_max = np.mean(preds_all, axis=0)+ np.std(preds_all, axis=0)




        plt.figure(figsize=(4.72, 3.5))

        # Ground Truth
        plt.plot(time, true_soc, label="True SOC", color="blue", linewidth=1.5)

        # Individual model predictions
        for idx, pred in enumerate(preds_all):
            plt.plot(time, pred, linewidth=0.8, linestyle='-', alpha=0.6,)

        # Ensemble prediction
        plt.plot(time, ensemble_pred, color="red", linewidth=1.5, linestyle="--", label="Mean Prediction")

        # Fill uncertainty band
        plt.fill_between(time, pred_all_min, pred_all_max, color='green', alpha=0.2, label="Uncertainty Band")

        # Styling
        plt.title(f" Ensemble Prediction on {source}")
        plt.xlabel("Time (s)")
        plt.ylabel("SOC")
        # plt.ylim(0.2, .8)
        plt.legend(loc="best", framealpha=0.6, fontsize='small')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Total_unc_filled{source}.png"), dpi=300)
        plt.close()









        zoom_start = int(np.floor(time.iloc[len(time) // 2] / 50.0)) * 50
        zoom_end = zoom_start + 1000

        zoom_mask = (time >= zoom_start) & (time <= zoom_end)

        time_zoom = time[zoom_mask]
        gt_zoom = true_soc.flatten()[zoom_mask]
        ensemble_zoom = ensemble_pred[zoom_mask]

        plt.figure(figsize=(4.72, 3.5))
        plt.plot(time_zoom, gt_zoom, label="True SOC", color="blue")
        plt.plot(time_zoom, ensemble_zoom, label="Ensemble Predicted SOC", color="red", linestyle="--")

        plt.title(f"Prediction on {source}")
        plt.xlabel("Time (s)")
        plt.ylabel("SOC")
        plt.ylim(0, .8)
        plt.legend(loc="best", framealpha=0.5, fontsize='small')
        plt.grid(True)

        # Use the same zoom_start as xtick_start
        xtick_start = zoom_start
        xtick_end = int(np.ceil(time_zoom.iloc[-1] / 50.0)) * 50
        plt.xticks(np.arange(xtick_start, xtick_end + 1, 200))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Zoomed_Total_{source}.png"), dpi=300)
        plt.close()


        plt.figure(figsize=(4.72, 3.5))

        # Ground truth
        plt.plot(time_zoom, gt_zoom, label="True SOC", color="blue", linewidth=1.5)

        # All 5 model predictions - thin solid lines
        for idx, pred in enumerate(preds_all):
            plt.plot(time_zoom, pred[zoom_mask], linewidth=0.8, linestyle='-', alpha=0.6)

        # Ensemble prediction - thick solid line
        plt.plot(time_zoom, ensemble_zoom, color="red", linewidth=1.5, linestyle="--", label="Mean Prediction")

        # Plot settings
        plt.title(f"Prediction on {source}")
        plt.xlabel("Time (s)")
        plt.ylabel("SOC")
        plt.ylim(0, .8)
        plt.legend(loc="best", framealpha=0.6, fontsize='small')
        plt.grid(True)

        xtick_start = zoom_start
        xtick_end = int(np.ceil(time_zoom.iloc[-1] / 50.0)) * 50
        plt.xticks(np.arange(xtick_start, xtick_end + 1, 200))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Zoomed_Total_ensemble_{source}.png"), dpi=300)
        plt.close()

        # Get zoomed predictions from all models
        preds_all_zoomed = np.stack(preds_all)[:, zoom_mask]  # Shape: (5, N_zoom)

        # Calculate min/max for uncertainty band
        # pred_min = np.min(preds_all_zoomed, axis=0)
        # pred_max = np.max(preds_all_zoomed, axis=0)


        pred_min = np.mean(preds_all_zoomed, axis=0)- np.std(preds_all_zoomed, axis=0)
        pred_max = np.mean(preds_all_zoomed, axis=0)+ np.std(preds_all_zoomed, axis=0)

        plt.figure(figsize=(4.72, 3.5))

        # Ground Truth
        plt.plot(time_zoom, gt_zoom, label="True SOC", color="blue", linewidth=2)

        # Individual model predictions
        for idx, pred in enumerate(preds_all):
            plt.plot(time_zoom, pred[zoom_mask], linewidth=0.8, linestyle='-', alpha=0.5, label=f"Model {idx+1}")

        # Ensemble prediction
        plt.plot(time_zoom, ensemble_zoom, color="red", linewidth=1.5, linestyle="--", label="Mean Prediction")

        # Fill uncertainty band
        plt.fill_between(time_zoom, pred_min, pred_max, color='green', alpha=0.2, label="Uncertainty Band")

        # Styling
        plt.title(f" Ensemble Prediction on {source}")
        plt.xlabel("Time (s)")
        plt.ylabel("SOC")
        # plt.ylim(0.2, .8)
        plt.legend(loc="best", framealpha=0.6, fontsize='small')
        plt.grid(True)

        xtick_start = zoom_start
        xtick_end = int(np.ceil(time_zoom.iloc[-1] / 50.0)) * 50
        plt.xticks(np.arange(xtick_start, xtick_end + 1, 200))

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"Zoomed_Total_unc_filled{source}.png"), dpi=300)
        plt.close()









        # Save data
        df_out = df_source.iloc[seq_length:].copy()
        df_out["Ensemble_Predicted_SOC"] = ensemble_pred
        df_out.to_excel(os.path.join(results_dir, f"PREDICTED_SOC_{source}.xlsx"), index=False)

        print(f"✅ Processed: {source} | Ensemble RMSE: {ensemble_rmse:.4f}, MAE: {ensemble_mae:.4f}")

    df_ensemble_metrics = pd.DataFrame(results)
    df_individual_metrics = pd.DataFrame(all_model_metrics)

    output_path = os.path.join(model_output_dir, "Ensemble_GRU_Evaluation_Metrics.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df_ensemble_metrics.to_excel(writer, sheet_name="Ensemble_Summary", index=False)
        df_individual_metrics.to_excel(writer, sheet_name="Model_Wise_Metrics", index=False)

    print(f"\n📁 Evaluation complete. Results saved at: {output_path}")



# === RUNNING ON TEST SET ===
if __name__ == "__main__":
    folder_path = "./test_set"
    test_data = BatteryDataProcessor(folder_path)
    df_test = test_data.load_data()
    evaluate_ensemble_gru_by_source_sorter(df_test)

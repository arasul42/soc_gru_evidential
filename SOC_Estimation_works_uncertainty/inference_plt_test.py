import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load predicted SOC file ===
file_path = "1PREDICTED_SOC_FUDS_80SOC_0C.xlsx"
df = pd.read_excel(file_path)

# === Clean and convert ===
df["Total Time"] = pd.to_numeric(df["Total Time"], errors="coerce")
df["Predicted_SOC"] = pd.to_numeric(df["Predicted_SOC"], errors="coerce")
df["gt_soc"] = pd.to_numeric(df["gt_soc"], errors="coerce")
df["Epistemic_Uncertainty"] = pd.to_numeric(df["Epistemic_Uncertainty"], errors="coerce")
df["Aleatoric_Uncertainty"] = pd.to_numeric(df["Aleatoric_Uncertainty"], errors="coerce")

# === Drop missing rows ===
df_clean = df.dropna(subset=[
    "Total Time", "Predicted_SOC", "gt_soc",
    "Epistemic_Uncertainty", "Aleatoric_Uncertainty"
])

# === Compute standard deviations ===
time = df_clean["Total Time"].to_numpy()
pred = df_clean["Predicted_SOC"].to_numpy()
gt_soc = df_clean["gt_soc"].to_numpy()
epistemic_std = np.sqrt(df_clean["Epistemic_Uncertainty"].to_numpy())
aleatoric_std = np.sqrt(df_clean["Aleatoric_Uncertainty"].to_numpy())

# === Define zoom range ===
start_time = 24000
end_time = start_time + 400
mask = (time >= start_time) & (time <= end_time)

# === Extract zoomed-in slice ===
time_zoom = time[mask]
pred_zoom = pred[mask]
gt_zoom = gt_soc[mask]
epistemic_std_zoom = epistemic_std[mask]
aleatoric_std_zoom = aleatoric_std[mask]

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(time_zoom, gt_zoom, 'b-', label='True SOC')
plt.plot(time_zoom, pred_zoom, 'r--', label='Predicted SOC')

# Epistemic band
plt.fill_between(
    time_zoom,
    pred_zoom - 1.96 * epistemic_std_zoom,
    pred_zoom + 1.96 * epistemic_std_zoom,
    alpha=0.3,
    color='green',
    label='Epistemic 95% CI'
)

# Aleatoric band
plt.fill_between(
    time_zoom,
    pred_zoom - 1.96 * aleatoric_std_zoom,
    pred_zoom + 1.96 * aleatoric_std_zoom,
    alpha=0.2,
    color='blue',
    label='Aleatoric 95% CI'
)

plt.xlabel("Total Time (s)")
plt.ylabel("SOC")
plt.title(f"Zoomed-in SOC Prediction with Uncertainty ({start_time}s to {end_time}s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

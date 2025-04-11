import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Your noise function (with reproducibility)
def add_salt_pepper_noise(df, prob=0.05, magnitude=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility

    df_noisy = df.copy()
    for col in df.columns:
        mask = np.random.rand(len(df)) < prob
        spikes = np.random.choice([-magnitude, magnitude], size=len(df))
        df_noisy[col] += mask * spikes
    return df_noisy

def add_input_noise(df, std_dict, seed=None):
    df_noisy = df.copy()
    if seed is not None:
        np.random.seed(seed)
    for col, std in std_dict.items():
        if col in df_noisy.columns:
            noise = np.random.normal(loc=0.0, scale=std, size=len(df_noisy))
            df_noisy[col] += noise
    return df_noisy

# Generate example data
x = np.linspace(0, 10, 500)
y = np.zeros_like(x)
df = pd.read_excel('1PREDICTED_SOC_FUDS_80SOC_0C.xlsx')

df["Total Time"] = pd.to_numeric(df["Total Time"], errors="coerce")


noise_std = {"Current": 0.5, "Voltage": 0.05, "Temperature": 5}
# Apply noise
# df_noisy = add_salt_pepper_noise(df, prob=0.05, magnitude=1.0, seed=42)
df_noisy = add_input_noise(df, noise_std, seed=42)
# Plot original vs noisy
plt.figure(figsize=(4.72, 3.5))
tab10_colors = plt.cm.tab10.colors
current_color = tab10_colors[0]  # Blue
voltage_color = tab10_colors[1]  # Orange
soc_color = tab10_colors[3]      # Green

# Create directory if it doesn't exist
output_dir = './plots/noise/'
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Current With Added Noise
plt.plot(df["Total Time"], df["Current"], label='Original Current', linewidth=2, color=current_color)
plt.plot(df_noisy["Total Time"], df_noisy["Current"], label='Noisy Current', linestyle='--', color=current_color, linewidth=1)
plt.title('Current With Added Noise on FUDS_80SOC_0C')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'current_with_noise.png'))
plt.close()

# Zoomed-in plot for 100 timesteps starting at the middle of time
zoom_start = int(np.floor(df["Total Time"].iloc[len(df) // 2] / 50.0)) * 50
zoom_end = zoom_start + 100
zoom_mask = (df["Total Time"] >= zoom_start) & (df["Total Time"] <= zoom_end)

time_zoom = df["Total Time"][zoom_mask]
current_zoom = df["Current"][zoom_mask]
current_noisy_zoom = df_noisy["Current"][zoom_mask]

plt.figure(figsize=(4.72, 3.5))
plt.plot(time_zoom, current_zoom, label="Original Current", color=current_color)
plt.plot(time_zoom, current_noisy_zoom, label="Noisy Current", color="black", linestyle="--", linewidth=1)
plt.title("Current With Added Noise on FUDS_80SOC_0C")
plt.xlabel("Time (s)")
plt.ylabel("Current")
plt.legend(loc="best", framealpha=0.5, fontsize='small')
plt.grid(True)
xtick_start = zoom_start
xtick_end = int(np.ceil(time_zoom.iloc[-1] / 50.0)) * 50
plt.xticks(np.arange(xtick_start, xtick_end + 1, 50))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'current_zoom_with_noise.png'))
plt.close()

# Plot 2: Voltage With Added Noise
plt.figure(figsize=(4.72, 3.5))
plt.plot(time_zoom, df["Voltage"][zoom_mask], label="Original Voltage", color=voltage_color)
plt.plot(time_zoom, df_noisy["Voltage"][zoom_mask], label="Noisy Voltage", color="black", linestyle="--", linewidth=1)
plt.title("Voltage With Added Noise on FUDS_80SOC_0C")
plt.xlabel("Time (s)")
plt.ylabel("Voltage")
plt.legend(loc="best", framealpha=0.5, fontsize='small')
plt.grid(True)
plt.xticks(np.arange(xtick_start, xtick_end + 1, 50))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'voltage_zoom_with_noise.png'))
plt.close()

# Plot 3: Zero Signal With Added Noise
zero_signal = np.zeros_like(x)
zero_signal_df = pd.DataFrame({"Time": x, "Signal": zero_signal})
zero_signal_noisy_df = add_input_noise(zero_signal_df, {"Signal": 0.5}, seed=42)

plt.figure(figsize=(4.72, 3.5))
plt.plot(zero_signal_df["Time"], zero_signal_df["Signal"], label="Zero Signal", linewidth=1, color="black")
plt.plot(zero_signal_noisy_df["Time"], zero_signal_noisy_df["Signal"], label="Noisy Zero Signal", linestyle="--", color="black", linewidth=1)
plt.title("Added Noise")
plt.xlabel("Time (s)")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'zero_signal_with_noise.png'))
plt.close()

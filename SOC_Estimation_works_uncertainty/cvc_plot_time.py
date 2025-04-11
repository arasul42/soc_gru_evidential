import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib



# Load the dataset
# file_path = "/home/arasul42@tntech.edu/mecc/SOC_Estimation_works/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_50SOC.xls"  # Update with correct filename if needed
file_path = "SOC_Estimation_works/training_Set/02_27_2016_SP20-2_0C_BJDST_80SOC.xls"  # Update with correct filename if needed
sheet_name = "Channel_1-006"

# Read the dataset
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=0)

# Print the dataset to inspect its contents
print(df.head())

# Rename relevant columns
time_column = "Date_Time"
current_column = "Current(A)"
voltage_column = "Voltage(V)"
discharge_capacity_column = "Discharge_Capacity(Ah)"
soc_gt_column = "SOC"

# Convert time column to datetime
df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

# Convert numerical columns to float
df[current_column] = pd.to_numeric(df[current_column], errors='coerce')
df[voltage_column] = pd.to_numeric(df[voltage_column], errors='coerce')
df[discharge_capacity_column] = pd.to_numeric(df[discharge_capacity_column], errors='coerce')
df[soc_gt_column] = pd.to_numeric(df[soc_gt_column], errors='coerce')

# Compute timestep (elapsed time in seconds)
df["Time_Step"] = (df[time_column] - df[time_column].iloc[0]).dt.total_seconds()

# Compute SOC using discharge capacity as loss (Initial SOC = 100%, Capacity = 2000 mAh)
nominal_capacity = 2.0  # 2000 mAh = 2.0 Ah


# Ensure SOC remains between 0 and 1
df["SOC"] = df["SOC"].clip(0, 1)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(df["Time_Step"], df[voltage_column], label="Voltage (V)", color='b')
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(df["Time_Step"], df[current_column], label="Current (A)", color='r')
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(df["Time_Step"], df["SOC"], label="SOC", color='g')
plt.xlabel("Time Step (s)")
plt.ylabel("SOC")
plt.legend()
plt.grid()

plt.suptitle("Voltage, Current, and SOC over Time Steps")
plt.tight_layout()
plt.show()
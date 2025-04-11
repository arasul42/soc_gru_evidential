import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/home/arasul42@tntech.edu/mecc/battery-state-estimation/battery-state-estimation/data/LG 18650HG2 Li-ion Battery Data/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC/551_LA92.csv"
df = pd.read_csv(file_path, skiprows=28)

# Rename relevant columns for easier reference
time_column = "Step Time"
cycle_column = "Cycle"
current_column = "Current"
voltage_column = "Voltage"
capacity_column = "Capacity"

# Convert Step Time to total seconds
df[time_column] = pd.to_timedelta(df[time_column]).dt.total_seconds()
df[current_column]=pd.to_numeric(df[current_column], errors='coerce')
df[voltage_column]=pd.to_numeric(df[voltage_column], errors='coerce')
df[cycle_column]=pd.to_numeric(df[cycle_column], errors='coerce')
df[capacity_column]=pd.to_numeric(df[capacity_column], errors='coerce')


# Ensure first step time is not NaN
df[time_column] = df[time_column].fillna(0)

# Compute accumulated total time
df["Total Time"] = 0
total_time = 0
previous_cycle = df[cycle_column].iloc[0]

for i in range(len(df)):
    if i == 0:
        df.at[i, "Total Time"] = df.at[i, time_column]
    else:
        if df.at[i, cycle_column] == previous_cycle:
            total_time += df.at[i, time_column] - df.at[i-1, time_column]
        else:
            total_time += df.at[i, time_column]  # Do not reset on cycle change
            previous_cycle = df.at[i, cycle_column]

        df.at[i, "Total Time"] = total_time

# Plot Voltage and Current against Accumulated Total Time
plt.figure(figsize=(12, 6))
plt.plot(df["Total Time"], df[voltage_column], label="Voltage (V)", color="b")
plt.plot(df["Total Time"], df[current_column], label="Current (A)", color="r", linestyle="dashed")
plt.plot(df["Total Time"], df[capacity_column], label="Capacity (Ah)", color="g",linewidth=2)
plt.xlabel("Accumulated Total Time (s)")
plt.ylabel("Voltage (V) / Current (A)")
plt.title("Voltage and Current vs Accumulated Total Time")
plt.legend()
plt.grid()
plt.show()

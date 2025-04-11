import pandas as pd
import matplotlib.pyplot as plt

filepath="/home/arasul42@tntech.edu/mecc/SOC_Predictions_New_Dataset.xlsx"
sheetname="Sheet1"

df = pd.read_excel(filepath, sheet_name=sheetname,skiprows=0)

print(df.columns)

time_step="Test_Time(s)"
predicted_soc="Predicted_SOC"
actual_soc="SOC"

df[time_step] = pd.to_numeric(df[time_step], errors='coerce')  
df[predicted_soc] = pd.to_numeric(df[predicted_soc], errors='coerce')
df[actual_soc] = pd.to_numeric(df[actual_soc], errors='coerce')

plt.figure(figsize=(12, 6))
plt.plot(df[time_step], df[predicted_soc], label="Predicted SOC", color='b')
plt.plot(df[time_step], df[actual_soc], label="Actual SOC", color='r')
plt.xlabel("Time Step (s)")
plt.ylabel("SOC")
plt.title("Predicted vs Actual SOC over Time Steps")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

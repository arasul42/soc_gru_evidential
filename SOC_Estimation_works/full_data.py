import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Load the dataset
file_path = "/home/arasul42@tntech.edu/mecc/SOC_Estimation_works/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_80SOC.xls"
sheet_name = "Channel_1-006"
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=0)

# Rename relevant columns
time_column = "Date_Time"
current_column = "Current(A)"
voltage_column = "Voltage(V)"
discharge_capacity_column = "Discharge_Capacity(Ah)"

# Convert time column to datetime
df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

# Convert numerical columns to float
df[current_column] = pd.to_numeric(df[current_column], errors='coerce')
df[voltage_column] = pd.to_numeric(df[voltage_column], errors='coerce')
df[discharge_capacity_column] = pd.to_numeric(df[discharge_capacity_column], errors='coerce')

# Compute timestep (elapsed time in seconds)
df["Time_Step"] = (df[time_column] - df[time_column].iloc[0]).dt.total_seconds()

# Compute SOC using discharge capacity as loss (Initial SOC = 100%, Capacity = 2000 mAh)
nominal_capacity = 2.0  # 2000 mAh = 2.0 Ah
df["SOC"] = 1.0 - (df[discharge_capacity_column] / nominal_capacity)

# Ensure SOC remains between 0 and 1
df["SOC"] = df["SOC"].clip(0, 1)

# Step 1: Prepare dataset with Current and Voltage as inputs
def create_sequences(features, target, seq_length):
    sequences, targets = [], []
    for i in range(len(features) - seq_length):
        sequences.append(features[i:i + seq_length])
        targets.append(target[i + seq_length])
    return np.array(sequences), np.array(targets)

# Define sequence length
seq_length = 100

# Use the entire dataset for training
features = df[["Current(A)", "Voltage(V)"]].values
soc = df["SOC"].values.reshape(-1, 1)

# Normalize data (MinMax Scaling)
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler_features.fit_transform(features)
soc_scaled = scaler_target.fit_transform(soc)

# Create sequences for training
X, y = create_sequences(features_scaled, soc_scaled, seq_length)

# Convert to PyTorch tensors
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Step 2: Define the GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take last timestep's output
        return out

# Define model parameters
input_size = 2  # Current & Voltage as input features
hidden_size = 150  # Number of hidden neurons
num_layers = 2  # Number of GRU layers
output_size = 1  # Predicting SOC

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)

X_torch = X_torch.to(device)
y_torch = y_torch.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the Model
num_epochs = 500
batch_size = 60

train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Save the model's state_dict (weights)
torch.save(model.state_dict(), "gru_soc_model.pth")
print("Model saved successfully!")

# Save the scalers for future inference
joblib.dump(scaler_features, "scaler_features.pkl")
joblib.dump(scaler_target, "scaler_target.pkl")
print("Scalers saved successfully!")

# Step 4: Make Predictions on Full Dataset
model.eval()
with torch.no_grad():
    y_pred_torch = model(X_torch)

# Convert predictions back to original scale
y_pred = scaler_target.inverse_transform(y_pred_torch.cpu().numpy())  # Move to CPU
y_true = scaler_target.inverse_transform(y_torch.cpu().numpy())

# Compute RMSE & MAE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# Display RMSE and MAE results
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Step 5: Plot Results
plt.figure(figsize=(10, 5))
plt.plot(y_true, label="True SOC", color="b")
plt.plot(y_pred, label="Predicted SOC", color="r", linestyle="dashed")
plt.xlabel("Time Step Index")
plt.ylabel("SOC")
plt.ylim(0, 1)
plt.title("GRU Model - True vs Predicted SOC (Trained on Full Dataset)")
plt.legend()
plt.grid(True)
plt.savefig("GRU_Model_True_vs_Predicted_SOC_Full_Dataset.png")
plt.show()

# Plot Training Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Full Dataset)")
plt.savefig("Training_Loss_Curve_Full_Dataset.png")
plt.show()
